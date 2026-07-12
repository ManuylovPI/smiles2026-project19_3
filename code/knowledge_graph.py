import json
import spacy
import re

from dataclasses import dataclass
from fastcoref import FCoref
from openai import OpenAI


@dataclass
class Entity:
    id: int
    text: str

@dataclass
class EntityMention:
    entity: Entity
    word_span: tuple[int, int]   # (start_word, end_word)
    char_span: tuple[int, int]   # (start_char, end_char)

@dataclass
class Document:
    text: str
    entities: list[Entity]
    entity_mentions: list[EntityMention]

@dataclass
class Paragraph:
    text: str
    entity_mentions: list[EntityMention]

@dataclass
class Relation:
    head: int
    relation: str
    tail: int


def _mark_entities(paragraph: Paragraph) -> str:
    """
    Surround entity mentions with XML-like tags.

    Parameters
    ----------
    paragraph
        Paragraph containing the text and entity mentions.

    Returns
    -------
    str
        Paragraph with entity tags inserted.
    """

    marked = paragraph.text

    mentions = sorted(
        paragraph.entity_mentions,
        key=lambda m: m.char_span[0],
        reverse=True
    )

    for mention in mentions:
        start, end = mention.char_span

        marked = (
            marked[:start]
            + f'<entity id="{mention.entity.id}">'
            + marked[start:end]
            + "</entity>"
            + marked[end:]
        )

    return marked


def _split_into_paragraphs(document: Document) -> list[Paragraph]:
    """
    Split a document into paragraphs and assign entity mentions
    to the corresponding paragraphs.

    Parameters
    ----------
    document
        Document with extracted entity mentions.

    Returns
    -------
    list[Paragraph]
    """

    paragraphs = []
    text = document.text
    offset = 0

    for paragraph_text in text.split("\n\n"):

        paragraph_start = offset
        paragraph_end = paragraph_start + len(paragraph_text)

        mentions = []
        for mention in document.entity_mentions:
            start, end = mention.char_span
            if paragraph_start <= start and end <= paragraph_end:

                mentions.append(
                    EntityMention(
                        entity=mention.entity,
                        word_span=mention.word_span,
                        char_span=(
                            start - paragraph_start,
                            end - paragraph_start
                        )
                    )
                )

        paragraphs.append(
            Paragraph(
                text=paragraph_text,
                entity_mentions=mentions
            )
        )
        offset = paragraph_end + 2

    return paragraphs


def extract_entities(
    text: str,
    model=None,
    nlp=None
) -> Document:
    """
    Extract entities from a document using NER + coreference resolution.

    Returns
    -------
    Document
        Document containing unique entities and all their mentions.
    """

    if model is None:
        model = FCoref(device="cpu")

    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    preds = model.predict(texts=[text])

    # Use spans instead of strings
    clusters = preds[0].get_clusters(as_strings=False)
    doc = nlp(text)
    result = []
    all_indices_in_clusters = set()

    # Coreference clusters
    for cluster_id, cluster in enumerate(clusters):
        for start_char, end_char in cluster:
            span = doc.char_span(
                start_char,
                end_char,
                alignment_mode="expand"
            )

            if span is None:
                continue

            indices = tuple(token.i for token in span)
            result.append({
                "cluster_id": cluster_id,
                "text": span.text,
                "indices": indices,
                "char_span": (start_char, end_char)
            })
            all_indices_in_clusters.update(indices)

    # Named entities
    ner_entities = []
    for ent in doc.ents:
        indices = tuple(token.i for token in ent)
        ner_entities.append({
            "text": ent.text,
            "indices": indices,
            "char_span": (ent.start_char, ent.end_char)
        })

    #Remove entities already covered by clusters
    remaining_ner = []

    for entity in ner_entities:
        if not set(entity["indices"]).issubset(all_indices_in_clusters):
            remaining_ner.append(entity)

    # Add singleton entities
    next_cluster_id = len(set(item["cluster_id"] for item in result))
    for entity in remaining_ner:
        result.append({
            "cluster_id": next_cluster_id,
            "text": entity["text"],
            "indices": entity["indices"],
            "char_span": entity["char_span"]
        })
        next_cluster_id += 1

    # Renumber entity ids
    unique_cluster_ids = sorted(
        set(item["cluster_id"] for item in result)
    )
    cluster_mapping = {
        old: new
        for new, old in enumerate(unique_cluster_ids)
    }

    # Create unique Entity objects
    entities = {}
    for item in result:
        entity_id = cluster_mapping[item["cluster_id"]]
        if entity_id not in entities:
            entities[entity_id] = Entity(
                id=entity_id,
                text=item["text"]
            )

    # Create EntityMention objects
    mentions = []
    for item in result:
        indices = item["indices"]
        if len(indices) == 0:
            continue

        mentions.append(
            EntityMention(
                entity=entities[cluster_mapping[item["cluster_id"]]],
                word_span=(indices[0], indices[-1]),
                char_span=item["char_span"]
            )
        )

    return Document(
        text=text,
        entities=list(entities.values()),
        entity_mentions=mentions
    )


def extract_relations(
    paragraph: Paragraph,
    client: OpenAI,
    model: str = "llama3.1:8b"
) -> list[dict]:
    """
    Extract directed semantic relations between entities
    occurring in a paragraph.
    """

    marked_paragraph = _mark_entities(paragraph)

    unique_entities = {}

    for mention in paragraph.entity_mentions:
        unique_entities[mention.entity.id] = mention.entity.text

    entity_description = "\n".join(
        f"Entity ID {entity_id}: {text}"
        for entity_id, text in sorted(unique_entities.items())
    )

    prompt = f"""
You are an expert in Relation Extraction.

Extract ONLY explicit directed semantic relations between the marked entities.

Rules:

- Use ONLY entities marked with <entity id="...">.
- Use ONLY entity IDs listed below.
- Do NOT infer implicit relations.
- Do NOT invent entities.
- Ignore unrelated entities.
- Relations are directed.
- If no relations exist, return [].

Your response MUST consist ONLY of a JSON array.

Valid outputs:

[]

or

[
    {{
        "head": 0,
        "relation": "developed",
        "tail": 1
    }}
]

Do not output markdown.
Do not output explanations.
Do not output any text before or after the JSON.

Paragraph:

{marked_paragraph}

Entities:

{entity_description}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You extract directed semantic relations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    # Sometimes local LLMs append explanations after the JSON.
    match = re.search(r"\[[\s\S]*?\]", content)

    if match is None:
        raise ValueError(
            f"Model did not return a valid JSON array.\n\nResponse:\n{content}"
        )

    relations_json = json.loads(match.group())

    return [
        Relation(
            head=item["head"],
            relation=item["relation"],
            tail=item["tail"]
        )
        for item in relations_json
    ]


def build_graph(
    text: str,
    client: OpenAI,
    model: str = "llama3.1:8b"
) -> tuple[Document, list[Relation]]:
    """
    Build a knowledge graph from a document.

    Parameters
    ----------
    text
        Input document.

    client
        OpenAI-compatible client.

    model
        Relation extraction model.

    Returns
    -------
    tuple[Document, list[Relation]]

        document
            Document with extracted entities.

        relations
            Directed relations extracted from all paragraphs.
    """

    # Load models
    coref_model = FCoref(device="cpu")
    nlp = spacy.load("en_core_web_sm")

    # Extract entities
    document = extract_entities(
        text=text,
        model=coref_model,
        nlp=nlp
    )

    # Split document into paragraphs
    paragraphs = _split_into_paragraphs(document)

    # Extract relations
    relations: list[Relation] = []

    for paragraph in paragraphs:
        paragraph_relations = extract_relations(
            paragraph=paragraph,
            client=client,
            model=model
        )

        relations.extend(paragraph_relations)

    return document, relations