import json
import re

import spacy

from dataclasses import dataclass
from fastcoref import FCoref


@dataclass
class Entity:
    id: int
    text: str


@dataclass
class EntityMention:
    entity: Entity
    word_span: tuple[int, int]
    char_span: tuple[int, int]


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


def _mark_entities(
    paragraph: Paragraph,
) -> str:
    """
    Surround entity mentions with XML-like tags.
    """

    marked = paragraph.text

    mentions = sorted(
        paragraph.entity_mentions,
        key=lambda m: m.char_span[0],
        reverse=True,
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


def _split_into_paragraphs(
    document: Document,
) -> list[Paragraph]:
    """
    Split document into paragraphs and assign
    entity mentions to corresponding paragraphs.
    """

    paragraphs = []

    text = document.text

    offset = 0

    for paragraph_text in text.split(
        "\n\n"
    ):

        paragraph_start = offset

        paragraph_end = (
            paragraph_start
            + len(paragraph_text)
        )

        mentions = []

        for mention in (
            document.entity_mentions
        ):

            start, end = (
                mention.char_span
            )

            if (
                paragraph_start
                <= start
                and end
                <= paragraph_end
            ):

                mentions.append(
                    EntityMention(
                        entity=mention.entity,
                        word_span=mention.word_span,
                        char_span=(
                            start
                            - paragraph_start,
                            end
                            - paragraph_start,
                        ),
                    )
                )

        paragraphs.append(
            Paragraph(
                text=paragraph_text,
                entity_mentions=mentions,
            )
        )

        offset = (
            paragraph_end + 2
        )

    return paragraphs


def extract_entities(
    text: str,
    model=None,
    nlp=None,
) -> Document:
    """
    Extract entities using FastCoref + spaCy NER.
    """

    if model is None:

        model = FCoref(
            device="cpu"
        )

    if nlp is None:

        nlp = spacy.load(
            "en_core_web_sm"
        )

    preds = model.predict(
        texts=[text]
    )

    clusters = (
        preds[0]
        .get_clusters(
            as_strings=False
        )
    )

    doc = nlp(
        text
    )

    result = []

    all_indices_in_clusters = set()

    for cluster_id, cluster in enumerate(
        clusters
    ):

        for start_char, end_char in cluster:

            span = doc.char_span(
                start_char,
                end_char,
                alignment_mode="expand",
            )

            if span is None:
                continue

            indices = tuple(
                token.i
                for token in span
            )

            result.append(
                {
                    "cluster_id": cluster_id,
                    "text": span.text,
                    "indices": indices,
                    "char_span": (
                        start_char,
                        end_char,
                    ),
                }
            )

            all_indices_in_clusters.update(
                indices
            )

    ner_entities = []

    for ent in doc.ents:

        indices = tuple(
            token.i
            for token in ent
        )

        ner_entities.append(
            {
                "text": ent.text,
                "indices": indices,
                "char_span": (
                    ent.start_char,
                    ent.end_char,
                ),
            }
        )

    remaining_ner = []

    for entity in ner_entities:

        if not set(
            entity["indices"]
        ).issubset(
            all_indices_in_clusters
        ):

            remaining_ner.append(
                entity
            )

    existing_cluster_ids = {
        item["cluster_id"]
        for item in result
    }

    if existing_cluster_ids:

        next_cluster_id = (
            max(
                existing_cluster_ids
            )
            + 1
        )

    else:

        next_cluster_id = 0

    for entity in remaining_ner:

        result.append(
            {
                "cluster_id": next_cluster_id,
                "text": entity["text"],
                "indices": entity["indices"],
                "char_span": entity["char_span"],
            }
        )

        next_cluster_id += 1

    unique_cluster_ids = sorted(
        {
            item["cluster_id"]
            for item in result
        }
    )

    cluster_mapping = {
        old: new
        for new, old in enumerate(
            unique_cluster_ids
        )
    }

    entities = {}

    for item in result:

        entity_id = cluster_mapping[
            item["cluster_id"]
        ]

        if entity_id not in entities:

            entities[
                entity_id
            ] = Entity(
                id=entity_id,
                text=item["text"],
            )

    mentions = []

    for item in result:

        indices = item[
            "indices"
        ]

        if len(indices) == 0:
            continue

        entity_id = cluster_mapping[
            item["cluster_id"]
        ]

        mentions.append(
            EntityMention(
                entity=entities[
                    entity_id
                ],
                word_span=(
                    indices[0],
                    indices[-1],
                ),
                char_span=item[
                    "char_span"
                ],
            )
        )

    return Document(
        text=text,
        entities=list(
            entities.values()
        ),
        entity_mentions=mentions,
    )


def _prepare_relation_request(
    document: Document,
    paragraphs: list[Paragraph],
    custom_id: str,
) -> dict:

    marked_paragraphs = []

    for paragraph_index, paragraph in enumerate(
        paragraphs
    ):

        marked_paragraph = _mark_entities(
            paragraph
        )

        marked_paragraphs.append(
            f"Paragraph {paragraph_index}:\n"
            f"{marked_paragraph}"
        )

    marked_text = "\n\n".join(
        marked_paragraphs
    )

    entity_description = "\n".join(
        f"Entity ID {entity.id}: {entity.text}"
        for entity in document.entities
    )

    system_prompt = """
You are an expert in extracting structured semantic relations from text.

Your task is to identify explicit factual relations between entities that
have been marked in the input text.

You must be precise, conservative, and follow the direction of the relation
exactly as stated in the text.
"""

    user_prompt = f"""
Extract all explicit semantic relations between the marked entities.

IMPORTANT:

1. Use ONLY the entities provided in the entity list below.
2. Use ONLY the integer entity IDs from that list.
3. Do NOT create new entities.
4. Do NOT use entity names as head or tail. Use their integer IDs.
5. Do NOT infer facts that are not explicitly stated or directly expressed
   by the text.
6. Do NOT use outside knowledge.
7. Extract relations from all paragraphs.
8. Return each distinct relation only once.
9. Relations are directed.
10. The HEAD is the entity that the relation describes.
11. The TAIL is the entity that fills the relation's value.
12. Do not determine direction only from grammatical subject/object.
13. For example:
    "Steve Bright is the creator of Bananaman."
    means:
    Bananaman --creator--> Steve Bright
14. Do not copy the wording of the sentence as the relation name.
15. Normalize obvious grammatical variants to a concise relation name.
16. If the text explicitly states a relation involving an entity and a
    date, number, place, organization, or other marked entity, extract it
    when that entity is marked.
17. If no valid relation exists between the marked entities, return [].

Examples of relation direction:

"The film was produced by Tom Simon."
Correct:
{{"head": FILM_ID, "relation": "producer", "tail": TOM_SIMON_ID}}

"Steve Bright is the creator of Bananaman."
Correct:
{{"head": BANANAMAN_ID, "relation": "creator", "tail": STEVE_BRIGHT_ID}}

"Wharton Tiers was born on January 1, 1953."
Correct:
{{"head": WHARTON_TIERS_ID, "relation": "birthDate", "tail": DATE_ID}}

"The album was recorded in the United States."
Correct:
{{"head": ALBUM_ID, "relation": "recordedIn", "tail": USA_ID}}

"The film includes music composed by Jamie Lawrence."
Correct:
{{"head": FILM_ID, "relation": "musicComposer", "tail": JAMIE_LAWRENCE_ID}}

"Abraham A. Ribicoff was married to Ruth Ribicoff."
Correct:
{{"head": RIBICOFF_ID, "relation": "spouse", "tail": RUTH_ID}}

Return ONLY a JSON array.

The JSON objects MUST have exactly these fields:

[
    {{
        "head": integer,
        "relation": string,
        "tail": integer
    }}
]

If there are no relations, return:

[]

Do not output markdown.
Do not output explanations.
Do not output any text before or after the JSON.

DOCUMENT:

{marked_text}

ENTITY LIST:

{entity_description}
"""

    return {
        "custom_id": custom_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": 0.0,
    }


def _parse_relation_response(
    content: str,
) -> list[dict]:
    """
    Parse GPT response into raw relation dictionaries.
    """

    if not content:
        return []

    content = content.strip()

    if content.startswith(
        "```"
    ):

        lines = (
            content.splitlines()
        )

        lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):

            lines = lines[:-1]

        content = "\n".join(
            lines
        ).strip()

    match = re.search(
        r"\[[\s\S]*\]",
        content,
    )

    if match is None:

        print(
            "Warning: model did not return "
            "a valid JSON array."
        )

        print(
            content
        )

        return []

    try:

        relations_json = json.loads(
            match.group()
        )

    except json.JSONDecodeError as error:

        print(
            f"Warning: invalid JSON: {error}"
        )

        print(
            content
        )

        return []

    if not isinstance(
        relations_json,
        list,
    ):

        return []

    return relations_json


def _relations_from_response(
    document: Document,
    response: str,
) -> list[Relation]:
    """
    Convert one document response into Relation objects.
    """

    raw_relations = (
        _parse_relation_response(
            response
        )
    )

    valid_entity_ids = {
        entity.id
        for entity in document.entities
    }

    relations = []

    for item in raw_relations:

        if not isinstance(
            item,
            dict,
        ):

            continue

        try:

            head = int(
                item["head"]
            )

            tail = int(
                item["tail"]
            )

            relation = str(
                item["relation"]
            ).strip()

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

        if head not in valid_entity_ids:
            continue

        if tail not in valid_entity_ids:
            continue

        if not relation:
            continue

        relations.append(
            Relation(
                head=head,
                relation=relation,
                tail=tail,
            )
        )

    return relations


def extract_relations(
    document: Document,
    paragraphs: list[Paragraph],
    llm,
) -> list[Relation]:
    """
    Extract relations from ONE document.

    One document = one request inside a Batch.
    """

    if not paragraphs:

        return []

    request = (
        _prepare_relation_request(
            document=document,
            paragraphs=paragraphs,
            custom_id="document-0",
        )
    )

    responses = (
        llm.generate_batch(
            [request]
        )
    )

    response = responses.get(
        "document-0",
        "",
    )

    return _relations_from_response(
        document=document,
        response=response,
    )


def extract_relations_batch(
    documents: list[Document],
    paragraphs_list: list[list[Paragraph]],
    llm,
) -> list[list[Relation]]:
    """
    Extract relations from multiple documents
    using ONE ProxyAPI Batch API call.

    Each document becomes ONE independent request
    inside the same Batch.

    Example:

        Document 0 -> request document-0
        Document 1 -> request document-1
        Document 2 -> request document-2

    Returns
    -------
    list[list[Relation]]

        Relations are returned in the same order
        as input documents.
    """

    if not documents:

        return []

    requests = []

    document_map = {}

    for document_index, (
        document,
        paragraphs,
    ) in enumerate(
        zip(
            documents,
            paragraphs_list,
        )
    ):

        custom_id = (
            f"document-{document_index}"
        )

        request = (
            _prepare_relation_request(
                document=document,
                paragraphs=paragraphs,
                custom_id=custom_id,
            )
        )

        requests.append(
            request
        )

        document_map[
            custom_id
        ] = (
            document_index,
            document,
        )

    print(
        f"Prepared {len(requests)} "
        f"documents for one ProxyAPI Batch."
    )

    responses = (
        llm.generate_batch(
            requests
        )
    )

    all_relations = [
        []
        for _ in documents
    ]

    for custom_id, response in (
        responses.items()
    ):

        if custom_id not in document_map:

            print(
                f"Warning: unknown custom_id "
                f"{custom_id}"
            )

            continue

        document_index, document = (
            document_map[
                custom_id
            ]
        )

        relations = (
            _relations_from_response(
                document=document,
                response=response,
            )
        )

        all_relations[
            document_index
        ] = relations

    return all_relations


def build_graph(
    text: str,
    llm,
    coref_model,
    nlp,
) -> tuple[
    Document,
    list[Relation],
]:
    """
    Build a knowledge graph from ONE document.
    """

    document = extract_entities(
        text=text,
        model=coref_model,
        nlp=nlp,
    )

    paragraphs = (
        _split_into_paragraphs(
            document
        )
    )

    relations = extract_relations(
        document=document,
        paragraphs=paragraphs,
        llm=llm,
    )

    return (
        document,
        relations,
    )

def build_graph_batch(
    texts: list[str],
    llm,
    coref_model,
    nlp,
) -> list[
    tuple[
        Document,
        list[Relation],
    ]
]:
    """
    Build knowledge graphs for multiple documents.

    Entity extraction is performed locally for every document.

    Relation extraction is performed using ONE ProxyAPI
    Batch API call.

    Each document corresponds to one independent request
    inside the Batch.

    Returns
    -------
    list[tuple[Document, list[Relation]]]

        Results are returned in the same order
        as input texts.
    """

    if not texts:

        return []

    documents = []

    paragraphs_list = []

    for text in texts:

        document = extract_entities(
            text=text,
            model=coref_model,
            nlp=nlp,
        )

        paragraphs = (
            _split_into_paragraphs(
                document
            )
        )

        documents.append(
            document
        )

        paragraphs_list.append(
            paragraphs
        )

    all_relations = (
        extract_relations_batch(
            documents=documents,
            paragraphs_list=paragraphs_list,
            llm=llm,
        )
    )

    results = []

    for document, relations in zip(
        documents,
        all_relations,
    ):

        results.append(
            (
                document,
                relations,
            )
        )

    return results