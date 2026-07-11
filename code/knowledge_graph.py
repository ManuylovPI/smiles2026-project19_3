from dataclasses import dataclass
from openai import OpenAI
import json


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
class Paragraph:
    text: str
    entity_mentions: list[EntityMention]


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


def extract_relations(
    paragraph: Paragraph,
    client: OpenAI,
    model: str = "meta-llama/llama-3.3-70b-instruct"
) -> list[dict]:
    """
    Extract directed semantic relations between entities
    occurring in a paragraph.

    Parameters
    ----------
    paragraph
        Paragraph with entity mentions.

    Returns
    -------
    list[dict]

    Example
    -------
    [
        {
            "head": 0,
            "relation": "developed",
            "tail": 1
        }
    ]
    """

    marked_paragraph = _mark_entities(paragraph)

    # Collect unique entities appearing in the paragraph
    unique_entities = {}

    for mention in paragraph.entity_mentions:
        unique_entities[mention.entity.id] = mention.entity.text

    entity_description = "\n".join(
        f"Entity ID {entity_id}: {text}"
        for entity_id, text in sorted(unique_entities.items())
    )

    prompt = f"""
You are an expert in Relation Extraction.

Your task is to extract explicit directed semantic relations between the marked entities.

Rules:

- Use ONLY entities marked with <entity id="..."> tags.
- Return ONLY relations explicitly stated in the paragraph.
- Do not infer missing information.
- Ignore unrelated entities.
- Relations are directed.
- Use entity IDs in the output.
- If no relations exist, return [].
- Return ONLY valid JSON.

Paragraph:

{marked_paragraph}

Entities:

{entity_description}

Output format:

[
    {{
        "head": 0,
        "relation": "developed",
        "tail": 1
    }}
]
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You extract directed semantic relations between marked entities."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)