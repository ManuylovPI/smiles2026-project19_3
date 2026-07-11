from openai import OpenAI
from knowledge_graph import Entity, Paragraph, EntityMention, _mark_entities, extract_relations
# Document entities

google = Entity(
    id=0,
    text="Google"
)

gemini = Entity(
    id=1,
    text="Gemini"
)

deepmind = Entity(
    id=2,
    text="DeepMind"
)

paragraph_text = (
    "Google developed Gemini. "
    "Google acquired DeepMind."
)

paragraph = Paragraph(
    text=paragraph_text,
    entity_mentions=[
        EntityMention(
            entity=google,
            word_span=(0, 0),
            char_span=(0, 6)
        ),
        EntityMention(
            entity=gemini,
            word_span=(2, 2),
            char_span=(17, 23)
        ),
        EntityMention(
            entity=google,
            word_span=(3, 3),
            char_span=(25, 31)
        ),
        EntityMention(
            entity=deepmind,
            word_span=(5, 5),
            char_span=(41, 49)
        )
    ]
)

print(_mark_entities(paragraph))

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",  # любое непустое значение
)

relations = extract_relations(
    paragraph=paragraph,
    client=client,
    model="llama3.1:8b"
)

print(relations)