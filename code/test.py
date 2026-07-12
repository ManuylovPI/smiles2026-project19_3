from openai import OpenAI

from knowledge_graph import build_graph

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

text = """
OpenAI was founded in 2015 by Sam Altman and Elon Musk.

The company developed ChatGPT, which became popular worldwide.

Microsoft invested in OpenAI and provides cloud infrastructure.
"""

document, relations = build_graph(
    text=text,
    client=client,
    model="llama3.1:8b"
)

print("\nEntities")
print("-" * 40)

for entity in document.entities:
    print(f"{entity.id}: {entity.text}")

print("\nRelations")
print("-" * 40)

for relation in relations:
    head = document.entities[relation.head].text
    tail = document.entities[relation.tail].text

    print(
        f"{head} -- {relation.relation} --> {tail}"
    )