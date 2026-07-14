import json
import pandas as pd

from openai import OpenAI
from datasets import Dataset
from knowledge_graph import build_graph


def llm_match(
    predicted: tuple[str, str, str],
    gold: tuple[str, str, str],
    client: OpenAI,
    model: str = "llama3.1:8b",
) -> bool:
    """
    Determine whether two triples are semantically equivalent.

    Parameters
    ----------
    predicted
        (head, relation, tail) predicted by the system.

    gold
        Gold-standard triple.

    Returns
    -------
    bool
        True if the triples express the same fact.
    """

    prompt = f"""
You are evaluating a knowledge graph extraction system.

Determine whether the following two triples express the SAME factual relation.

Consider:

- synonymous entity names
- synonymous relations
- passive vs active voice
- different wording

Return ONLY JSON.

Predicted:

{{
    "head": "{predicted[0]}",
    "relation": "{predicted[1]}",
    "tail": "{predicted[2]}"
}}

Gold:

{{
    "head": "{gold[0]}",
    "relation": "{gold[1]}",
    "tail": "{gold[2]}"
}}

Output:

{{"match": true}}

or

{{"match": false}}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You compare knowledge graph triples."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)
    return result["match"]



def semantic_f1(predicted, gold, client, model="llama3.1:8b"):
    matched_gold = set()
    tp = 0
    for pred in predicted:
        for i, gold_triple in enumerate(gold):
            if i in matched_gold:
                continue

            if llm_match(
                pred,
                gold_triple,
                client,
                model,
            ):
                tp += 1
                matched_gold.add(i)
                break

    fp = len(predicted) - tp
    fn = len(gold) - tp

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def normalize(text: str) -> str:
    return text.strip().lower()



df = pd.read_parquet(
    "hf://datasets/GEM/web_nlg@refs/convert/parquet/en/test/0000.parquet"
)

dataset = Dataset.from_pandas(df)
MODEL = "llama3.1:8b"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

metrics = []

N = 50
dataset_subset = dataset.shuffle(seed=42).select(range(N))

for sample in dataset_subset:

    text = sample["target"]

    document, relations = build_graph(
        text=text,
        client=client,
        model=MODEL
    )

    entity_map = {
        entity.id: normalize(entity.text)
        for entity in document.entities
    }

    predicted = []

    for relation in relations:

        head = entity_map.get(relation.head)
        tail = entity_map.get(relation.tail)

        if head is None or tail is None:
            continue

        predicted.append(
            (
                head,
                normalize(relation.relation),
                tail
            )
        )

    gold = []

    for triple in sample["input"]:
        head, relation, tail = [
            part.strip().strip('"').replace("_", " ")
            for part in triple.split("|")
        ]

        gold.append(
            (
                normalize(head),
                normalize(relation),
                normalize(tail)
            )
        )

    metrics.append(
        semantic_f1(
            predicted=predicted,
            gold=gold,
            client=client,
            model=MODEL
        )
    )

precision = sum(m["precision"] for m in metrics) / len(metrics)
recall = sum(m["recall"] for m in metrics) / len(metrics)
f1 = sum(m["f1"] for m in metrics) / len(metrics)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")