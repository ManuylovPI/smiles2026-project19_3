from typing import List, Tuple, Optional
import pandas as pd

from knowledge_graph import build_graph, Document, Relation
from metrics import GraphMetricsCalculator

def relations_to_triplets(
    relations: List[Relation]
) -> List[Tuple[int, str, int]]:
    return [(r.head, r.relation, r.tail) for r in relations]

def compute_metrics_for_pair(
    context_text: str,
    answer_text: str,
    client,
    calculator: GraphMetricsCalculator,
    model: str = "llama3.1:8b",
) -> dict:
    ctx_doc, ctx_relations = build_graph(context_text, client=client, model=model)
    ans_doc, ans_relations = build_graph(answer_text, client=client, model=model)

    ctx_triplets = relations_to_triplets(ctx_relations)
    ans_triplets = relations_to_triplets(ans_relations)

    result = calculator.calculate_metrics(
        context_entities=ctx_doc.entities,
        context_triplets=ctx_triplets,
        answer_entities=ans_doc.entities,
        answer_triplets=ans_triplets,
    )
    return result

def build_feature_table(
    dataset: pd.DataFrame,
    client,
    model: str = "llama3.1:8b",
    context_col: str = "context",
    answer_col: str = "answer",
    label_col: str = "label",
    id_col: str = "example_id",
    doc_col: Optional[str] = "doc_id",
    output_csv: Optional[str] = "features.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    calculator = GraphMetricsCalculator()  

    rows = []
    total = len(dataset)

    for i, (_, row) in enumerate(dataset.iterrows()):
        example_id = row[id_col] if id_col in dataset.columns else i
        doc_id = row[doc_col] if (doc_col and doc_col in dataset.columns) else example_id
        label = row[label_col] if label_col in dataset.columns else None

        try:
            m = compute_metrics_for_pair(
                context_text=str(row[context_col]),
                answer_text=str(row[answer_col]),
                client=client,
                calculator=calculator,
                model=model,
            )
            rows.append({
                "example_id": example_id,
                "doc_id": doc_id,
                "EG": m["EG"],
                "RP": m["RP"],
                "SC": m["SC"],
                "label": label,
            })
        except Exception as e:
            if verbose:
                print(f"[{i+1}/{total}] пример {example_id}: ОШИБКА — {e}")
            continue

    features = pd.DataFrame(rows, columns=["example_id", "doc_id", "EG", "RP", "SC", "label"])

    if output_csv:
        features.to_csv(output_csv, index=False)

    return features


if __name__ == "__main__":
    demo_relations = [
        Relation(head=0, relation="founded", tail=1),
        Relation(head=1, relation="developed", tail=2),
    ]
    print(relations_to_triplets(demo_relations))
