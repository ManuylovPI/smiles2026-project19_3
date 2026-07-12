# smiles2026-project19_3
This repository is created for completing project 19_3 for SMILES 2026

Hallucination detection for RAG via knowledge-graph alignment: the retrieved context and the
model's answer are converted into graphs (entities + relations), and we measure how well the
answer graph is supported by the context graph.

---

## Idea

Three support metrics → a single uncertainty score:

| Metric | What it measures |
|---|---|
| **EG** — Entity Grounding | fraction of answer entities found in the context |
| **RP** — Relation Preservation | fraction of answer relations supported by the context |
| **SC** — Subgraph Connectivity | connectivity of the answer's supported subgraph |

`Uncertainty = f(EG, RP, SC)` — aggregated either by a hand-tuned weighted sum or a trained
classifier. A high uncertainty score signals a likely hallucination.

---

## Pipeline

```
context / answer
   │
   ▼  knowledge_graph.build_graph()      NER + coreference (fastcoref) → relations (LLM)
context & answer graphs
   │
   ▼  metrics.GraphMetricsCalculator     EG · RP · SC
   │
   ▼  pipeline.build_feature_table()     → features.csv
   │
   ▼  main.py                            aggregation classifier → Uncertainty
```

## Structure

```
code/
  knowledge_graph.py   entity & relation extraction → graph
  metrics.py           EG / RP / SC + aggregation
  pipeline.py          glue layer: text → metrics → features.csv
  main.py              classifier (LogReg / GradBoost / weighted-sum)
  test.py              minimal graph-pipeline run on a single text
requirements.txt
```

---

## Setup

> **Requires Python 3.11** (the graph stack won't build on 3.14).

```bash
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Relation extraction needs a local LLM via [Ollama](https://ollama.com):

```bash
ollama pull llama3.1:8b
```

**Compatible versions** (if `fastcoref` fails on `all_tied_weights_keys`):
`transformers==4.36.2`, `tokenizers==0.15.2`, `sentence-transformers==2.7.0`.

---

## Usage

```bash
cd code

python test.py        # check the graph pipeline on a single text (Ollama required)
python main.py         # classifier; falls back to synthetic data if features.csv is absent
```

`build_feature_table()` in `pipeline.py` runs a dataset (RAGTruth) and produces
`features.csv` with columns `example_id, doc_id, EG, RP, SC, label` — the input for `main.py`.

---

## Evaluation

`main.py` compares the weighted sum, Logistic Regression and Gradient Boosting on
**AUROC** and **AU-PRC** (plus an ablation over each metric alone), selects a threshold via
Youden's J, and saves the best model to `classifier.joblib`.

## Status

- ✅ Graph pipeline runs on real text (end-to-end via Ollama)
- ✅ EG / RP / SC metrics implemented
- ✅ Aggregation classifier assembled
- ⬜ Run on RAGTruth and train on real data
- ⬜ Plug the trained classifier back into `metrics.py`

