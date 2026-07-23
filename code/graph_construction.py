import json
import os

import spacy

from datasets import load_dataset
from fastcoref import FCoref

from knowledge_graph import build_graph_batch
from llm import ProxyAPIBatchLLM


MODEL = "gpt-5.4-mini"

OUTPUT_DIR = "ragtruth_knowledge_graph_results_train"

api_key = os.environ.get(
    "PROXYAPI_KEY"
)

if not api_key:
    raise RuntimeError(
        "PROXYAPI_KEY environment variable "
        "is not set."
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


llm = ProxyAPIBatchLLM(
    api_key=api_key,
    model=MODEL,
    poll_interval=60,
    max_completion_tokens=1000,
)


print("Loading FastCoref model...")
coref_model = FCoref(device="cpu")

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("NLP models loaded.")

print("Loading RAGTruth dataset...")
dataset = load_dataset(
    "wandb/RAGTruth-processed",
    split="train",
)

dataset_size = len(dataset)

BATCH_SIZE = 10
START_INDEX = 0
END_INDEX = dataset_size


if START_INDEX < 0:
    raise ValueError(
        "START_INDEX must be >= 0."
    )


if START_INDEX >= dataset_size:
    raise ValueError(
        f"START_INDEX {START_INDEX} is outside "
        f"the dataset of size {dataset_size}."
    )


END_INDEX = min(
    END_INDEX,
    dataset_size,
)


if END_INDEX <= START_INDEX:
    raise ValueError(
        "END_INDEX must be greater than "
        "START_INDEX."
    )


print(
    f"Dataset size: "
    f"{dataset_size}"
)

print(
    f"Processing rows "
    f"{START_INDEX}–{END_INDEX - 1}"
)

print(
    f"Batch size: "
    f"{BATCH_SIZE} documents."
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)


for batch_start in range(
    START_INDEX,
    END_INDEX,
    BATCH_SIZE,
):

    batch_end = min(
        batch_start + BATCH_SIZE,
        END_INDEX,
    )


    batch_dataset = dataset.select(
        range(
            batch_start,
            batch_end,
        )
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        (
            "ragtruth_knowledge_graph_results_"
            f"{batch_start}_{batch_end}.json"
        ),
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Processing batch:"
    )

    print(
        f"Dataset rows "
        f"{batch_start}–{batch_end - 1}"
    )

    print(
        f"Number of documents: "
        f"{len(batch_dataset)}"
    )

    print(
        f"Output file: "
        f"{output_file}"
    )


    texts = [
        sample["output"]
        for sample in batch_dataset
    ]


    results = build_graph_batch(
        texts=texts,
        llm=llm,
        coref_model=coref_model,
        nlp=nlp,
    )


    batch_output = []


    for document_index, (
        document,
        relations,
    ) in enumerate(
        results
    ):

        document_id = (
            batch_start
            + document_index
        )


        sample = batch_dataset[
            document_index
        ]


        entities = [
            {
                "id": entity.id,
                "text": entity.text,
            }
            for entity in document.entities
        ]


        serialized_relations = [
            {
                "head": relation.head,
                "relation": relation.relation,
                "tail": relation.tail,
            }
            for relation in relations
        ]


        output_item = {
            "document_id": document_id,
            "ragtruth_id": sample["id"],
            "text": document.text,
            "entities": entities,
            "relations": serialized_relations,
        }


        batch_output.append(
            output_item
        )


    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            batch_output,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print(
        f"Batch completed."
    )

    print(
        f"Processed "
        f"{batch_start}–{batch_end - 1}"
    )

    print(
        f"Saved "
        f"{len(batch_output)} documents."
    )

    print(
        f"File: "
        f"{output_file}"
    )


    del results
    del batch_output
    del batch_dataset
    del texts


print(
    "\n"
    + "=" * 80
)

print(
    "Done."
)

print(
    f"Processed rows "
    f"{START_INDEX}–{END_INDEX - 1}"
)

print(
    f"Results saved to directory: "
    f"{OUTPUT_DIR}"
)