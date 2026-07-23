import json
import os
import time

from openai import OpenAI


class ProxyAPIBatchLLM:
    """
    ProxyAPI Batch API client.

    Sends many independent OpenAI-compatible chat completion
    requests as a single Batch and returns responses by custom_id.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4-mini",
        base_url: str = "https://api.proxyapi.ru/openai/v1",
        poll_interval: int = 60,
        max_completion_tokens: int = 1000,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model
        self.poll_interval = poll_interval
        self.max_completion_tokens = (
            max_completion_tokens
        )

    def generate_batch(
        self,
        requests: list[dict],
    ) -> dict[str, str]:
        """
        Send multiple requests as one ProxyAPI Batch.

        Each request must contain:

        {
            "custom_id": "...",
            "system_prompt": "...",
            "user_prompt": "...",
            "temperature": 0.0,
        }

        Returns
        -------
        dict[str, str]
            Mapping:
                custom_id -> model response
        """

        if not requests:
            return {}

        os.makedirs(
            "batch_temp",
            exist_ok=True,
        )

        jsonl_path = (
            "batch_temp/"
            f"requests_{int(time.time())}.jsonl"
        )

        print(
            f"Preparing {len(requests)} requests "
            f"for ProxyAPI Batch..."
        )

        with open(
            jsonl_path,
            "w",
            encoding="utf-8",
        ) as f:

            for request in requests:

                batch_request = {
                    "custom_id": request[
                        "custom_id"
                    ],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": request[
                                    "system_prompt"
                                ],
                            },
                            {
                                "role": "user",
                                "content": request[
                                    "user_prompt"
                                ],
                            },
                        ],
                        "temperature": request.get(
                            "temperature",
                            0.0,
                        ),
                        "max_completion_tokens": (
                            self.max_completion_tokens
                        ),
                    },
                }

                f.write(
                    json.dumps(
                        batch_request,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


        print(
            f"Uploading batch input file: "
            f"{jsonl_path}"
        )

        with open(
            jsonl_path,
            "rb",
        ) as f:

            batch_input_file = (
                self.client.files.create(
                    file=f,
                    purpose="batch",
                )
            )

        print(
            f"Input file ID: "
            f"{batch_input_file.id}"
        )

        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        print(
            f"Batch created: "
            f"{batch.id}"
        )

        while True:

            batch_status = (
                self.client.batches.retrieve(
                    batch.id
                )
            )

            print(
                f"Batch status: "
                f"{batch_status.status}"
            )

            if batch_status.status in {
                "completed",
                "failed",
                "expired",
                "cancelled",
            }:
                break

            time.sleep(
                self.poll_interval
            )

        if batch_status.status != "completed":

            error_file_id = getattr(
                batch_status,
                "error_file_id",
                None,
            )

            if error_file_id:

                error_content = (
                    self.client.files.content(
                        error_file_id
                    )
                    .content
                    .decode("utf-8")
                )

                print(
                    "Batch error file:"
                )

                print(
                    error_content
                )

            raise RuntimeError(
                "Batch failed with status: "
                f"{batch_status.status}"
            )


        output_file_id = getattr(
            batch_status,
            "output_file_id",
            None,
        )

        if output_file_id is None:

            raise RuntimeError(
                "Batch completed but "
                "output_file_id is missing."
            )

        print(
            f"Downloading results: "
            f"{output_file_id}"
        )

        result_content = (
            self.client.files.content(
                output_file_id
            )
            .content
            .decode("utf-8")
        )

        results = {}

        for line in (
            result_content.splitlines()
        ):

            if not line.strip():
                continue

            result = json.loads(
                line
            )

            custom_id = result.get(
                "custom_id"
            )

            response = result.get(
                "response",
                {},
            )

            body = response.get(
                "body",
                {},
            )

            choices = body.get(
                "choices",
                [],
            )

            if not choices:

                results[
                    custom_id
                ] = ""

                continue

            content = (
                choices[0]
                .get(
                    "message",
                    {},
                )
                .get(
                    "content",
                    "",
                )
            )

            results[
                custom_id
            ] = content

        print(
            f"Received {len(results)} "
            f"batch results."
        )

        return results