# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import itertools  # noqa: I001
import json
import logging
import os
import random
import tempfile
from functools import cache
from hashlib import sha256
from pathlib import Path

import datasets
import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer

CONFIG = {
    "tokens_to_generate": 32,
    "template": """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
    "answer_prefix": """Answer:""",
}
SEED = 42
TEMPLATE = CONFIG["template"]
DOCUMENT_PROMPT = "Document {i}:\n{document}"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_MIRROR_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json"
SQUAD_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
HOTPOTQA_SHA256 = "7f0cd50fe0b6abc95596a8e0b9c4de931a2f6d61a3e75eec18632bfa7a9b1b76"


def _json_cache_path(url):
    root = Path(os.environ.get("RULER_DATA_CACHE", Path(datasets.config.HF_DATASETS_CACHE) / "ruler" / "raw"))
    return root / f"{sha256(url.encode()).hexdigest()}.json"


def _cache_json(url, data):
    path = _json_cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete JSON files, including when several ranks prepare data.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@cache
def download_json(url, fallback_urls=(), expected_sha256=None) -> dict | list:
    path = _json_cache_path(url)
    if path.is_file():
        try:
            payload = path.read_bytes()
            if expected_sha256 and sha256(payload).hexdigest() != expected_sha256:
                raise ValueError("cached source SHA-256 does not match frozen protocol")
            return json.loads(payload)
        except ValueError:
            logging.getLogger(__name__).warning("Ignoring invalid JSON cache: %s", path)

    last_error = None
    for source in (url, *fallback_urls):
        for attempt in range(2):
            try:
                with requests.get(source, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    payload = response.content
                    if (
                        expected_sha256
                        and sha256(payload).hexdigest() != expected_sha256
                    ):
                        raise ValueError(
                            f"source SHA-256 mismatch for {source}: "
                            f"expected {expected_sha256}, got {sha256(payload).hexdigest()}"
                        )
                    data = json.loads(payload)
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                logging.getLogger(__name__).warning(
                    "JSON download failed (%s, attempt %s/2): %s", source, attempt + 1, exc
                )
                continue
            if expected_sha256:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb", dir=path.parent, delete=False
                    ) as handle:
                        temporary = Path(handle.name)
                        handle.write(payload)
                    os.replace(temporary, path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            else:
                _cache_json(url, data)
            return data
    raise last_error


@cache
def read_squad(
    url=SQUAD_URL,
) -> tuple[list[dict], list[str]]:
    local_path = os.environ.get("RULER_SQUAD_PATH")
    if local_path:
        payload = Path(local_path).read_bytes()
        observed = sha256(payload).hexdigest()
        if observed != SQUAD_SHA256:
            raise RuntimeError(
                f"RULER_SQUAD_PATH SHA-256 mismatch: expected {SQUAD_SHA256}, "
                f"got {observed}"
            )
        data = json.loads(payload)
    else:
        try:
            data = download_json(
                url,
                (SQUAD_MIRROR_URL,) if url == SQUAD_URL else (),
                SQUAD_SHA256 if url == SQUAD_URL else None,
            )
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                "Could not download SQuAD dev-v2.0.json. Set RULER_SQUAD_PATH to a local "
                "original dev-v2.0.json, or restore network access. Successful downloads "
                "are saved in RULER_DATA_CACHE (default: HF datasets cache/ruler/raw)."
            ) from exc
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )

    return total_qas, total_docs


@cache
def read_hotpotqa(
    url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
) -> tuple[list[dict], list[str]]:
    local_path = os.environ.get("RULER_HOTPOTQA_PATH")
    if local_path:
        payload = Path(local_path).read_bytes()
        observed = sha256(payload).hexdigest()
        if observed != HOTPOTQA_SHA256:
            raise RuntimeError(
                f"RULER_HOTPOTQA_PATH SHA-256 mismatch: expected {HOTPOTQA_SHA256}, "
                f"got {observed}"
            )
        data = json.loads(payload)
    else:
        try:
            data = download_json(
                url,
                expected_sha256=(
                    HOTPOTQA_SHA256
                    if url
                    == "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
                    else None
                ),
            )
        except (requests.RequestException, ValueError):
            if url != "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json":
                raise
            logging.getLogger(__name__).warning(
                "HotpotQA JSON download failed; using hotpotqa/hotpot_qa "
                "distractor validation from Hugging Face."
            )
            # Fetch only the dev split, and reuse the Hub cache across methods.
            try:
                parquet_path = hf_hub_download(
                    repo_id="hotpotqa/hotpot_qa",
                    repo_type="dataset",
                    filename="distractor/validation-00000-of-00001.parquet",
                    revision="1908d6afbbead072334abe2965f91bd2709910ab",
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not download HotpotQA from the original site or Hugging Face. "
                    "Set RULER_HOTPOTQA_PATH to a local hotpot_dev_distractor_v1.json "
                    "or check your HF_ENDPOINT/network configuration."
                ) from exc
            validation = datasets.load_dataset(
                "parquet", data_files={"validation": parquet_path}, split="validation"
            )
            data = [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "context": list(
                        zip(row["context"]["title"], row["context"]["sentences"])
                    ),
                }
                for row in validation
            ]
            _cache_json(url, data)
            cached = _json_cache_path(url)
            observed = sha256(cached.read_bytes()).hexdigest()
            if observed != HOTPOTQA_SHA256:
                raise RuntimeError(
                    "HotpotQA fallback conversion does not match the frozen source: "
                    f"expected {HOTPOTQA_SHA256}, got {observed}"
                )
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


def generate_input_output(
    index: int, num_docs: int, qas: list[dict], docs: list[str]
) -> tuple[str, list[str]]:
    curr_q: str = qas[index]["query"]
    curr_a: list[str] = qas[index]["outputs"]
    curr_docs: list[int] = qas[index]["context"]
    curr_more: list[int] = qas[index].get("more_context", [])
    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i, d in enumerate(docs) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))

        all_docs = [docs[idx] for idx in all_docs]
    else:
        all_docs = docs

    random.Random(SEED).shuffle(all_docs)

    context = "\n\n".join(
        [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    input_text = TEMPLATE.format(context=context, query=curr_q)
    return input_text, curr_a


def generate_samples(
    tokenizer,
    docs: list[str],
    qas: list[dict],
    max_seq_length: int,
    num_samples: int = 500,
    tokens_to_generate: int = 32,
    pre_samples: int = 0,
    incremental: int = 10,
    remove_newline_tab=False,
) -> list[dict]:
    write_jsons = []
    tokens_to_generate = tokens_to_generate

    # Find the perfect num_docs
    num_docs = incremental

    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer = generate_input_output(0, num_docs, qas=qas, docs=docs)
        # Calculate the number of tokens in the example
        total_tokens = len(tokenizer(input_text + f" {answer}").input_ids)
        # print(
        #     f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}"
        # )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break

        num_docs += incremental
        if num_docs > len(docs):
            num_docs = len(docs)
            break
    # print("Number of documents:", num_docs)

    # Generate samples
    for index in tqdm(
        range(num_samples), desc=f"Generating QA Samples | {max_seq_length}"
    ):
        used_docs = num_docs
        while True:
            try:
                input_text, answer = generate_input_output(
                    index + pre_samples, used_docs, qas=qas, docs=docs
                )
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:  # noqa: E722
                if used_docs > incremental:
                    used_docs -= incremental

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": "Answer:",
        }
        write_jsons.append(formatted_output)

    return write_jsons


def get_dataset(pretrained, docs, qas, max_seq_length=None, num_samples=500, **kwargs) -> list[dict]:
    tokenizer = get_tokenizer(pretrained)
    write_jsons = generate_samples(
        tokenizer=tokenizer,
        docs=docs,
        qas=qas,
        num_samples=num_samples,
        tokens_to_generate=32,
        max_seq_length=max_seq_length,
    )
    return write_jsons


def get_qa_dataset(ds, **kwargs) -> dict[str, datasets.Dataset]:
    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
    num_samples = kwargs.pop("num_samples", 500)
    if ds == "squad":
        qas, docs = read_squad()
    else:
        qas, docs = read_hotpotqa()
    df = (
        get_dataset(pretrained=pretrained, docs=docs, qas=qas, max_seq_length=seq, num_samples=num_samples)
        for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    )

    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }


def get_squad(**kwargs):
    return get_qa_dataset("squad", **kwargs)


def get_hotpotqa(**kwargs):
    return get_qa_dataset("hotpotqa", **kwargs)
