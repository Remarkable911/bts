#!/usr/bin/env python3
"""Export deterministic RULER requests for the vLLM serving benchmark.

The exporter uses the submission's bundled lm-evaluation-harness task builders
and the model tokenizer, but it never loads model weights.  Prompts are already
chat-templated in the resulting JSONL files; the serving benchmark must
therefore pass ``--skip-chat-template``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
SUBMISSION_ROOT = EVAL_ROOT.parent
BUNDLED_HARNESS = EVAL_ROOT / "common" / "lm_eval_harness"

sys.path.insert(0, str(SUBMISSION_ROOT))
sys.path.insert(1, str(BUNDLED_HARNESS))

DEFAULT_LENGTHS = (32768, 65536, 131072)
DEFAULT_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)
SCHEMA_VERSION = 1
PROTOCOL = "ruler_vllm_serving_workload_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_int_list(value: str, *, label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must contain integers") from error
    if not result or len(result) != len(set(result)) or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            f"{label} must contain unique positive comma-separated integers"
        )
    return result


def parse_names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("--tasks must contain unique names")
    unknown = sorted(set(result) - set(DEFAULT_TASKS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown RULER tasks: {', '.join(unknown)}")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    # RULER CWE keeps a module-local RNG rather than using the global random
    # state.  Reset it as well so constructing multiple lengths in one export
    # is equivalent to the suite's one-length-per-process workers.
    try:
        from lm_eval.tasks.ruler import cwe_utils

        cwe_utils.RNG.seed(42)
    except ImportError:
        pass


class TokenizerAdapter:
    """Small lm-eval-compatible facade without loading model weights."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.tokenizer_name = str(tokenizer.name_or_path).replace("/", "__")

    def tok_encode(self, text: str) -> list[int]:
        # Match vLLM CustomDataset, which calls tokenizer(prompt).input_ids.
        return list(self.tokenizer(text).input_ids)

    def tok_decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        try:
            return self.tokenizer.apply_chat_template(
                chat_history,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            # Non-Qwen tokenizers may not expose enable_thinking.  The formal
            # model contract below still restricts this protocol to Qwen3.
            return self.tokenizer.apply_chat_template(chat_history, **kwargs)


def left_truncate_like_hflm(
    adapter: TokenizerAdapter,
    prompt: str,
    *,
    max_prompt_tokens: int,
) -> tuple[str, list[int], int]:
    """Apply HFLM's generation-time token truncation to a string workload.

    lm-eval's causal HFLM keeps the final ``max_length - max_gen_toks``
    prompt tokens.  vLLM's CustomDataset accepts text rather than token IDs,
    so the retained suffix must also survive decode/re-encode exactly.  A
    byte-level tokenizer can rarely leave the cut at an invalid text boundary;
    in that case drop only the few additional leading tokens needed to find a
    stable suffix and record the exact count in the workload row.
    """

    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    original_ids = adapter.tok_encode(prompt)
    if len(original_ids) <= max_prompt_tokens:
        return prompt, original_ids, 0

    hflm_suffix = original_ids[-max_prompt_tokens:]
    # UTF-8/BPE boundaries normally stabilize immediately.  Keep the search
    # bounded so a tokenizer/configuration mismatch fails closed.
    for additional_drop in range(min(32, len(hflm_suffix))):
        retained_ids = hflm_suffix[additional_drop:]
        candidate = adapter.tok_decode(retained_ids)
        if adapter.tok_encode(candidate) == retained_ids:
            dropped = len(original_ids) - len(retained_ids)
            return candidate, retained_ids, dropped
    raise RuntimeError(
        "could not serialize the HFLM-truncated prompt as stable text within "
        "32 token-boundary attempts"
    )


def model_identity(model: str, trust_remote_code: bool) -> tuple[dict[str, Any], Any]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    fields = {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", -1)),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
        "head_dim": int(head_dim),
        "max_position_embeddings": int(
            getattr(config, "max_position_embeddings", -1)
        ),
        "rope_scaling": getattr(config, "rope_scaling", None),
    }
    expected = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    failed = [name for name, value in expected.items() if fields.get(name) != value]
    if failed:
        raise RuntimeError(
            "vLLM BTS serving workload requires Qwen3-8B; mismatched fields: "
            + ", ".join(failed)
        )
    identity = {
        "requested_model": model,
        **fields,
        "config_sha256": canonical_sha256(config.to_dict()),
        "huggingface_commit": getattr(config, "_commit_hash", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "chat_template_sha256": hashlib.sha256(
            str(getattr(tokenizer, "chat_template", "")).encode("utf-8")
        ).hexdigest(),
    }
    return identity, tokenizer


def build_task_rows(
    task_manager: Any,
    adapter: TokenizerAdapter,
    *,
    task_name: str,
    samples_per_task: int,
    context_length: int,
    seed: int,
    fixed_output_tokens: int | None,
) -> list[dict[str, Any]]:
    from lm_eval.tasks import get_task_dict

    seed_everything(seed)
    task_dict = get_task_dict([task_name], task_manager)
    seed_everything(seed)
    task = task_dict[task_name]
    task.set_config(key="num_fewshot", value=0)
    task.set_fewshot_seed(seed=seed)
    task.build_all_requests(
        samples=list(range(samples_per_task)),
        rank=0,
        world_size=1,
        cache_requests=False,
        rewrite_requests_cache=False,
        system_instruction=None,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        chat_template=adapter.apply_chat_template,
        tokenizer_name=adapter.tokenizer_name,
    )
    requests = list(task.instances)
    if len(requests) != samples_per_task:
        raise RuntimeError(
            f"{task_name} built {len(requests)} requests; expected {samples_per_task}"
        )

    rows = []
    for ordinal, request in enumerate(requests):
        if request.request_type != "generate_until" or int(request.repeats) != 1:
            raise RuntimeError(f"{task_name}[{ordinal}] is not one generate request")
        if len(request.args) != 2 or not isinstance(request.args[0], str):
            raise RuntimeError(f"{task_name}[{ordinal}] has malformed arguments")
        prompt, generation_kwargs = request.args
        if not isinstance(generation_kwargs, dict):
            raise RuntimeError(f"{task_name}[{ordinal}] has malformed generation args")
        output_tokens = (
            int(fixed_output_tokens)
            if fixed_output_tokens is not None
            else int(generation_kwargs.get("max_gen_toks", -1))
        )
        if output_tokens <= 0:
            raise RuntimeError(f"{task_name}[{ordinal}] has no positive output budget")
        original_prompt_tokens = len(adapter.tok_encode(prompt))
        prompt, prompt_token_ids, left_truncated_tokens = left_truncate_like_hflm(
            adapter,
            prompt,
            max_prompt_tokens=context_length - output_tokens,
        )
        prompt_tokens = len(prompt_token_ids)
        if prompt_tokens <= 0 or prompt_tokens + output_tokens > context_length:
            raise RuntimeError(
                f"{task_name}[{ordinal}] uses {prompt_tokens}+{output_tokens} tokens, "
                f"exceeding the {context_length} context budget after HFLM-style "
                "left truncation"
            )
        doc = dict(request.doc)
        sample_index = int(doc.get("index", ordinal))
        rows.append(
            {
                # The first two fields are the vLLM CustomDataset contract.
                "prompt": prompt,
                "output_tokens": output_tokens,
                # Remaining fields make the workload independently auditable.
                "task": task_name,
                "sample_index": sample_index,
                "context_length": context_length,
                "prompt_tokens": prompt_tokens,
                "original_prompt_tokens": original_prompt_tokens,
                "left_truncated_tokens": left_truncated_tokens,
                "hflm_prompt_token_limit": context_length - output_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "document_sha256": canonical_sha256(doc),
                "generation_kwargs_sha256": canonical_sha256(generation_kwargs),
            }
        )
    return rows


def source_manifest() -> dict[str, Any]:
    roots = (
        HERE,
        BUNDLED_HARNESS / "lm_eval" / "tasks" / "ruler",
    )
    files = {
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml"}
    }
    entries = {
        str(path.relative_to(SUBMISSION_ROOT)): file_sha256(path)
        for path in sorted(files)
    }
    return {"files": entries, "manifest_sha256": canonical_sha256(entries)}


def summarize_ints(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty workload")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "sum": sum(values),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", help="Local Qwen3-8B YaRN model/tokenizer path")
    result.add_argument("--output", type=Path)
    result.add_argument(
        "--lengths",
        type=lambda value: parse_int_list(value, label="--lengths"),
        default=DEFAULT_LENGTHS,
    )
    result.add_argument("--tasks", type=parse_names, default=DEFAULT_TASKS)
    result.add_argument("--samples-per-task", type=int, default=8)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--fixed-output-tokens",
        type=int,
        help=(
            "Override every RULER generation budget. By default each task's "
            "frozen max_gen_toks is retained."
        ),
    )
    result.add_argument("--trust-remote-code", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.samples_per_task <= 0:
        raise SystemExit("--samples-per-task must be positive")
    if args.seed < 0:
        raise SystemExit("--seed must be nonnegative")
    if args.fixed_output_tokens is not None and args.fixed_output_tokens <= 0:
        raise SystemExit("--fixed-output-tokens must be positive")
    plan = {
        "protocol": PROTOCOL,
        "model": args.model,
        "lengths": list(args.lengths),
        "tasks": list(args.tasks),
        "samples_per_task": args.samples_per_task,
        "requests_per_length": len(args.tasks) * args.samples_per_task,
        "seed": args.seed,
        "fixed_output_tokens": args.fixed_output_tokens,
        "prompts_are_chat_templated": True,
        "vllm_bench_requires_skip_chat_template": True,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not args.model:
        raise SystemExit("--model is required unless --dry-run is used")
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    identity, tokenizer = model_identity(args.model, args.trust_remote_code)
    if max(args.lengths) > int(identity["max_position_embeddings"]):
        raise RuntimeError(
            f"requested {max(args.lengths)} tokens but model supports only "
            f"{identity['max_position_embeddings']}"
        )

    # Reuse the already-audited public-data resolver from the Density suite.
    from eval.Density.run import prepare_qa_sources
    from lm_eval.tasks import TaskManager

    public_sources = prepare_qa_sources(output, args.tasks)
    adapter = TokenizerAdapter(tokenizer)
    lengths: dict[str, Any] = {}
    for context_length in args.lengths:
        task_manager = TaskManager(
            metadata={
                "max_seq_lengths": [context_length],
                "max_length": context_length,
                "tokenizer": args.model,
                "pretrained": args.model,
                "num_samples": args.samples_per_task,
            }
        )
        rows: list[dict[str, Any]] = []
        for task_name in args.tasks:
            rows.extend(
                build_task_rows(
                    task_manager,
                    adapter,
                    task_name=task_name,
                    samples_per_task=args.samples_per_task,
                    context_length=context_length,
                    seed=args.seed,
                    fixed_output_tokens=args.fixed_output_tokens,
                )
            )
        hashes = [str(row["prompt_sha256"]) for row in rows]
        if len(hashes) != len(set(hashes)):
            raise RuntimeError(f"{context_length} workload contains duplicate prompts")
        path = output / f"ruler-{context_length}.jsonl"
        atomic_write_jsonl(path, rows)
        lengths[str(context_length)] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "requests": len(rows),
            "prompt_tokens": summarize_ints(
                [int(row["prompt_tokens"]) for row in rows]
            ),
            "output_tokens": summarize_ints(
                [int(row["output_tokens"]) for row in rows]
            ),
            "total_tokens": summarize_ints(
                [
                    int(row["prompt_tokens"]) + int(row["output_tokens"])
                    for row in rows
                ]
            ),
            "left_truncation": {
                "requests_truncated": sum(
                    int(row["left_truncated_tokens"]) > 0 for row in rows
                ),
                "tokens_dropped": summarize_ints(
                    [int(row["left_truncated_tokens"]) for row in rows]
                ),
            },
            "tasks": {
                task: sum(row["task"] == task for row in rows) for task in args.tasks
            },
        }

    sources = source_manifest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "model": identity,
        "tasks": list(args.tasks),
        "samples_per_task": args.samples_per_task,
        "seed": args.seed,
        "fixed_output_tokens": args.fixed_output_tokens,
        "prompts_are_chat_templated": True,
        "enable_thinking": False,
        "vllm_custom_dataset_contract": {
            "skip_chat_template": True,
            "custom_output_len": -1,
            "ignore_eos": True,
            "disable_shuffle": True,
            "prefix_caching": False,
        },
        "public_sources": public_sources,
        "source_manifest": sources,
        "lengths": lengths,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
