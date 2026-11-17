#!/usr/bin/env python3
"""Shared lm-eval runtime used by the LongBench and RULER entry points."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
SUBMISSION_ROOT = EVAL_ROOT.parent
BUNDLED_HARNESS = HERE / "lm_eval_harness"
METHOD_DISPLAY_NAME = "BTS"
DEFAULT_RULER_SUITE = (
    SUBMISSION_ROOT.parent / "results/ruler/topk040_tail256_official_protocol_20261105"
)
OFFICIAL_RULER_GROUP = "official_fixed_ruler"
DEFAULT_RULER_TASKS = (
    "niah_single_1",
    "niah_multikey_1",
    "ruler_fwe",
    "ruler_qa_hotpot",
    "niah_multiquery",
)
OFFICIAL_RULER_LENGTHS = (32768, 65536, 131072)
OFFICIAL_RULER_SAMPLES_PER_TASK_LENGTH = 100
OFFICIAL_RULER_MANIFEST_SHA256 = (
    "7c594b7eb6f03f8bfd5cd514a24fb9eea046368640cc5c6aac406fdc569818aa"
)
OFFICIAL_RULER_FILES_MANIFEST_SHA256 = (
    "cdfa2509a3338dd01adedc2fb49fa41a041a1f25902aee4c168ef35b667df12b"
)
OFFICIAL_RULER_DATA_INVENTORY_SHA256 = (
    "03a840e4f38df058f6dd75bbd2210696138c655085af0ee9039f9c48701b415f"
)

if not (BUNDLED_HARNESS / "lm_eval" / "__init__.py").is_file():
    raise RuntimeError(f"bundled lm-eval harness is missing: {BUNDLED_HARNESS}")

DATA_CONTRACT = {
    "longbench_revision": "944462f8d7475c8c153be22f890a299fe30254ee",
    "ruler_paul_graham_revision": "792d672b77d1a67720f7e9820d646358fc18792e",
    "ruler_hotpot_hub_fallback_revision": "1908d6afbbead072334abe2965f91bd2709910ab",
    "ruler_squad_raw_sha256": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    "ruler_hotpot_raw_sha256": "7f0cd50fe0b6abc95596a8e0b9c4de931a2f6d61a3e75eec18632bfa7a9b1b76",
}

# Prefer this submission's frozen ``bts`` package and the bundled harness.
# No development-repository path is consulted.
sys.path.insert(0, str(SUBMISSION_ROOT))
sys.path.insert(1, str(BUNDLED_HARNESS))

from baselines.registry import (
    ATTENTION_NAMES,
    configure_environment,
    method_contract,
    normalize_name,
)


SCENARIOS: dict[str, tuple[str, ...]] = {
    "math": ("gsm8k", "minerva_math500"),
    "code": ("longbench_code",),
    "long-text": (
        "longbench_single",
        "longbench_multi",
        "longbench_summarization",
        "ruler_qa_squad",
    ),
    "ruler": ("ruler",),
}
SCENARIOS["all"] = tuple(
    dict.fromkeys(
        task for name in ("math", "code", "long-text") for task in SCENARIOS[name]
    )
)

SUITE_TASKS = {
    "longbench": "longbench",
    "ruler": ",".join(DEFAULT_RULER_TASKS),
}

LONG_BENCH_LEAVES = {
    "longbench_lcc",
    "longbench_repobench-p",
    "longbench_trec",
    "longbench_triviaqa",
    "longbench_samsum",
    "longbench_lsht",
    "longbench_hotpotqa",
    "longbench_2wikimqa",
    "longbench_musique",
    "longbench_dureader",
    "longbench_narrativeqa",
    "longbench_qasper",
    "longbench_multifieldqa_en",
    "longbench_multifieldqa_zh",
    "longbench_gov_report",
    "longbench_qmsum",
    "longbench_multi_news",
    "longbench_vcsum",
    "longbench_passage_count",
    "longbench_passage_retrieval_en",
    "longbench_passage_retrieval_zh",
}

RULER_LEAVES = {
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
}

OFFICIAL_RULER_LEAVES = {f"official_fixed_{task}" for task in RULER_LEAVES}

VERIFIED_REFERENCES = {
    "longbench": {
        "method": "Top-K0.4 + Tail256",
        "score": 40.8784696310,
        "score_text": "40.8784696310",
        "raw_result_path": (
            "../results/longbench/topk040_tail256_20261031/"
            "results_2026-11-01T03-39-55.872698.json"
        ),
        "raw_result_sha256": (
            "9826abeecb30c7c6c1662361cf3bc0b0ec914f1dfa89bd0b2ed8042bc9522b81"
        ),
        "historical_reference": True,
        "reproduced_by_current_run": False,
        "provenance": "verified_historical_reference_not_reproduced_by_this_run",
    },
    "ruler": {
        "method": "Top-K0.4 + Tail256; block thresholds 0.95/0.95",
        "scores": {
            "32768": 82.2628205,
            "65536": 75.1666667,
            "131072": 69.6769231,
        },
        "score_text": {
            "32768": "82.2628205",
            "65536": "75.1666667",
            "131072": "69.6769231",
        },
        "raw_result_path": (
            "/data3/lsf_data/prism_data/results/ruler/"
            "topk040_tail256_official_protocol_threshold095_20261105/outputs/"
            "results_2026-11-06T22-31-42.346281.json"
        ),
        "raw_result_sha256": (
            "cc333bc31b199b2ccc1f01a6e586d15d649a71d906020f146107f432960bd406"
        ),
        "historical_reference": True,
        "reproduced_by_current_run": False,
        "provenance": "verified_historical_reference_not_reproduced_by_this_run",
    },
}


def _verified_reference(name: str) -> dict:
    """Resolve and hash-check a historical result without claiming reproduction."""
    reference = json.loads(json.dumps(VERIFIED_REFERENCES[name]))
    path = Path(reference["raw_result_path"]).expanduser()
    if not path.is_absolute():
        path = SUBMISSION_ROOT / path
    path = path.resolve(strict=False)
    observed = _sha256(path) if path.is_file() else None
    reference.update(
        {
            "resolved_raw_result_path": str(path),
            "source_present": path.is_file(),
            "observed_raw_result_sha256": observed,
            "source_hash_verified": observed == reference["raw_result_sha256"],
        }
    )
    return reference


def _limit(value: str) -> int | float:
    """Accept either an example count or lm-eval's fractional limit."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    if parsed >= 1 and parsed.is_integer():
        return int(parsed)
    if parsed >= 1:
        raise argparse.ArgumentTypeError("non-integer limits >= 1 are ambiguous")
    return parsed


def _parser(suite: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    if suite is None:
        choice = parser.add_mutually_exclusive_group()
        choice.add_argument("--scenario", choices=tuple(SCENARIOS), default="all")
        choice.add_argument(
            "--tasks",
            help="Comma-separated bundled lm-eval task or group names",
        )
    else:
        if suite not in SUITE_TASKS:
            raise ValueError(f"unknown evaluation suite: {suite}")
        parser.set_defaults(scenario=suite)
        if suite == "ruler":
            parser.add_argument(
                "--tasks",
                default=SUITE_TASKS[suite],
                help=(
                    "Comma-separated frozen RULER task IDs. Base IDs such as "
                    "niah_single_1 are mapped to official_fixed_* tasks; use "
                    "'ruler' for the complete 13-task group."
                ),
            )
        else:
            parser.set_defaults(tasks=SUITE_TASKS[suite])
    parser.add_argument(
        "--model",
        help="Local path or Hugging Face supported text-model ID",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Pinned Hugging Face model revision (ignored by local model directories)",
    )
    parser.add_argument(
        "--attention",
        choices=ATTENTION_NAMES,
        default="bts",
        help=(
            "Attention method: BTS, dense SDPA, FlashAttention-2, "
            "FlexPrefill, MInference, or XAttention"
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=131072 if suite == "ruler" else 32768,
    )
    parser.add_argument(
        "--ruler-suite",
        type=Path,
        default=DEFAULT_RULER_SUITE,
        help=(
            "Frozen official RULER suite root containing manifest.json and "
            "tasks/official_fixed_ruler.yaml"
        ),
    )
    parser.add_argument(
        "--ruler-samples-per-task",
        type=int,
        default=100,
        help="Samples built for each RULER task at the selected length (formal: 100)",
    )
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit", type=_limit)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parallelize", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking in the chat template (disabled by default)",
    )
    parser.add_argument(
        "--confirm-run-unsafe-code",
        action="store_true",
        help="Required by lm-eval for local Python-backed tasks such as RULER",
    )
    parser.add_argument("--no-log-samples", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--numpy-random-seed", type=int, default=1234)
    parser.add_argument("--torch-random-seed", type=int, default=1234)
    parser.add_argument("--fewshot-random-seed", type=int, default=1234)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Enforce the full-sample, batch-1, auditable evaluation contract",
    )
    parser.add_argument("--list-tasks", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _official_ruler_suite(path: Path) -> dict:
    """Validate and describe the immutable external 3,900-sample suite."""
    root = path.expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.absolute()
    manifest_path = root / "manifest.json"
    frozen_files_path = root / "frozen_files_sha256.json"
    data_inventory_path = root / "data_inventory.json"
    tasks_path = root / "tasks"
    group_path = tasks_path / f"{OFFICIAL_RULER_GROUP}.yaml"
    if not manifest_path.is_file():
        raise RuntimeError(f"RULER suite manifest is missing: {manifest_path}")
    if not group_path.is_file():
        raise RuntimeError(f"RULER external group is missing: {group_path}")
    if not frozen_files_path.is_file():
        raise RuntimeError(
            f"RULER frozen-files manifest is missing: {frozen_files_path}"
        )
    if not data_inventory_path.is_file():
        raise RuntimeError(f"RULER data inventory is missing: {data_inventory_path}")
    missing_task_files = sorted(
        task
        for task in OFFICIAL_RULER_LEAVES
        if not (tasks_path / f"{task}.yaml").is_file()
    )
    if missing_task_files:
        raise RuntimeError(
            "RULER suite is missing task definitions: " + ", ".join(missing_task_files)
        )

    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != OFFICIAL_RULER_MANIFEST_SHA256:
        raise RuntimeError(
            "RULER suite manifest hash mismatch: "
            f"expected {OFFICIAL_RULER_MANIFEST_SHA256}, got {manifest_sha256}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen_files = json.loads(frozen_files_path.read_text(encoding="utf-8"))
        data_inventory = json.loads(data_inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read RULER suite manifest: {error}") from error

    frozen_files_sha256 = _sha256(frozen_files_path)
    if frozen_files_sha256 != OFFICIAL_RULER_FILES_MANIFEST_SHA256:
        raise RuntimeError(
            "RULER frozen-files manifest hash mismatch: "
            f"expected {OFFICIAL_RULER_FILES_MANIFEST_SHA256}, "
            f"got {frozen_files_sha256}"
        )
    data_inventory_sha256 = _sha256(data_inventory_path)
    if data_inventory_sha256 != OFFICIAL_RULER_DATA_INVENTORY_SHA256:
        raise RuntimeError(
            "RULER data inventory hash mismatch: "
            f"expected {OFFICIAL_RULER_DATA_INVENTORY_SHA256}, "
            f"got {data_inventory_sha256}"
        )

    expected = {
        "ready": True,
        "max_length": 131072,
        "inference_lengths": list(OFFICIAL_RULER_LENGTHS),
        "task_names": [
            "official_fixed_niah_single_1",
            "official_fixed_niah_single_2",
            "official_fixed_niah_single_3",
            "official_fixed_niah_multikey_1",
            "official_fixed_niah_multikey_2",
            "official_fixed_niah_multikey_3",
            "official_fixed_niah_multiquery",
            "official_fixed_niah_multivalue",
            "official_fixed_ruler_vt",
            "official_fixed_ruler_cwe",
            "official_fixed_ruler_fwe",
            "official_fixed_ruler_qa_squad",
            "official_fixed_ruler_qa_hotpot",
        ],
        "samples_per_task_length": OFFICIAL_RULER_SAMPLES_PER_TASK_LENGTH,
        "inference_samples": 3900,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "RULER suite manifest does not match the frozen protocol: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    length_directories = {32768: "32k", 65536: "64k", 131072: "128k"}
    selected_relative_paths = [
        f"tasks/{OFFICIAL_RULER_GROUP}.yaml",
        "tasks/scoring.py",
        *(f"tasks/{task}.yaml" for task in sorted(OFFICIAL_RULER_LEAVES)),
        *(
            f"data/{directory}/{task}.jsonl"
            for task in sorted(RULER_LEAVES)
            for _length, directory in length_directories.items()
        ),
    ]
    selected_files: dict[str, str] = {}
    validation_errors: list[str] = []
    inventory_tasks = data_inventory.get("tasks", {})
    for relative in selected_relative_paths:
        candidate = root / relative
        if not candidate.is_file():
            validation_errors.append(f"missing frozen file: {relative}")
            continue
        observed = _sha256(candidate)
        selected_files[relative] = observed
        expected_hash = frozen_files.get(relative)
        if expected_hash != observed:
            validation_errors.append(f"frozen file hash mismatch: {relative}")
        if relative.startswith("data/"):
            base_task = candidate.stem
            directory = Path(relative).parts[1]
            length = next(
                length
                for length, name in length_directories.items()
                if name == directory
            )
            row = inventory_tasks.get(base_task, {}).get(str(length), {})
            if row.get("path") != relative:
                validation_errors.append(
                    f"data inventory path mismatch: {base_task}/{length}"
                )
            if row.get("samples") != OFFICIAL_RULER_SAMPLES_PER_TASK_LENGTH:
                validation_errors.append(
                    f"data inventory sample count mismatch: {base_task}/{length}"
                )
            if row.get("sha256") != observed:
                validation_errors.append(
                    f"data inventory hash mismatch: {base_task}/{length}"
                )
    if validation_errors:
        raise RuntimeError(
            "RULER frozen suite file verification failed: "
            + "; ".join(validation_errors)
        )
    selected_files_manifest_sha256 = hashlib.sha256(
        "\n".join(
            f"{relative}  {selected_files[relative]}"
            for relative in sorted(selected_files)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "root": str(root),
        "tasks_path": str(tasks_path),
        "group": OFFICIAL_RULER_GROUP,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "frozen_files_manifest_path": str(frozen_files_path),
        "frozen_files_manifest_sha256": frozen_files_sha256,
        "data_inventory_path": str(data_inventory_path),
        "data_inventory_sha256": data_inventory_sha256,
        "verified_file_count": len(selected_files),
        "verified_files_manifest_sha256": selected_files_manifest_sha256,
        "max_length": 131072,
        "inference_lengths": list(OFFICIAL_RULER_LENGTHS),
        "task_names": expected["task_names"],
        "samples_per_task_length": OFFICIAL_RULER_SAMPLES_PER_TASK_LENGTH,
        "inference_samples": 3900,
    }


def _configure_bts_suite_profile(
    suite: str | None,
    tasks: list[str],
    attention: str,
    *,
    synchronize_modules: bool = True,
):
    """Apply the suite-specific block threshold before BTS is prepared."""
    if attention != "bts":
        return None

    is_ruler = suite == "ruler" or _uses_ruler(tasks)
    block_threshold = 0.95 if is_ruler else 0.93
    token_ratio = 0.4
    dense_tail_tokens = 256
    if synchronize_modules:
        from bts import config as bts_config

        if float(bts_config.TOKEN_RATIO) != token_ratio:
            raise RuntimeError("the frozen BTS token ratio must remain 0.4")
        if int(bts_config.DENSE_TAIL_TOKENS) != dense_tail_tokens:
            raise RuntimeError("the frozen BTS dense tail must remain Tail256")
        bts_config.LOW_FREQ_THRESHOLD = block_threshold
        bts_config.HIGH_FREQ_THRESHOLD = block_threshold

        # block_selection imports these numbers by value, so changing only
        # config.py is insufficient when this process has already imported BTS.
        from bts import block_selection

        block_selection.LOW_FREQ_THRESHOLD = block_threshold
        block_selection.HIGH_FREQ_THRESHOLD = block_threshold
    os.environ["LOW_FREQ_THRESHOLD"] = str(block_threshold)
    os.environ["HIGH_FREQ_THRESHOLD"] = str(block_threshold)
    return {
        "block_low_freq_threshold": block_threshold,
        "block_high_freq_threshold": block_threshold,
        "token_topp": 0.93,
        "token_ratio": token_ratio,
        "dense_tail_tokens": dense_tail_tokens,
        "module_globals_synchronized": synchronize_modules,
        "changed_fields": [
            "block_low_freq_threshold",
            "block_high_freq_threshold",
        ],
        "preserved_fields": ["token_topp", "token_ratio", "dense_tail_tokens"],
    }


def _model_contract(
    model: str,
    revision: str,
    max_length: int,
    *,
    require_yarn4: bool = False,
    require_qwen3_8b: bool = False,
) -> dict:
    """Load and validate the context contract before allocating model weights."""
    from transformers import AutoConfig

    local = Path(model).expanduser()
    load_kwargs = {"trust_remote_code": False}
    if not local.is_dir():
        load_kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(model, **load_kwargs)
    model_type = getattr(config, "model_type", None)
    if model_type not in {"qwen3", "llama"}:
        raise RuntimeError(
            "the evaluation runtime supports Qwen3 and Llama text models only"
        )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = int(getattr(config, "hidden_size", 0))
        num_heads = int(getattr(config, "num_attention_heads", 0))
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise RuntimeError("could not derive head_dim from the model config")
        head_dim = hidden_size // num_heads
    head_dim = int(head_dim)
    if head_dim != 128:
        raise RuntimeError("the frozen evaluation runtime requires head_dim=128")
    if require_qwen3_8b:
        if model_type != "qwen3":
            raise RuntimeError("formal evaluation requires the frozen Qwen3-8B model")
        expected_shape = {
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        actual_shape = {name: int(getattr(config, name, -1)) for name in expected_shape}
        if actual_shape != expected_shape:
            raise RuntimeError(
                "formal evaluation requires the frozen Qwen3-8B architecture "
                f"{expected_shape}, got {actual_shape}"
            )

    configured_length = int(getattr(config, "max_position_embeddings", 0))
    if configured_length < max_length:
        hint = (
            "; prepare a 128K YaRN overlay with eval/common/prepare_yarn_model.py"
            if model_type == "qwen3"
            else "; choose a Llama checkpoint/config with sufficient native context"
        )
        raise RuntimeError(
            f"model supports {configured_length} positions, but --max-length={max_length}"
            f"{hint}"
        )

    rope = getattr(config, "rope_scaling", None)
    if require_yarn4:
        if model_type != "qwen3":
            raise RuntimeError("the frozen YaRN4 contract requires a Qwen3 checkpoint")
        expected = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
        actual = {
            "rope_type": (rope or {}).get("rope_type", (rope or {}).get("type")),
            "factor": (rope or {}).get("factor"),
            "original_max_position_embeddings": (rope or {}).get(
                "original_max_position_embeddings"
            ),
        }
        if actual != expected:
            raise RuntimeError(
                "128K formal evaluation requires the frozen YaRN4 contract "
                f"{expected}, got {actual}; use eval/common/prepare_yarn_model.py"
            )
        if configured_length != 131072:
            raise RuntimeError(
                "the frozen YaRN4 contract requires max_position_embeddings=131072"
            )

    config_path = local / "config.json" if local.is_dir() else None
    overlay_manifest = local / "bts_yarn_manifest.json" if local.is_dir() else None
    return {
        "model_type": model_type,
        "architectures": list(getattr(config, "architectures", None) or []),
        "head_dim": head_dim,
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0)),
        "max_position_embeddings": configured_length,
        "rope_scaling": rope,
        "commit_hash": getattr(config, "_commit_hash", None),
        "config_sha256": (
            _sha256(config_path) if config_path and config_path.is_file() else None
        ),
        "overlay_manifest_sha256": (
            _sha256(overlay_manifest)
            if overlay_manifest and overlay_manifest.is_file()
            else None
        ),
    }


def _uses_ruler(tasks: list[str]) -> bool:
    """Return whether a selection contains a bundled or official RULER task."""
    return (
        "ruler" in tasks
        or bool(RULER_LEAVES.intersection(tasks))
        or _uses_official_ruler(tasks)
    )


def _uses_official_ruler(tasks: list[str]) -> bool:
    return OFFICIAL_RULER_GROUP in tasks or bool(
        OFFICIAL_RULER_LEAVES.intersection(tasks)
    )


def _is_complete_official_ruler_selection(tasks: list[str]) -> bool:
    return tasks == [OFFICIAL_RULER_GROUP] or (
        len(tasks) == len(OFFICIAL_RULER_LEAVES) and set(tasks) == OFFICIAL_RULER_LEAVES
    )


def _validate_formal_results(tasks: list[str], results: dict) -> dict[str, object]:
    """Reject a nominally formal run that did not finish its frozen population."""
    n_samples = results.get("n-samples")
    samples = results.get("samples")
    if not isinstance(n_samples, dict) or not isinstance(samples, dict):
        raise RuntimeError("formal evaluation returned no auditable sample population")

    counts: dict[str, int] = {}
    for task_name, row in n_samples.items():
        if not isinstance(row, dict) or "effective" not in row:
            raise RuntimeError(f"formal sample count is malformed for {task_name}")
        counts[task_name] = int(row["effective"])

    if "longbench" in tasks:
        expected_tasks = LONG_BENCH_LEAVES
        expected_total = 4750
        protocol = "longbench_21_tasks_4750_samples"
    elif _uses_official_ruler(tasks):
        expected_tasks = OFFICIAL_RULER_LEAVES
        expected_total = 3900
        protocol = "official_fixed_ruler_13_tasks_3_lengths_3900_samples"
        if any(counts.get(task) != 300 for task in expected_tasks):
            raise RuntimeError(
                "formal official_fixed RULER did not return exactly 300 samples per task"
            )
    elif "ruler" in tasks:
        expected_tasks = RULER_LEAVES
        expected_total = 1300
        protocol = "ruler_13_tasks_100_samples_one_length"
        if any(counts.get(task) != 100 for task in expected_tasks):
            raise RuntimeError(
                "formal RULER did not return exactly 100 samples per task"
            )
    else:
        return {
            "protocol": "full_selected_tasks",
            "leaf_tasks": len(counts),
            "effective_samples": sum(counts.values()),
        }

    if set(counts) != expected_tasks:
        missing = sorted(expected_tasks - set(counts))
        extra = sorted(set(counts) - expected_tasks)
        raise RuntimeError(
            f"formal {protocol} task set mismatch; missing={missing}, extra={extra}"
        )
    if sum(counts.values()) != expected_total:
        raise RuntimeError(
            f"formal {protocol} expected {expected_total} effective samples, "
            f"got {sum(counts.values())}"
        )
    if set(samples) != expected_tasks:
        raise RuntimeError(f"formal {protocol} per-sample logs are incomplete")
    for task_name in expected_tasks:
        if len(samples[task_name]) != counts[task_name]:
            raise RuntimeError(
                f"formal {task_name} log count does not match effective samples"
            )
    length_counts = None
    if _uses_official_ruler(tasks):
        expected_length_counts = {
            length: OFFICIAL_RULER_SAMPLES_PER_TASK_LENGTH
            for length in OFFICIAL_RULER_LENGTHS
        }
        length_counts = {}
        for task_name in sorted(expected_tasks):
            observed = Counter()
            for sample in samples[task_name]:
                doc = sample.get("doc") if isinstance(sample, dict) else None
                if not isinstance(doc, dict) or "max_length" not in doc:
                    raise RuntimeError(
                        f"formal {task_name} sample has no doc.max_length"
                    )
                try:
                    observed[int(doc["max_length"])] += 1
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"formal {task_name} has an invalid doc.max_length"
                    ) from error
            observed_dict = dict(sorted(observed.items()))
            if observed_dict != expected_length_counts:
                raise RuntimeError(
                    f"formal {task_name} length population mismatch; "
                    f"expected={expected_length_counts}, got={observed_dict}"
                )
            length_counts[task_name] = {
                str(length): count for length, count in observed_dict.items()
            }
    validated = {
        "protocol": protocol,
        "leaf_tasks": len(counts),
        "effective_samples": expected_total,
    }
    if length_counts is not None:
        validated["samples_per_task_by_length"] = length_counts
    return validated


def _validate_formal_args(
    args: argparse.Namespace,
    tasks: list[str],
    parser: argparse.ArgumentParser,
) -> None:
    if not args.formal:
        return
    if _uses_official_ruler(tasks) and not _is_complete_official_ruler_selection(tasks):
        parser.error(
            "--formal official_fixed RULER requires the complete 13-task group; "
            "use --tasks ruler, or omit --formal for a selected task set"
        )
    if args.limit is not None:
        parser.error("--formal forbids --limit; run the complete task set")
    if str(args.batch_size) != "1":
        parser.error("--formal requires --batch-size 1")
    if args.no_log_samples:
        parser.error("--formal requires per-sample logs")
    if not args.apply_chat_template:
        parser.error("--formal requires --apply-chat-template for the frozen protocol")
    if args.parallelize:
        parser.error("--formal requires one GPU and forbids --parallelize")
    if not str(args.device).startswith("cuda"):
        parser.error("--formal requires a CUDA device")
    if args.dtype != "bfloat16":
        parser.error("--formal requires --dtype bfloat16")
    if args.enable_thinking:
        parser.error("--formal requires Qwen3 thinking to remain disabled")
    if args.num_fewshot is not None:
        parser.error("--formal uses each frozen task's own few-shot setting")
    expected_seeds = (0, 1234, 1234, 1234)
    actual_seeds = (
        args.random_seed,
        args.numpy_random_seed,
        args.torch_random_seed,
        args.fewshot_random_seed,
    )
    if actual_seeds != expected_seeds:
        parser.error(
            "--formal requires seeds random=0,numpy=1234,torch=1234,fewshot=1234"
        )
    if (
        args.model
        and not Path(args.model).expanduser().is_dir()
        and args.revision == "main"
    ):
        parser.error("--formal requires a pinned --revision for a remote model ID")
    if _uses_ruler(tasks):
        if args.ruler_samples_per_task != 100:
            parser.error("formal RULER requires --ruler-samples-per-task 100")
        if _uses_official_ruler(tasks) and args.max_length != 131072:
            parser.error("formal official_fixed RULER requires --max-length 131072")
        if not _uses_official_ruler(tasks) and args.max_length not in (
            32768,
            65536,
            131072,
        ):
            parser.error("formal RULER length must be 32768, 65536, or 131072")
        if not args.confirm_run_unsafe_code:
            parser.error("formal RULER requires --confirm-run-unsafe-code")
    if "longbench" in tasks and args.max_length != 32768:
        parser.error("formal full LongBench requires --max-length 32768")


def _selected_tasks(args: argparse.Namespace, *, suite: str | None = None) -> list[str]:
    if args.tasks is not None:
        tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
        if not tasks:
            raise ValueError("--tasks did not contain a task name")
    else:
        tasks = list(SCENARIOS[args.scenario])
    if len(tasks) != len(set(tasks)):
        raise ValueError("--tasks contains duplicates")
    if suite != "ruler":
        return tasks

    normalized: list[str] = []
    for task in tasks:
        if task in {"ruler", OFFICIAL_RULER_GROUP}:
            official = OFFICIAL_RULER_GROUP
        elif task in RULER_LEAVES:
            official = f"official_fixed_{task}"
        elif task in OFFICIAL_RULER_LEAVES:
            official = task
        else:
            raise ValueError(f"unsupported frozen RULER task: {task}")
        if official in normalized:
            raise ValueError(
                "--tasks contains duplicates after frozen-task normalization"
            )
        normalized.append(official)
    if OFFICIAL_RULER_GROUP in normalized and len(normalized) != 1:
        raise ValueError("the complete RULER group cannot be mixed with leaf tasks")
    return normalized


def main(argv: list[str] | None = None, *, suite: str | None = None) -> int:
    parser = _parser(suite)
    args = parser.parse_args(argv)
    if args.max_length <= 0 or args.max_length > 131072:
        parser.error("--max-length must be in [1, 131072]")
    if args.ruler_samples_per_task <= 0:
        parser.error("--ruler-samples-per-task must be positive")
    if args.num_fewshot is not None and args.num_fewshot < 0:
        parser.error("--num-fewshot must be nonnegative")
    if not args.list_tasks:
        if not args.model:
            parser.error("--model is required unless --list-tasks is used")
        if args.output is None:
            parser.error("--output is required for an evaluation")

    try:
        selected = _selected_tasks(args, suite=suite)
    except ValueError as error:
        parser.error(str(error))
    _validate_formal_args(args, selected, parser)
    official_ruler_suite = None
    if _uses_official_ruler(selected):
        if args.max_length != 131072:
            parser.error("official_fixed RULER requires --max-length 131072")
        try:
            official_ruler_suite = _official_ruler_suite(args.ruler_suite)
        except RuntimeError as error:
            parser.error(str(error))
    args.attention = normalize_name(args.attention)
    # Baseline modules read their knobs at import time. Lock them before the
    # bundled harness (and therefore Torch) is imported.
    effective_method_environment = configure_environment(args.attention)
    bts_profile = _configure_bts_suite_profile(
        suite,
        selected,
        args.attention,
        synchronize_modules=not args.list_tasks,
    )
    if bts_profile is not None:
        effective_method_environment.update(
            {
                "LOW_FREQ_THRESHOLD": str(bts_profile["block_low_freq_threshold"]),
                "HIGH_FREQ_THRESHOLD": str(bts_profile["block_high_freq_threshold"]),
            }
        )
    attention_contract = method_contract(args.attention)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    metadata = {
        "max_seq_lengths": (
            list(OFFICIAL_RULER_LENGTHS)
            if official_ruler_suite is not None
            else [args.max_length]
        ),
        "tokenizer": args.model,
        "pretrained": args.model,
        "max_length": args.max_length,
    }
    if _uses_ruler(selected) and official_ruler_suite is None:
        metadata["num_samples"] = args.ruler_samples_per_task
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(
        metadata=metadata,
        include_path=(
            official_ruler_suite["tasks_path"]
            if official_ruler_suite is not None
            else None
        ),
        include_defaults=official_ruler_suite is None,
    )

    if args.list_tasks:
        print("\n".join(task_manager.all_tasks))
        return 0

    missing = [task for task in selected if task not in task_manager.all_tasks]
    if missing:
        parser.error("tasks are not available to lm-eval: " + ", ".join(missing))

    reference_key = None
    if official_ruler_suite is not None or suite == "ruler":
        reference_key = "ruler"
    elif "longbench" in selected or suite == "longbench":
        reference_key = "longbench"

    verified_reference = (
        _verified_reference(reference_key) if reference_key is not None else None
    )
    if reference_key == "ruler" and verified_reference is not None:
        verified_reference["task_set_matches_reference"] = (
            _is_complete_official_ruler_selection(selected)
        )
        if not verified_reference["task_set_matches_reference"]:
            verified_reference["task_set_note"] = (
                "The historical aggregate covers all 13 tasks; it is provenance "
                "only and is not the aggregate for this selected task set."
            )

    runtime = {
        "schema_version": 1,
        "scenario": suite or (args.scenario if not args.tasks else "custom"),
        "requested_tasks": args.tasks,
        "tasks": selected,
        "attention": args.attention,
        "method_display_name": attention_contract["display_name"],
        "attention_method": attention_contract,
        "effective_method_environment": effective_method_environment,
        "bts_suite_profile": bts_profile,
        "model": args.model,
        "revision": args.revision,
        "max_length": args.max_length,
        "ruler_samples_per_task": (
            args.ruler_samples_per_task if _uses_ruler(selected) else None
        ),
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": args.dtype,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "log_samples": not args.no_log_samples,
        "apply_chat_template": args.apply_chat_template,
        "enable_thinking": args.enable_thinking,
        "formal": args.formal,
        "seeds": {
            "random": args.random_seed,
            "numpy": args.numpy_random_seed,
            "torch": args.torch_random_seed,
            "fewshot": args.fewshot_random_seed,
        },
        "harness": "bundled lm-eval 0.4.9.2 compact runtime",
        "data_contract": DATA_CONTRACT,
        "ruler_suite": official_ruler_suite,
        "verified_reference": verified_reference,
    }

    if args.formal and args.output.exists():
        if not args.output.is_dir():
            parser.error("--output must be a directory")
        if any(args.output.iterdir()):
            parser.error("--formal requires a new or empty --output directory")

    runtime["model_contract"] = _model_contract(
        args.model,
        args.revision,
        args.max_length,
        require_yarn4=args.formal and _uses_ruler(selected),
        require_qwen3_8b=args.formal,
    )

    from baselines import install_runtime_hooks, prepare_attention

    model_type = runtime["model_contract"]["model_type"]
    if args.enable_thinking and model_type != "qwen3":
        parser.error("--enable-thinking is supported only for Qwen3 models")
    prepared = prepare_attention(
        args.attention,
        args.model,
        model_type=model_type,
        revision=args.revision,
    )
    runtime["attention_method"] = prepared.contract
    runtime["patched_attention_classes"] = prepared.patched_attention_classes
    runtime["runtime_hooks"] = install_runtime_hooks(prepared)
    if args.attention == "bts":
        runtime["bts_method"] = prepared.contract["method_details"]

    args.output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.output / "runtime_config.json", runtime)

    model_args = {
        "pretrained": args.model,
        "tokenizer": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "parallelize": args.parallelize,
        "trust_remote_code": args.trust_remote_code,
    }
    if model_type == "qwen3":
        model_args["enable_thinking"] = args.enable_thinking
    model_args.update(prepared.model_args)

    from lm_eval import simple_evaluate
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.utils import make_table

    tracker = EvaluationTracker(output_path=str(args.output))
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=selected,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        log_samples=not args.no_log_samples,
        evaluation_tracker=tracker,
        apply_chat_template=args.apply_chat_template,
        task_manager=task_manager,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        metadata=metadata,
        random_seed=args.random_seed,
        numpy_random_seed=args.numpy_random_seed,
        torch_random_seed=args.torch_random_seed,
        fewshot_random_seed=args.fewshot_random_seed,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results")

    if args.formal:
        runtime["formal_result_contract"] = _validate_formal_results(selected, results)
        _write_json_atomic(args.output / "runtime_config.json", runtime)

    samples = results.pop("samples", None)
    results["bts_submission"] = runtime
    tracker.save_results_aggregated(results=results, samples=samples)
    if samples is not None:
        for task_name in results["configs"]:
            tracker.save_results_samples(task_name, samples[task_name])

    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
