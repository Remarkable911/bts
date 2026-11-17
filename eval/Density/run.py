#!/usr/bin/env python3
"""Replay the parent experiment's audited fixed-10 128K Density protocol.

The default protocol evaluates source indices 0,11,...,99 from each of five
classes.  It reads the exact chat prompts recorded in the frozen official
lm-eval output logs, repeats each prompt's token row to exactly 131072 tokens,
captures every attention-layer input during one *dense* base-Qwen forward, and
then replays BTS attention alone over those 36 dense-upstream inputs.  No text
generation is part of this measurement.

The external RULER suite and the bundled 50-row parent-plan manifest are
immutable inputs.  Their whole-file and selected-row identities are checked
before a sample can be measured or resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
SUBMISSION_ROOT = EVAL_ROOT.parent

# Always prefer the frozen submission.  In particular, do not make a sibling
# development checkout importable.
sys.path.insert(0, str(SUBMISSION_ROOT))

from baselines.registry import (
    ATTENTION_NAMES,
    DENSITY_DEFINITIONS,
    configure_environment,
    method_contract,
    normalize_name,
)

MAX_LENGTH = 131072
DEFAULT_SEED = 42
TARGET_DENSITY = 0.10
SCHEMA_VERSION = 1
METHOD_DISPLAY_NAME = "BTS"
PROTOCOL_NAME = "selected5_official_output_exact128k_dense_replay_v2"
FIXED_SAMPLE_INDICES = tuple(range(0, 100, 11))
DEFAULT_SAMPLES_PER_TASK = len(FIXED_SAMPLE_INDICES)
REFERENCE_MODEL_ID = "Qwen/Qwen3-8B"
REFERENCE_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
REFERENCE_CONFIG_FILE_SHA256 = (
    "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"
)
REFERENCE_TOKENIZER_JSON_SHA256 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)
REFERENCE_TOKENIZER_CONFIG_SHA256 = (
    "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
)
FIXED_INPUT_MANIFEST = HERE / "fixed10_input_manifest.json"
EXPECTED_FIXED_INPUT_MANIFEST_SHA256 = (
    "b74b9ae15df304dd3250532131838da617108edef78c5de8078b0a256098bfd3"
)

DEFAULT_RULER_SUITE = (
    SUBMISSION_ROOT.parent
    / "results"
    / "ruler"
    / "topk040_tail256_official_protocol_20261105"
)
EXPECTED_RULER_SUITE_MANIFEST_SHA256 = (
    "7c594b7eb6f03f8bfd5cd514a24fb9eea046368640cc5c6aac406fdc569818aa"
)
EXPECTED_RULER_SUITE_FILES_SHA256 = (
    "cdfa2509a3338dd01adedc2fb49fa41a041a1f25902aee4c168ef35b667df12b"
)

HISTORICAL_DENSITY_REFERENCE = {
    "reference_only": True,
    "source": "work/density_multiclass_pilot10_topk040_summary_20261107/summary.json",
    "source_sha256": (
        "9b2885529829062fa23d1bf747a09dd317a5eb6ee9baed1b61bfe3cd0e2663ca"
    ),
    "configuration": {
        "historical_method": "PRISM+Token (project main method)",
        "submission_method": "BTS",
        "model": REFERENCE_MODEL_ID,
        "model_revision": REFERENCE_MODEL_REVISION,
        "model_role": (
            "base Qwen3 dense-upstream capture; the suite YaRN4 model only "
            "produced the logged official prompts"
        ),
        "block_retention_thresholds": {"low": 0.93, "high": 0.93},
        "token_ratio": 0.4,
        "dense_tail_tokens": 256,
        "sequence_length": MAX_LENGTH,
        "samples_per_task": DEFAULT_SAMPLES_PER_TASK,
        "sample_indices": list(FIXED_SAMPLE_INDICES),
        "input_length_policy": (
            "repeat whole prompt token row, then truncate to exactly 131072"
        ),
        "execution_scope": (
            "dense full-model capture followed by attention-only replay over "
            "all 36 layers"
        ),
        "generation_performed": False,
        "runtime_prism_config_sha256": (
            "7b2d0c54dd7b048de153025214400fd7bca5e72fc55885443f53e28f0e3a8449"
        ),
        "execution_source_lock_sha256": (
            "408e50f6af31d606ab8e5065f39edf6c4ffbc6cd6e61dea7431d6c334c757492"
        ),
    },
    "mean_density_percent": {
        "niah_single_1": 10.910451104388974,
        "niah_multikey_1": 7.434323624608316,
        "ruler_fwe": 11.189820162972138,
        "ruler_qa_hotpot": 8.182113209132579,
        "niah_multiquery": 7.44724488297129,
    },
    "class_average_density_at_most_0_10": {
        "niah_single_1": False,
        "niah_multikey_1": True,
        "ruler_fwe": False,
        "ruler_qa_hotpot": True,
        "niah_multiquery": True,
    },
    "new_run_reproduction_claim": False,
    "note": (
        "These audited fixed-10 values are reference metadata only. NIAH "
        "single-1 and RULER frequent-word extraction exceed the 0.10 density "
        "target. A new run must report its own measured summary and is not "
        "presumed to reproduce the historical values."
    ),
}

# These distributions affect either the RULER input builders, the bundled
# lm-eval adapter, model/tokenizer loading, or the frozen attention runtime.
# Keep the mapping explicit so runtime_config.json is useful without relying on
# an enormous and noisy ``pip freeze`` from the host environment.
RUNTIME_DISTRIBUTIONS = (
    "torch",
    "triton",
    "ninja",
    "transformers",
    "datasets",
    "nltk",
    "wonderwords",
    "tokenizers",
    "numpy",
    "huggingface-hub",
    "accelerate",
    "safetensors",
    "requests",
    "pyarrow",
    "pandas",
    "fsspec",
    "flash-attn",
    "einops",
    "minference",
    "block-sparse-attn",
)

# Public-source resolver for the one QA workload in the Density five.
QA_SOURCE_SPECS = {
    "ruler_qa_hotpot": {
        "environment": "RULER_HOTPOTQA_PATH",
        "filename": "hotpot_dev_distractor_v1.json",
        "sha256": "7f0cd50fe0b6abc95596a8e0b9c4de931a2f6d61a3e75eec18632bfa7a9b1b76",
        "urls": (
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        ),
    },
}

TASK_SPECS: tuple[tuple[str, str], ...] = (
    ("niah_single_1", "NIAH single-1"),
    ("niah_multikey_1", "NIAH multi-key-1"),
    ("ruler_fwe", "RULER frequent-word extraction"),
    ("ruler_qa_hotpot", "RULER QA-Hotpot"),
    ("niah_multiquery", "NIAH multi-query"),
)
TASK_LABELS = dict(TASK_SPECS)


def protocol_sample_indices(samples_per_task: int) -> tuple[int, ...]:
    """Return the frozen panel indices for the audited default protocol."""
    if samples_per_task <= 0 or samples_per_task > 100:
        raise ValueError("samples_per_task must be between 1 and 100")
    if samples_per_task == DEFAULT_SAMPLES_PER_TASK:
        return FIXED_SAMPLE_INDICES
    # Non-default counts are supported for development only.  They are a
    # deterministic prefix and never qualify as the audited fixed-10 protocol.
    return tuple(range(samples_per_task))


def fixed_input_manifest() -> dict[str, Any]:
    """Load and validate the 50-row identity manifest copied from parent plans."""
    if not FIXED_INPUT_MANIFEST.is_file():
        raise FileNotFoundError(
            f"fixed Density input manifest is missing: {FIXED_INPUT_MANIFEST}"
        )
    observed_sha256 = file_sha256(FIXED_INPUT_MANIFEST)
    try:
        payload = json.loads(FIXED_INPUT_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"fixed Density input manifest is unreadable: {FIXED_INPUT_MANIFEST}"
        ) from error
    errors: list[str] = []
    if observed_sha256 != EXPECTED_FIXED_INPUT_MANIFEST_SHA256:
        errors.append(
            "manifest sha256 mismatch: "
            f"{observed_sha256} != {EXPECTED_FIXED_INPUT_MANIFEST_SHA256}"
        )
    if payload.get("schema_version") != 1:
        errors.append("schema_version is not 1")
    if payload.get("sequence_length") != MAX_LENGTH:
        errors.append("sequence_length is not 131072")
    if payload.get("fixed_indices") != list(FIXED_SAMPLE_INDICES):
        errors.append("fixed_indices do not match 0,11,...,99")
    expected_model = {
        "id": REFERENCE_MODEL_ID,
        "revision": REFERENCE_MODEL_REVISION,
        "config_file_sha256": REFERENCE_CONFIG_FILE_SHA256,
        "tokenizer_json_sha256": REFERENCE_TOKENIZER_JSON_SHA256,
        "tokenizer_config_sha256": REFERENCE_TOKENIZER_CONFIG_SHA256,
    }
    if payload.get("model") != expected_model:
        errors.append("base-model identity does not match the parent protocol")
    task_payload = payload.get("tasks")
    expected_tasks = [task for task, _ in TASK_SPECS]
    if not isinstance(task_payload, dict) or list(task_payload) != expected_tasks:
        errors.append("task order does not match the selected five")
    else:
        for task in expected_tasks:
            rows = task_payload[task].get("inputs")
            if not isinstance(rows, list) or len(rows) != len(FIXED_SAMPLE_INDICES):
                errors.append(f"{task} does not contain ten input identities")
                continue
            indices = [row.get("index") for row in rows]
            if indices != list(FIXED_SAMPLE_INDICES):
                errors.append(f"{task} input indices changed: {indices}")
            for row in rows:
                if row.get("task") != task:
                    errors.append(f"{task} contains a row for {row.get('task')!r}")
                original = row.get("token_count_before_exact_length")
                appended = row.get("tokens_appended_from_prompt_prefix")
                if (
                    not isinstance(original, int)
                    or not isinstance(appended, int)
                    or original + appended != MAX_LENGTH
                    or row.get("tokens_truncated") != 0
                    or row.get("benchmark_token_count") != MAX_LENGTH
                ):
                    errors.append(
                        f"{task}[{row.get('index')}] has a changed exact-length plan"
                    )
                historical_density = row.get("historical_mean_density")
                historical_result = row.get("historical_raw_result_sha256")
                if (
                    not isinstance(historical_density, (int, float))
                    or not math.isfinite(float(historical_density))
                    or not 0.0 < float(historical_density) <= 1.0
                    or not isinstance(historical_result, str)
                    or re.fullmatch(r"[0-9a-f]{64}", historical_result) is None
                ):
                    errors.append(
                        f"{task}[{row.get('index')}] lacks its parent raw result"
                    )
            if all(
                isinstance(row.get("historical_mean_density"), (int, float))
                for row in rows
            ):
                observed_percent = (
                    statistics.fmean(
                        float(row["historical_mean_density"]) for row in rows
                    )
                    * 100.0
                )
                expected_percent = HISTORICAL_DENSITY_REFERENCE["mean_density_percent"][
                    task
                ]
                if observed_percent != expected_percent:
                    errors.append(
                        f"{task} parent sample means aggregate to "
                        f"{observed_percent}, expected {expected_percent}"
                    )
    result = {
        **payload,
        "path": str(FIXED_INPUT_MANIFEST),
        "sha256": observed_sha256,
        "expected_sha256": EXPECTED_FIXED_INPUT_MANIFEST_SHA256,
        "available": True,
        "validation_errors": errors,
        "verified": not errors,
    }
    if errors:
        raise RuntimeError(
            "fixed Density input manifest verification failed: " + "; ".join(errors)
        )
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode an identity object deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_density_reference() -> dict[str, Any]:
    """Return the prior result as reference metadata, never as a new result."""
    source = SUBMISSION_ROOT.parent / str(HISTORICAL_DENSITY_REFERENCE["source"])
    observed_sha256 = file_sha256(source) if source.is_file() else None
    return {
        **HISTORICAL_DENSITY_REFERENCE,
        "resolved_source": str(source.resolve(strict=False)),
        "source_present": source.is_file(),
        "observed_source_sha256": observed_sha256,
        "source_hash_verified": (
            observed_sha256 == HISTORICAL_DENSITY_REFERENCE["source_sha256"]
        ),
    }


def bts_frozen_config_contract() -> dict[str, Any]:
    """Load the data-only BTS config without importing the torch runtime."""
    config_path = SUBMISSION_ROOT / "bts" / "config.py"
    spec = importlib.util.spec_from_file_location(
        "_density_bts_frozen_config",
        config_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen BTS config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    describe = getattr(module, "describe", None)
    if not callable(describe):
        raise RuntimeError(f"frozen BTS config has no describe(): {config_path}")
    return dict(describe())


def ruler_suite_contract(
    path: Path,
    tasks: Sequence[str],
) -> dict[str, Any]:
    """Inspect and verify the frozen official RULER suite."""
    requested = path.expanduser()
    suite = requested.resolve(strict=False)
    manifest_path = suite / "manifest.json"
    frozen_files_path = suite / "frozen_files_sha256.json"
    inventory_path = suite / "data_inventory.json"
    tasks_path = suite / "tasks"
    outputs_path = suite / "outputs"
    task_mapping = {task: f"official_fixed_{task}" for task in tasks}
    parent_inputs = fixed_input_manifest()

    required_paths = {
        "suite": suite,
        "manifest": manifest_path,
        "frozen_files_manifest": frozen_files_path,
        "data_inventory": inventory_path,
        "tasks": tasks_path,
        "outputs": outputs_path,
        "task_scoring": tasks_path / "scoring.py",
    }
    for base_task, suite_task in task_mapping.items():
        required_paths[f"task:{suite_task}"] = tasks_path / f"{suite_task}.yaml"
        required_paths[f"data:{base_task}:128k"] = (
            suite / "data" / "128k" / f"{base_task}.jsonl"
        )
        if parent_inputs.get("verified") is True:
            output_name = parent_inputs["tasks"][base_task]["official_output_filename"]
            required_paths[f"output:{base_task}"] = outputs_path / output_name

    missing = [
        f"{label}={candidate}"
        for label, candidate in required_paths.items()
        if not (
            candidate.is_dir()
            if label in {"suite", "tasks", "outputs"}
            else candidate.is_file()
        )
    ]
    base_contract: dict[str, Any] = {
        "requested_path": str(requested),
        "resolved_path": str(suite),
        "manifest_path": str(manifest_path),
        "tasks_path": str(tasks_path),
        "outputs_path": str(outputs_path),
        "task_mapping": task_mapping,
        "fixed_input_manifest": {
            key: parent_inputs.get(key)
            for key in (
                "path",
                "sha256",
                "expected_sha256",
                "available",
                "validation_errors",
                "verified",
            )
        },
        "missing": missing,
        "available": not missing,
        "verified": False,
    }
    if missing:
        raise FileNotFoundError(
            "formal density requires the frozen official RULER suite; missing: "
            + "; ".join(missing)
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen_files = json.loads(frozen_files_path.read_text(encoding="utf-8"))
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"formal density RULER suite metadata is unreadable: {suite}"
        ) from error

    manifest_sha256 = file_sha256(manifest_path)
    frozen_files_sha256 = file_sha256(frozen_files_path)
    validation_errors: list[str] = []
    if manifest_sha256 != EXPECTED_RULER_SUITE_MANIFEST_SHA256:
        validation_errors.append(
            "manifest sha256 mismatch: "
            f"{manifest_sha256} != {EXPECTED_RULER_SUITE_MANIFEST_SHA256}"
        )
    if frozen_files_sha256 != EXPECTED_RULER_SUITE_FILES_SHA256:
        validation_errors.append(
            "frozen-files manifest sha256 mismatch: "
            f"{frozen_files_sha256} != {EXPECTED_RULER_SUITE_FILES_SHA256}"
        )

    expected_manifest_fields = {
        "ready": True,
        "max_length": MAX_LENGTH,
        "samples_per_task_length": 100,
        "apply_chat_template": True,
        "enable_thinking": False,
        "token_ratio": 0.4,
        "dense_tail_tokens": 256,
    }
    for name, expected in expected_manifest_fields.items():
        if manifest.get(name) != expected:
            validation_errors.append(
                f"manifest.{name}={manifest.get(name)!r}, expected {expected!r}"
            )
    if manifest.get("block_retention_thresholds") != {"low": 0.93, "high": 0.93}:
        validation_errors.append(
            "manifest block_retention_thresholds are not low=high=0.93"
        )
    if MAX_LENGTH not in list(manifest.get("inference_lengths", ())):
        validation_errors.append("manifest does not include 131072-token inference")

    manifest_tasks = set(manifest.get("tasks", ()))
    manifest_task_names = set(manifest.get("task_names", ()))
    for base_task, suite_task in task_mapping.items():
        if base_task not in manifest_tasks:
            validation_errors.append(f"manifest is missing base task {base_task}")
        if suite_task not in manifest_task_names:
            validation_errors.append(f"manifest is missing suite task {suite_task}")

    selected_files: dict[str, dict[str, Any]] = {}
    selected_relative_paths = ["tasks/scoring.py"]
    selected_relative_paths.extend(
        f"tasks/{suite_task}.yaml" for suite_task in task_mapping.values()
    )
    selected_relative_paths.extend(
        f"data/128k/{base_task}.jsonl" for base_task in task_mapping
    )
    inventory_tasks = inventory.get("tasks", {})
    for relative in selected_relative_paths:
        candidate = suite / relative
        observed = file_sha256(candidate)
        expected = frozen_files.get(relative)
        selected_files[relative] = {
            "sha256": observed,
            "frozen_sha256": expected,
            "verified": isinstance(expected, str) and observed == expected,
        }
        if expected is None:
            validation_errors.append(
                f"frozen-files manifest has no entry for {relative}"
            )
        elif observed != expected:
            validation_errors.append(f"frozen file hash mismatch: {relative}")
        if relative.startswith("data/128k/"):
            base_task = Path(relative).stem
            inventory_row = inventory_tasks.get(base_task, {}).get(str(MAX_LENGTH), {})
            if inventory_row.get("path") != relative:
                validation_errors.append(
                    f"data inventory path mismatch for {base_task}: "
                    f"{inventory_row.get('path')!r}"
                )
            if inventory_row.get("samples") != 100:
                validation_errors.append(
                    f"data inventory sample count for {base_task} is not 100"
                )
            if inventory_row.get("sha256") != observed:
                validation_errors.append(
                    f"data inventory hash mismatch for {base_task}"
                )

    official_outputs: dict[str, dict[str, Any]] = {}
    if parent_inputs.get("verified") is not True:
        validation_errors.append("fixed parent-plan input manifest is unverified")
    else:
        for base_task in task_mapping:
            task_plan = parent_inputs["tasks"][base_task]
            output_name = str(task_plan["official_output_filename"])
            output_path = outputs_path / output_name
            observed = file_sha256(output_path)
            expected = str(task_plan["official_output_file_sha256"])
            official_outputs[base_task] = {
                "path": str(output_path),
                "filename": output_name,
                "sha256": observed,
                "expected_sha256": expected,
                "verified": observed == expected,
            }
            if observed != expected:
                validation_errors.append(
                    f"official output hash mismatch for {base_task}: "
                    f"{observed} != {expected}"
                )

    contract = {
        **base_contract,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "expected_manifest_sha256": EXPECTED_RULER_SUITE_MANIFEST_SHA256,
        "frozen_files_manifest_path": str(frozen_files_path),
        "frozen_files_manifest_sha256": frozen_files_sha256,
        "expected_frozen_files_manifest_sha256": (EXPECTED_RULER_SUITE_FILES_SHA256),
        "data_inventory_path": str(inventory_path),
        "data_inventory_sha256": file_sha256(inventory_path),
        "selected_files": selected_files,
        "official_outputs": official_outputs,
        "validation_errors": validation_errors,
        "verified": not validation_errors,
    }
    if validation_errors:
        raise RuntimeError(
            "formal density RULER suite verification failed: "
            + "; ".join(validation_errors)
        )
    return contract


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                        default=_json_default,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_manifest() -> dict[str, Any]:
    """Hash every frozen implementation file that can affect this run."""
    files: set[Path] = {
        Path(__file__).resolve(),
        (HERE / "protocol.py").resolve(),
        FIXED_INPUT_MANIFEST.resolve(),
        (EVAL_ROOT / "efficiency" / "run.py").resolve(),
    }
    for root in (
        SUBMISSION_ROOT / "bts",
        SUBMISSION_ROOT / "baselines",
    ):
        files.update(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".json", ".yaml", ".cu", ".cuh", ".h", ".cpp"}
        )
    entries = {
        str(path.relative_to(SUBMISSION_ROOT)): file_sha256(path)
        for path in sorted(files)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "files": entries,
        "manifest_sha256": canonical_sha256(entries),
    }


def _distribution_version(name: str) -> str | None:
    """Return an installed distribution version without importing it."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_manifest() -> dict[str, Any]:
    """Describe the Python, platform, dependency, and accelerator runtime.

    Reading the compiled CUDA/cuDNN metadata does not allocate a CUDA tensor,
    so this helper is also safe in CPU-only audit and test environments.
    """
    import torch

    libc_name, libc_version = platform.libc_ver()
    return {
        "schema_version": SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "full_version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": {"name": libc_name or None, "version": libc_version or None},
        },
        "packages": {
            name: _distribution_version(name) for name in RUNTIME_DISTRIBUTIONS
        },
        "accelerator_runtime": {
            "torch_compiled_cuda_version": getattr(torch.version, "cuda", None),
            "torch_compiled_hip_version": getattr(torch.version, "hip", None),
            "torch_compiled_cudnn_version": torch.backends.cudnn.version(),
            "cudnn_available": torch.backends.cudnn.is_available(),
            "cuda_available_at_capture": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def gpu_inventory() -> list[dict[str, Any]]:
    """Return the stable physical-GPU inventory reported by nvidia-smi."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("nvidia-smi GPU inventory failed") from error
    rows = []
    for fields in csv.reader(result.stdout.splitlines()):
        if not fields:
            continue
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {fields}")
        index, uuid, name, memory = (field.strip() for field in fields)
        rows.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(memory),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi reported no physical GPUs")
    return rows


def resolve_single_gpu(
    device: str,
    inventory: Sequence[Mapping[str, Any]],
    cuda_visible_devices: str | None,
) -> dict[str, Any]:
    """Map a torch CUDA device to one unambiguous physical GPU."""
    match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
    if match is None:
        raise ValueError(f"invalid CUDA device: {device!r}")
    logical_index = int(match.group(1) or 0)
    by_index = {int(row["physical_index"]): row for row in inventory}
    visible = None
    if cuda_visible_devices is not None:
        visible = [
            item.strip() for item in cuda_visible_devices.split(",") if item.strip()
        ]
        if logical_index >= len(visible):
            raise RuntimeError(
                f"{device} is outside CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}"
            )
        selector = visible[logical_index]
        if selector.isdigit():
            physical = by_index.get(int(selector))
        elif selector.startswith("GPU-"):
            matches = [
                row for row in inventory if str(row["uuid"]).startswith(selector)
            ]
            physical = matches[0] if len(matches) == 1 else None
        else:
            raise RuntimeError(
                "formal density runs do not support MIG or ambiguous "
                f"CUDA_VISIBLE_DEVICES entries: {selector!r}"
            )
    else:
        physical = by_index.get(logical_index)
    if physical is None:
        raise RuntimeError(f"could not resolve {device} to one physical GPU")
    return {
        **dict(physical),
        "torch_device": device,
        "logical_index": logical_index,
        "cuda_visible_devices": cuda_visible_devices,
        "visible_device_count": len(visible) if visible is not None else len(inventory),
    }


def gpu_compute_check(selected_gpu: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Reject compute processes other than this runner on the selected GPU."""
    command = [
        "nvidia-smi",
        f"--id={int(selected_gpu['physical_index'])}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"nvidia-smi process check failed at {label}") from error
    processes = []
    for fields in csv.reader(result.stdout.splitlines()):
        if not fields:
            continue
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi process row: {fields}")
        pid, name, memory = (field.strip() for field in fields)
        if not pid.isdigit():
            continue
        processes.append(
            {
                "pid": int(pid),
                "process_name": name,
                "used_gpu_memory_mib": (int(memory) if memory.isdigit() else memory),
            }
        )
    foreign = [row for row in processes if int(row["pid"]) != os.getpid()]
    return {
        "label": label,
        "physical_index": int(selected_gpu["physical_index"]),
        "gpu_uuid": selected_gpu["uuid"],
        "runner_pid": os.getpid(),
        "processes": processes,
        "foreign_processes": foreign,
        "passed": not foreign,
    }


class GpuAudit:
    """Persist foreign-process checks across restart sessions."""

    def __init__(
        self,
        path: Path,
        *,
        run_identity_sha256: str,
        selected_gpu: Mapping[str, Any],
        preflight: Mapping[str, Any],
    ) -> None:
        self.path = path
        self.run_identity_sha256 = run_identity_sha256
        self.selected_gpu = dict(selected_gpu)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise RuntimeError(f"invalid existing GPU audit: {path}") from error
            if (
                payload.get("run_identity_sha256") != run_identity_sha256
                or payload.get("selected_gpu") != self.selected_gpu
            ):
                raise RuntimeError("GPU audit identity changed on resume")
        else:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "run_identity_sha256": run_identity_sha256,
                "selected_gpu": self.selected_gpu,
                "sessions": [],
            }
        self.payload = payload
        self.session = {
            "session_index": len(payload["sessions"]),
            "runner_pid": os.getpid(),
            "checks": [dict(preflight)],
        }
        payload["sessions"].append(self.session)
        self._write()

    def _write(self) -> None:
        write_json_atomic(self.path, self.payload)

    def check(self, label: str) -> dict[str, Any]:
        result = gpu_compute_check(self.selected_gpu, label=label)
        self.session["checks"].append(result)
        self._write()
        if not result["passed"]:
            write_json_atomic(
                self.path.parent / "formal_invalid.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_identity_sha256": self.run_identity_sha256,
                    "formal_valid": False,
                    "reason": "foreign compute process on selected GPU",
                    "failed_check": result,
                },
            )
            raise RuntimeError(
                f"selected GPU has foreign compute processes at {label}: "
                f"{result['foreign_processes']}"
            )
        return result


def _download_verbatim_json(
    destination: Path,
    *,
    urls: Sequence[str],
    expected_sha256: str,
) -> str | None:
    """Try public raw sources, publishing only the preregistered byte stream."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
        try:
            with requests.get(url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if file_sha256(temporary) != expected_sha256:
                continue
            # Parse before publishing so a matching-but-unreadable cache cannot
            # enter the formal plan (the SHA check is still the authority).
            with temporary.open(encoding="utf-8") as handle:
                json.load(handle)
            os.replace(temporary, destination)
            return url
        except (OSError, ValueError, requests.RequestException):
            pass
        finally:
            temporary.unlink(missing_ok=True)
    return None


def _build_hotpot_from_public_parquet(
    destination: Path, expected_sha256: str
) -> dict[str, Any]:
    """Reproduce the bundled loader's deterministic HotpotQA fallback."""
    import datasets
    from huggingface_hub import hf_hub_download

    parquet = Path(
        hf_hub_download(
            repo_id="hotpotqa/hotpot_qa",
            repo_type="dataset",
            filename="distractor/validation-00000-of-00001.parquet",
            revision="1908d6afbbead072334abe2965f91bd2709910ab",
        )
    )
    validation = datasets.load_dataset(
        "parquet", data_files={"validation": str(parquet)}, split="validation"
    )
    data = [
        {
            "question": row["question"],
            "answer": row["answer"],
            "context": list(zip(row["context"]["title"], row["context"]["sentences"])),
        }
        for row in validation
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.convert-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            # Match qa_utils._cache_json exactly.
            json.dump(data, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        observed = file_sha256(temporary)
        if observed != expected_sha256:
            raise RuntimeError(
                "public HotpotQA conversion does not match the frozen formal "
                f"source: expected {expected_sha256}, got {observed}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "acquisition": "hotpotqa/hotpot_qa public parquet conversion",
        "upstream_parquet_path": str(parquet.resolve()),
        "upstream_parquet_sha256": file_sha256(parquet),
    }


def prepare_qa_sources(output: Path, tasks: Sequence[str]) -> dict[str, Any]:
    """Resolve and hash the exact QA corpora used by the formal protocol."""
    result: dict[str, Any] = {}
    for task in tasks:
        if task not in QA_SOURCE_SPECS:
            continue
        spec = QA_SOURCE_SPECS[task]
        environment = str(spec["environment"])
        configured = os.environ.get(environment)
        acquisition: dict[str, Any]
        if configured:
            path = Path(configured).expanduser().resolve()
            acquisition = {"acquisition": f"preconfigured {environment}"}
        else:
            path = (output / "public_sources" / str(spec["filename"])).resolve()
            if not path.exists():
                url = _download_verbatim_json(
                    path,
                    urls=spec["urls"],
                    expected_sha256=str(spec["sha256"]),
                )
                if url is None and task == "ruler_qa_hotpot":
                    _build_hotpot_from_public_parquet(path, str(spec["sha256"]))
                elif url is None:
                    raise RuntimeError(
                        f"could not download the frozen public source for {task}; "
                        f"set {environment} to the original JSON file"
                    )
            # This value deliberately describes the stable resolved state, not
            # whether this particular process populated or reused the cache.
            # Consequently runtime_config.json remains identical on resume.
            acquisition = {
                "acquisition": "verified restart-safe output cache",
                "public_origin": "HotpotQA distractor validation",
            }
        if not path.is_file():
            raise FileNotFoundError(f"{task} public source is absent: {path}")
        observed = file_sha256(path)
        expected = str(spec["sha256"])
        if observed != expected:
            raise RuntimeError(
                f"{task} public source hash mismatch: expected {expected}, "
                f"got {observed} ({path})"
            )
        # Force the bundled task to consume the verified bytes, rather than a
        # mutable implicit datasets cache or fallback endpoint.
        os.environ[environment] = str(path)
        result[task] = {
            "path": str(path),
            "sha256": observed,
            "expected_sha256": expected,
            "formal_source_verified": True,
            **acquisition,
        }
    return result


def validate_layer_densities(
    values: Sequence[float], expected_layers: int
) -> tuple[float, ...]:
    """Validate and normalize the collector output for one prompt."""
    if expected_layers <= 0:
        raise ValueError("expected_layers must be positive")
    if len(values) != expected_layers:
        raise RuntimeError(
            "density layer count mismatch: "
            f"expected {expected_layers}, observed {len(values)}"
        )
    normalized = tuple(float(value) for value in values)
    for layer_index, value in enumerate(normalized):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise RuntimeError(f"invalid density at layer {layer_index}: {value!r}")
    return normalized


def density_statistics(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty density sequence")
    normalized = [float(value) for value in values]
    mean = statistics.fmean(normalized)
    return {
        "count": len(normalized),
        "mean_density": mean,
        "mean_pruning_fraction": 1.0 - mean,
        "population_stdev_density": statistics.pstdev(normalized),
        "min_density": min(normalized),
        "max_density": max(normalized),
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    samples_per_task: int,
    expected_layers: int,
    run_identity_sha256: str,
    require_complete: bool = True,
    method_display_name: str = METHOD_DISPLAY_NAME,
    density_definition: str = DENSITY_DEFINITIONS["bts"],
    target_applies: bool = True,
) -> dict[str, Any]:
    """Build task and per-layer summaries from validated sample records."""
    by_task: dict[str, list[Mapping[str, Any]]] = {task: [] for task in tasks}
    for record in records:
        task = str(record["task"])
        if task not in by_task:
            raise RuntimeError(f"record belongs to an unexpected task: {task}")
        by_task[task].append(record)

    task_summaries = []
    complete = True
    for task in tasks:
        rows = sorted(by_task[task], key=lambda row: int(row["sample_index"]))
        indices = [int(row["sample_index"]) for row in rows]
        expected_indices = list(protocol_sample_indices(samples_per_task))
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"{task} contains duplicate sample indices")
        task_complete = indices == expected_indices
        complete &= task_complete
        if require_complete and not task_complete:
            raise RuntimeError(
                f"{task} is incomplete: observed indices {indices}, "
                f"expected {expected_indices}"
            )

        sample_means = [float(row["mean_density"]) for row in rows]
        layer_summaries = []
        if rows:
            for row in rows:
                validate_layer_densities(row["layer_densities"], expected_layers)
            for layer_index in range(expected_layers):
                layer_values = [
                    float(row["layer_densities"][layer_index]) for row in rows
                ]
                layer_summaries.append(
                    {
                        "layer_index": layer_index,
                        **density_statistics(layer_values),
                    }
                )

        stats = density_statistics(sample_means) if sample_means else None
        class_mean = stats["mean_density"] if stats is not None else None
        task_summaries.append(
            {
                "task": task,
                "category": TASK_LABELS.get(task, task),
                "complete": task_complete,
                "required_samples": samples_per_task,
                "sample_statistics": stats,
                "class_average_density_at_most_0_10": (
                    class_mean <= TARGET_DENSITY
                    if class_mean is not None and task_complete and target_applies
                    else None
                ),
                "layer_statistics": layer_summaries,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "method_display_name": method_display_name,
        "status": "complete" if complete else "in_progress",
        "run_identity_sha256": run_identity_sha256,
        "density_definition": density_definition,
        "target": {
            "maximum_class_mean_density": TARGET_DENSITY,
            "minimum_class_mean_pruning_fraction": 1.0 - TARGET_DENSITY,
            "applies_to_method": target_applies,
        },
        "completed_samples": len(records),
        "required_samples": len(tasks) * samples_per_task,
        # Classes are independent acceptance units.  Deliberately do not emit a
        # cross-class grand average, matching the preregistered density protocol.
        "task_results": task_summaries,
    }


def _density_percent(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.6f}%"


def render_terminal_summary(summary: Mapping[str, Any], output: Path) -> str:
    """Render a compact five-class table from this run's measured summary."""
    columns = (
        ("Category", 34),
        ("Task", 21),
        ("Samples", 9),
        ("Mean Density", 14),
        ("Mean Pruning", 14),
        ("Density Std", 14),
        ("Pruning Range", 25),
        ("<=10%", 7),
    )

    def table_row(values: Sequence[Any]) -> str:
        cells = []
        for value, (_header, width) in zip(values, columns):
            text = str(value)
            if len(text) > width:
                text = text[: width - 1] + "~"
            cells.append(f"{text:<{width}}")
        return " | ".join(cells)

    divider = "-+-".join("-" * width for _header, width in columns)
    lines = [
        "",
        "Density summary (measurements from this run; no cross-class average)",
        divider,
        table_row(header for header, _width in columns),
        divider,
    ]
    passed = 0
    failed = 0
    for result in summary.get("task_results") or []:
        stats = result.get("sample_statistics") or {}
        count = int(stats.get("count", 0))
        required = int(result.get("required_samples", 0))
        class_pass = result.get("class_average_density_at_most_0_10")
        if class_pass is True:
            target = "PASS"
            passed += 1
        elif class_pass is False:
            target = "FAIL"
            failed += 1
        else:
            target = "-"
        minimum_density = stats.get("min_density")
        maximum_density = stats.get("max_density")
        pruning_range = (
            f"{_density_percent(1.0 - float(maximum_density))}.."
            f"{_density_percent(1.0 - float(minimum_density))}"
            if minimum_density is not None and maximum_density is not None
            else "-"
        )
        lines.append(
            table_row(
                (
                    result.get("category", result.get("task", "unknown")),
                    result.get("task", "unknown"),
                    f"{count}/{required}",
                    _density_percent(stats.get("mean_density")),
                    _density_percent(stats.get("mean_pruning_fraction")),
                    _density_percent(stats.get("population_stdev_density")),
                    pruning_range,
                    target,
                )
            )
        )

    historical = summary.get("historical_reproduction") or {}
    matching = int(historical.get("matching_sample_count", 0))
    historical_required = int(historical.get("required_sample_count", 0))
    historical_status = (
        "PASS"
        if historical.get("reproduces_all_parent_sample_means") is True
        else "FAIL"
    )
    lines.extend(
        (
            divider,
            "Completed samples: "
            f"{summary.get('completed_samples', 0)}/{summary.get('required_samples', 0)}",
            f"Class target (mean density <=10%): {passed} PASS, {failed} FAIL",
            "Parent raw sample means: "
            f"{historical_status} ({matching}/{historical_required} matched)",
            "Formal protocol valid: "
            + ("PASS" if summary.get("formal_protocol_valid") is True else "FAIL"),
            f"Saved density summary to {(output / 'summary.json').resolve()}",
        )
    )
    return "\n".join(lines)


class SampleStore:
    """Atomic per-sample persistence and strict resume validation."""

    def __init__(
        self,
        root: Path,
        *,
        run_identity_sha256: str,
        expected_layers: int,
    ) -> None:
        self.root = root
        self.run_identity_sha256 = run_identity_sha256
        self.expected_layers = expected_layers

    def path_for(self, identity: Mapping[str, Any]) -> Path:
        return (
            self.root
            / str(identity["task"])
            / f"{int(identity['sample_index']):03d}.json"
        )

    def _validated_payload(
        self, payload: Mapping[str, Any], identity: Mapping[str, Any]
    ) -> dict[str, Any]:
        exact = (
            "task",
            "source_task",
            "category",
            "sample_index",
            "index",
            "prompt_sha256",
            "prompt_characters",
            "source_declared_length",
            "source_line_1_based",
            "source_row_sha256",
            "official_output_line_1_based",
            "official_output_row_sha256",
            "token_count_before_exact_length",
            "tokens_appended_from_prompt_prefix",
            "tokens_truncated",
            "benchmark_token_count",
            "length_policy",
            "exact_token_ids_sha256",
            "parent_plan_sha256",
            "official_output_file_sha256",
            "input_identity_sha256",
        )
        failed = [name for name in exact if payload.get(name) != identity.get(name)]
        if payload.get("schema_version") != SCHEMA_VERSION:
            failed.append("schema_version")
        if payload.get("run_identity_sha256") != self.run_identity_sha256:
            failed.append("run_identity_sha256")
        if failed:
            raise RuntimeError(
                f"saved density sample identity changed ({', '.join(failed)}): "
                f"{self.path_for(identity)}"
            )
        values = validate_layer_densities(
            payload.get("layer_densities", ()), self.expected_layers
        )
        observed_mean = float(payload.get("mean_density", float("nan")))
        expected_mean = statistics.fmean(values)
        if not math.isclose(observed_mean, expected_mean, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                f"saved density mean does not match layers: {self.path_for(identity)}"
            )
        if payload.get("generation_performed") is not False:
            raise RuntimeError(
                f"saved Density sample unexpectedly performed generation: "
                f"{self.path_for(identity)}"
            )
        if int(payload.get("collector_count", -1)) != self.expected_layers:
            raise RuntimeError(
                f"saved Density collector count is incomplete: "
                f"{self.path_for(identity)}"
            )
        pruning = float(payload.get("pruning_fraction", float("nan")))
        if not math.isclose(
            pruning,
            1.0 - observed_mean,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"saved Density pruning fraction does not match its mean: "
                f"{self.path_for(identity)}"
            )
        capture = payload.get("dense_capture")
        if not isinstance(capture, list) or [
            row.get("layer_index") for row in capture
        ] != list(range(self.expected_layers)):
            raise RuntimeError(
                f"saved dense capture layer order is invalid: "
                f"{self.path_for(identity)}"
            )
        invalid_capture = [
            row.get("layer_index")
            for row in capture
            if row.get("hidden_shape", ())[:2] != [1, MAX_LENGTH]
            or row.get("hidden_dtype") != "torch.bfloat16"
            or row.get("dense_upstream") is not True
        ]
        if invalid_capture:
            raise RuntimeError(
                "saved dense capture metadata changed at layers "
                f"{invalid_capture}: {self.path_for(identity)}"
            )
        return dict(payload)

    def load(self, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self.path_for(identity)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"invalid saved density sample: {path}") from error
        return self._validated_payload(payload, identity)

    def capture(
        self,
        identity: Mapping[str, Any],
        captured: Mapping[str, Any],
    ) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_identity_sha256": self.run_identity_sha256,
            **identity,
            "layer_densities": list(captured["layer_densities"]),
            "mean_density": float(captured["mean_density"]),
            "pruning_fraction": 1.0 - float(captured["mean_density"]),
            "collector_method": captured.get("collector_method"),
            "collector_count": int(captured["collector_count"]),
            "generation_performed": bool(captured["generation_performed"]),
            "execution_scope": str(captured["execution_scope"]),
            "dense_capture": list(captured["dense_capture"]),
        }
        path = self.path_for(identity)
        if path.exists():
            existing = self.load(identity)
            if existing != payload:
                raise RuntimeError(f"refusing to overwrite density sample: {path}")
            return
        write_json_atomic(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Base Qwen/Qwen3-8B checkpoint used by the parent Density run; "
            "do not pass the RULER suite's YaRN4 prompt-generation model"
        ),
    )
    parser.add_argument(
        "--revision",
        default=REFERENCE_MODEL_REVISION,
        help="Pinned base-model revision used by the parent Density run",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned model/tokenizer to exist in the local HF cache",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tasks",
        default=",".join(task for task, _ in TASK_SPECS),
        help="Comma-separated subset of the five frozen RULER density classes",
    )
    parser.add_argument(
        "--ruler-suite",
        type=Path,
        default=DEFAULT_RULER_SUITE,
        help=(
            "Frozen official-protocol RULER suite containing manifest.json, "
            "tasks/, and data/128k/"
        ),
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=DEFAULT_SAMPLES_PER_TASK,
        help=(
            "Samples per task. The audited default 10 selects frozen source "
            "indices 0,11,...,99; other counts are development-only prefixes."
        ),
    )
    parser.add_argument(
        "--attention",
        choices=ATTENTION_NAMES,
        default="bts",
        help="Attention method whose logical density is measured",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Runtime determinism seed only; exact prompts and sample indices "
            "are frozen and never regenerated from this seed"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parallelize", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    return parser


def _selected_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    if not tasks:
        raise ValueError("--tasks did not contain a task name")
    if len(tasks) != len(set(tasks)):
        raise ValueError("--tasks contains duplicates")
    unknown = [task for task in tasks if task not in TASK_LABELS]
    if unknown:
        raise ValueError("unsupported density tasks: " + ", ".join(unknown))
    return tasks


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _model_identity(
    model_name: str,
    revision: str | None,
    trust_remote_code: bool,
    *,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Verify the base checkpoint used by the parent Density measurement."""
    from transformers import AutoConfig
    from transformers.utils.hub import cached_file

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision:
        load_kwargs["revision"] = revision
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    config = AutoConfig.from_pretrained(model_name, **load_kwargs)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    fields = {
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", ()) or ()),
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", -1)),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
        "head_dim": int(head_dim),
        "max_position_embeddings": int(getattr(config, "max_position_embeddings", -1)),
        # Transformers 5 exposes its normalized default RoPE parameters via
        # ``config.rope_scaling`` even though the pinned serialized config has
        # ``rope_scaling: null``.  The serialized field is the stable parent
        # identity and is also protected by the raw config-file hash below.
        "rope_scaling": config.to_dict().get("rope_scaling"),
    }
    expected = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
    }
    failed = [name for name, value in expected.items() if fields.get(name) != value]
    if fields["rope_scaling"] is not None:
        failed.append("rope_scaling")
    if failed:
        raise RuntimeError(
            "the parent Density protocol requires the base Qwen3-8B config "
            "(40960 positions and no rope_scaling), not the suite YaRN4 model; "
            "mismatched fields: " + ", ".join(dict.fromkeys(failed))
        )
    artifact_kwargs = {
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    artifact_paths: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        resolved = cached_file(model_name, filename, **artifact_kwargs)
        if resolved is None:
            raise FileNotFoundError(
                f"could not resolve pinned base-model artifact {filename}"
            )
        artifact_paths[filename] = str(Path(resolved).resolve())
        artifact_hashes[filename] = file_sha256(Path(resolved))
    expected_hashes = {
        "config.json": REFERENCE_CONFIG_FILE_SHA256,
        "tokenizer.json": REFERENCE_TOKENIZER_JSON_SHA256,
        "tokenizer_config.json": REFERENCE_TOKENIZER_CONFIG_SHA256,
    }
    hash_verified = artifact_hashes == expected_hashes
    if not hash_verified:
        changed = [
            name
            for name, expected_hash in expected_hashes.items()
            if artifact_hashes.get(name) != expected_hash
        ]
        raise RuntimeError(
            "base model/tokenizer artifacts differ from the parent revision: "
            + ", ".join(changed)
        )
    commit = getattr(config, "_commit_hash", None)
    reference_verified = (
        model_name == REFERENCE_MODEL_ID
        and revision == REFERENCE_MODEL_REVISION
        and commit == REFERENCE_MODEL_REVISION
        and hash_verified
    )
    return {
        "requested_model": model_name,
        "requested_revision": revision,
        **fields,
        "config_sha256": artifact_hashes["config.json"],
        "huggingface_commit": commit,
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_hashes,
        "expected_artifact_sha256": expected_hashes,
        "artifact_hashes_verified": hash_verified,
        "reference_base_model_verified": reference_verified,
        "suite_yarn4_model_used_for_capture": False,
    }


def _locked_file(path: Path, value: Mapping[str, Any], label: str) -> None:
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"invalid existing {label}: {path}") from error
        if observed != value:
            raise RuntimeError(
                f"{label} changed since this output directory was created: {path}"
            )
    else:
        write_json_atomic(path, value)


def _official_prompts(
    suite: Mapping[str, Any],
    parent_inputs: Mapping[str, Any],
    *,
    tasks: Sequence[str],
    samples_per_task: int,
) -> list[Any]:
    """Load exact official prompts and match every field to the parent plans."""
    from eval.Density.protocol import load_official_prompt

    requested_indices = protocol_sample_indices(samples_per_task)
    prompts: list[Any] = []
    for task in tasks:
        task_plan = parent_inputs["tasks"][task]
        expected_by_index = {int(row["index"]): row for row in task_plan["inputs"]}
        output_path = Path(suite["official_outputs"][task]["path"])
        for index in requested_indices:
            prompt = load_official_prompt(
                Path(suite["resolved_path"]),
                task,
                index,
                maximum_length=MAX_LENGTH,
                official_output_path=output_path,
            )
            expected = expected_by_index.get(index)
            if expected is None:
                # Non-default sample counts are development-only.  Whole source
                # and official-output files remain hash locked, but the parent
                # pilot plan only preregistered 0,11,...,99.
                prompts.append(prompt)
                continue
            observed = prompt.metadata()
            exact_fields = (
                "task",
                "index",
                "source_line_1_based",
                "source_row_sha256",
                "official_output_line_1_based",
                "official_output_row_sha256",
                "prompt_sha256",
                "prompt_characters",
                "source_declared_length",
            )
            changed = [
                name
                for name in exact_fields
                if observed.get(name) != expected.get(name)
            ]
            if changed:
                raise RuntimeError(
                    f"{task}[{index}] differs from the parent Density plan: "
                    + ", ".join(changed)
                )
            prompts.append(prompt)
    hashes = [prompt.prompt_sha256 for prompt in prompts]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("the selected Density prompts are not globally unique")
    return prompts


def _tokenized_sample_identity(
    tokenizer: Any,
    prompt: Any,
    parent_inputs: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Build exact 128K IDs and verify parent token-count/repeat metadata."""
    from eval.Density.protocol import tokenize_official_prompt

    input_ids, token_metadata = tokenize_official_prompt(
        tokenizer,
        prompt,
        maximum_length=MAX_LENGTH,
    )
    task_plan = parent_inputs["tasks"][prompt.task]
    expected = next(
        (row for row in task_plan["inputs"] if int(row["index"]) == int(prompt.index)),
        None,
    )
    if expected is not None:
        checked = (
            "prompt_sha256",
            "token_count_before_exact_length",
            "tokens_appended_from_prompt_prefix",
            "tokens_truncated",
            "benchmark_token_count",
        )
        changed = [
            name for name in checked if token_metadata.get(name) != expected.get(name)
        ]
        if changed:
            raise RuntimeError(
                f"{prompt.task}[{prompt.index}] token identity differs from "
                "the parent Density plan: " + ", ".join(changed)
            )
    base = {
        "task": prompt.task,
        "source_task": f"official_fixed_{prompt.task}",
        "category": TASK_LABELS[prompt.task],
        "sample_index": int(prompt.index),
        **prompt.metadata(),
        **token_metadata,
        "parent_plan_sha256": task_plan["historical_plan_sha256"],
        "official_output_file_sha256": task_plan["official_output_file_sha256"],
    }
    return input_ids, {
        **base,
        "input_identity_sha256": canonical_sha256(base),
    }


def _persist_aggregate_outputs(
    output: Path,
    store: SampleStore,
    identities: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    samples_per_task: int,
    expected_layers: int,
    run_identity_sha256: str,
    require_complete: bool,
    method_display_name: str = METHOD_DISPLAY_NAME,
    density_definition: str = DENSITY_DEFINITIONS["bts"],
    target_applies: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for identity in identities:
        record = store.load(identity)
        if record is not None:
            records.append(record)
    records.sort(
        key=lambda row: (tasks.index(str(row["task"])), int(row["sample_index"]))
    )
    summary = build_summary(
        records,
        tasks=tasks,
        samples_per_task=samples_per_task,
        expected_layers=expected_layers,
        run_identity_sha256=run_identity_sha256,
        require_complete=require_complete,
        method_display_name=method_display_name,
        density_definition=density_definition,
        target_applies=target_applies,
    )
    write_jsonl_atomic(output / "samples.jsonl", records)
    write_json_atomic(output / "summary.json", summary)
    return records, summary


def compare_historical_sample_results(
    records: Sequence[Mapping[str, Any]],
    parent_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare all new sample means with the exact raw parent artifacts."""
    expected: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task, task_plan in parent_inputs["tasks"].items():
        for row in task_plan["inputs"]:
            expected[(task, int(row["index"]))] = row
    comparisons: list[dict[str, Any]] = []
    for record in records:
        key = (str(record["task"]), int(record["sample_index"]))
        parent = expected.get(key)
        if parent is None:
            comparisons.append(
                {
                    "task": key[0],
                    "sample_index": key[1],
                    "historical_result_available": False,
                    "matches_parent": False,
                }
            )
            continue
        observed = float(record["mean_density"])
        historical = float(parent["historical_mean_density"])
        difference = observed - historical
        comparisons.append(
            {
                "task": key[0],
                "sample_index": key[1],
                "historical_result_available": True,
                "historical_raw_result_sha256": parent["historical_raw_result_sha256"],
                "observed_mean_density": observed,
                "historical_mean_density": historical,
                "absolute_difference": abs(difference),
                "matches_parent": math.isclose(
                    observed,
                    historical,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
            }
        )
    complete_panel = set(expected) == {
        (str(record["task"]), int(record["sample_index"])) for record in records
    }
    return {
        "reference_only_until_measured": False,
        "comparison": "absolute tolerance 1e-15 on each unrounded sample mean",
        "required_sample_count": len(expected),
        "observed_sample_count": len(records),
        "complete_fixed_panel": complete_panel,
        "matching_sample_count": sum(row["matches_parent"] for row in comparisons),
        "reproduces_all_parent_sample_means": (
            complete_panel and all(row["matches_parent"] for row in comparisons)
        ),
        "samples": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_tasks:
        for task, label in TASK_SPECS:
            print(f"{task}\t{label}")
        return 0
    if args.output is None:
        parser.error("--output is required")
    args.output = args.output.expanduser().resolve()
    try:
        tasks = _selected_tasks(args.tasks)
        sample_indices = protocol_sample_indices(args.samples_per_task)
    except ValueError as error:
        parser.error(str(error))
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not args.device.startswith("cuda"):
        parser.error("the formal Density protocol requires a CUDA device")
    if args.parallelize:
        parser.error(
            "the Density protocol is single-GPU only; --parallelize is unsupported"
        )

    args.attention = normalize_name(args.attention)
    if args.attention != "bts":
        parser.error(
            "the audited parent Density result is BTS-only; other methods belong "
            "in their existing evaluation runners"
        )
    effective_method_environment = configure_environment(args.attention)
    attention_contract = method_contract(args.attention)
    density_definition = DENSITY_DEFINITIONS[args.attention]
    parent_inputs = fixed_input_manifest()
    suite = ruler_suite_contract(
        args.ruler_suite,
        tasks,
    )
    historical_reference = historical_density_reference()

    bts_frozen_config = bts_frozen_config_contract()
    observed_thresholds = (
        bts_frozen_config.get("low_freq_threshold"),
        bts_frozen_config.get("high_freq_threshold"),
    )
    if observed_thresholds != (0.93, 0.93):
        raise RuntimeError(
            "the frozen parent Density result requires BTS block thresholds "
            f"0.93/0.93, observed {observed_thresholds}"
        )

    requested_parent_model = (
        args.model == REFERENCE_MODEL_ID and args.revision == REFERENCE_MODEL_REVISION
    )
    fixed_manifest_summary = {
        key: parent_inputs.get(key)
        for key in (
            "path",
            "sha256",
            "expected_sha256",
            "available",
            "validation_errors",
            "verified",
        )
    }
    base_protocol = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "model_revision": args.revision,
        "reference_model": REFERENCE_MODEL_ID,
        "reference_model_revision": REFERENCE_MODEL_REVISION,
        "requested_parent_model": requested_parent_model,
        "tasks": tasks,
        "samples_per_task": args.samples_per_task,
        "sample_indices": list(sample_indices),
        "sequence_length": MAX_LENGTH,
        "input_length_policy": (
            "repeat_prompt_token_ids_then_truncate_to_exact_131072"
        ),
        "batch_size": 1,
        "dtype": "bfloat16",
        "seed": args.seed,
        "runtime_replay_seed": args.seed,
        "seed_affects_prompt_identity": False,
        "seed_role": (
            "runtime determinism only; prompts and selected rows are immutable"
        ),
        "device": args.device,
        "parallelize": args.parallelize,
        "execution_gpu_count": 1,
        "method_display_name": attention_contract["display_name"],
        "attention": args.attention,
        "attention_method": attention_contract,
        "effective_method_environment": effective_method_environment,
        "density_definition": density_definition,
        "bts_density_target_applies": True,
        "bts_frozen_config": bts_frozen_config,
        "prompt_source": "official_output.arguments.gen_args_0.arg_0",
        "chat_template_already_applied_in_official_output": True,
        "generation_performed": False,
        "execution_scope": (
            "unpatched dense base-Qwen full-model upstream capture followed by "
            "BTS attention-only replay across all 36 layers"
        ),
        "capture_storage": (
            "same CUDA device as the parent efficiency_comparison capture"
        ),
        "protocol_name": PROTOCOL_NAME,
        "historical_prompt_replay": (
            parent_inputs.get("verified") is True and suite.get("verified") is True
        ),
        "fixed_input_manifest": fixed_manifest_summary,
        "ruler_suite": suite,
        "historical_density_reference": historical_reference,
        "protocol_scope_note": (
            "The five classes replay indices 0,11,...,99 using the exact logged "
            "official prompts. The suite YaRN4 checkpoint produced those logs "
            "but is not the Density capture model; capture uses pinned base "
            "Qwen/Qwen3-8B exactly as the parent run did."
        ),
        "audited_fixed_ten_protocol": (
            tasks == [task for task, _ in TASK_SPECS]
            and sample_indices == FIXED_SAMPLE_INDICES
            and requested_parent_model
            and parent_inputs.get("verified") is True
            and suite.get("verified") is True
        ),
    }
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["COLLECT_DENSITY"] = "true"

    import gc
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    args.output.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    try:
        selected_gpu = resolve_single_gpu(
            args.device,
            inventory,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    except ValueError as error:
        parser.error(str(error))
    preflight_gpu_check = gpu_compute_check(selected_gpu, label="before_model_load")
    if not preflight_gpu_check["passed"]:
        write_json_atomic(
            args.output / "gpu_preflight_failure.json", preflight_gpu_check
        )
        raise RuntimeError(
            "selected GPU has foreign compute processes before model load: "
            f"{preflight_gpu_check['foreign_processes']}"
        )

    model_identity = _model_identity(
        args.model,
        args.revision,
        args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    expected_layers = int(model_identity["num_hidden_layers"])
    if expected_layers != 36:
        raise RuntimeError(
            f"parent Density protocol requires 36 layers, got {expected_layers}"
        )
    prompts = _official_prompts(
        suite,
        parent_inputs,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
    )
    sources = source_manifest()
    environment = environment_manifest()
    environment_sha = canonical_sha256(environment)

    protocol = {
        **base_protocol,
        "runner": "direct parent-protocol Density replay (no lm-eval generation)",
        "density_collector": "bts.bts.STAT_COLLECTOR",
        "method": attention_contract,
        "runtime_hooks": {
            "model_class_patch_before_dense_capture": False,
            "attention_replay": "direct bts_attention_forward call after capture",
        },
        "model_identity": model_identity,
        "selected_gpu": selected_gpu,
        "exact_prompt_rows_verified": True,
        "formal_protocol_valid_before_measurement": (
            base_protocol["audited_fixed_ten_protocol"]
            and re.fullmatch(r"cuda(?::[0-9]+)?", args.device) is not None
            and bool(selected_gpu["uuid"])
            and model_identity["reference_base_model_verified"] is True
            and suite["verified"] is True
            and preflight_gpu_check["passed"]
        ),
        "source_manifest_sha256": sources["manifest_sha256"],
    }
    protocol_sha = canonical_sha256(protocol)
    run_identity_sha = canonical_sha256(
        {
            "protocol_sha256": protocol_sha,
            "model_config_sha256": model_identity["config_sha256"],
            "source_manifest_sha256": sources["manifest_sha256"],
            "environment_sha256": environment_sha,
            "fixed_input_manifest_sha256": parent_inputs["sha256"],
        }
    )
    runtime_config = {
        **protocol,
        "protocol_sha256": protocol_sha,
        "environment": environment,
        "environment_sha256": environment_sha,
        "run_identity_sha256": run_identity_sha,
    }

    _locked_file(args.output / "source_manifest.json", sources, "source manifest")
    _locked_file(
        args.output / "runtime_config.json", runtime_config, "runtime configuration"
    )
    gpu_audit = GpuAudit(
        args.output / "gpu_checks.json",
        run_identity_sha256=run_identity_sha,
        selected_gpu=selected_gpu,
        preflight=preflight_gpu_check,
    )
    required_samples = len(tasks) * args.samples_per_task
    write_json_atomic(
        args.output / "status.json",
        {
            "state": "loading_dense_base_model",
            "run_identity_sha256": run_identity_sha,
            "selected_gpu": selected_gpu,
            "gpu_audit_path": str((args.output / "gpu_checks.json").resolve()),
            "completed_samples": 0,
            "required_samples": required_samples,
        },
    )

    from eval.Density.protocol import (
        capture_dense_attention_calls,
        load_dense_model_and_tokenizer,
        replay_bts_attention_density,
    )

    _seed_everything(args.seed)
    tokenizer, model = load_dense_model_and_tokenizer(
        model_id=args.model,
        device=args.device,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    gpu_audit.check("after_dense_model_load")

    # Importing the direct BTS forward does not patch the Qwen attention class.
    # This happens only after the dense model has been loaded and validated.
    from eval.efficiency.run import load_efficiency_method

    candidate_forward, collector, candidate_contract = load_efficiency_method("bts")
    if candidate_forward is None or collector is None:
        raise RuntimeError("the frozen BTS attention replay is unavailable")

    store = SampleStore(
        args.output / "sample_records",
        run_identity_sha256=run_identity_sha,
        expected_layers=expected_layers,
    )
    identities: list[dict[str, Any]] = []
    for prompt in prompts:
        input_ids, identity = _tokenized_sample_identity(
            tokenizer,
            prompt,
            parent_inputs,
        )
        identities.append(identity)
        existing = store.load(identity)
        stage = f"{identity['task']}:{int(identity['sample_index']):03d}"
        if existing is not None:
            del input_ids
            state = "resuming_completed"
        else:
            state = "running"
        write_json_atomic(
            args.output / "status.json",
            {
                "state": state,
                "run_identity_sha256": run_identity_sha,
                "selected_gpu": selected_gpu,
                "sample": stage,
                "completed_samples": sum(
                    store.load(item) is not None for item in identities
                ),
                "required_samples": required_samples,
            },
        )
        if existing is None:
            gpu_audit.check(f"before_dense_capture:{stage}")
            calls = ()
            try:
                calls = capture_dense_attention_calls(
                    model,
                    input_ids,
                    device=args.device,
                    expected_layers=expected_layers,
                    capture_device=None,
                )
                capture_metadata = [
                    {
                        "layer_index": call.layer_index,
                        "hidden_shape": list(call.hidden_shape),
                        "hidden_dtype": call.hidden_dtype,
                        "capture_device": call.capture_device,
                        "dense_upstream": True,
                    }
                    for call in calls
                ]
                captured = replay_bts_attention_density(
                    calls,
                    device=args.device,
                    expected_layers=expected_layers,
                    forward=candidate_forward,
                    collector=collector,
                )
                captured["dense_capture"] = capture_metadata
                store.capture(identity, captured)
            except BaseException:
                gpu_audit.check(f"after_replay_error:{stage}")
                raise
            finally:
                del calls, input_ids
                gc.collect()
                torch.cuda.empty_cache()
            gpu_audit.check(f"after_attention_replay:{stage}")

        _persist_aggregate_outputs(
            args.output,
            store,
            identities,
            tasks=tasks,
            samples_per_task=args.samples_per_task,
            expected_layers=expected_layers,
            run_identity_sha256=run_identity_sha,
            require_complete=False,
            method_display_name=attention_contract["display_name"],
            density_definition=density_definition,
            target_applies=True,
        )

    records, summary = _persist_aggregate_outputs(
        args.output,
        store,
        identities,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        expected_layers=expected_layers,
        run_identity_sha256=run_identity_sha,
        require_complete=True,
        method_display_name=attention_contract["display_name"],
        density_definition=density_definition,
        target_applies=True,
    )
    input_manifest = [
        {
            key: record[key]
            for key in (
                "task",
                "source_task",
                "sample_index",
                "source_row_sha256",
                "official_output_row_sha256",
                "prompt_sha256",
                "token_count_before_exact_length",
                "tokens_appended_from_prompt_prefix",
                "tokens_truncated",
                "benchmark_token_count",
                "exact_token_ids_sha256",
                "input_identity_sha256",
            )
        }
        for record in records
    ]
    summary["input_manifest_sha256"] = canonical_sha256(input_manifest)
    summary["fixed_input_manifest"] = fixed_manifest_summary
    summary["ruler_suite"] = suite
    summary["historical_density_reference"] = historical_reference
    summary["historical_reproduction"] = compare_historical_sample_results(
        records,
        parent_inputs,
    )
    summary["candidate_contract"] = candidate_contract
    gpu_audit.check("final")
    summary["gpu_audit"] = gpu_audit.payload
    gpu_checks_passed = all(
        check["passed"]
        for session in gpu_audit.payload["sessions"]
        for check in session["checks"]
    )
    summary["formal_protocol_valid"] = (
        runtime_config["formal_protocol_valid_before_measurement"]
        and summary["historical_reproduction"]["reproduces_all_parent_sample_means"]
        and gpu_checks_passed
    )
    if not summary["historical_reproduction"]["reproduces_all_parent_sample_means"]:
        write_json_atomic(
            args.output / "historical_reproduction_mismatch.json",
            summary["historical_reproduction"],
        )
    write_json_atomic(args.output / "summary.json", summary)
    write_json_atomic(
        args.output / "status.json",
        {
            "state": "complete",
            "run_identity_sha256": run_identity_sha,
            "selected_gpu": selected_gpu,
            "last_gpu_check": "final",
            "formal_protocol_valid": summary["formal_protocol_valid"],
            "historical_parent_sample_means_reproduced": summary[
                "historical_reproduction"
            ]["reproduces_all_parent_sample_means"],
            "completed_samples": len(records),
            "required_samples": required_samples,
            "input_manifest_sha256": summary["input_manifest_sha256"],
        },
    )
    print(render_terminal_summary(summary, args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
