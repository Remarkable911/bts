#!/usr/bin/env python3
"""Compare dense and BTS prefill throughput inside the same FastDeploy engine.

The benchmark is intentionally limited to a prefill-dominated deployment
scope: one active Qwen3 BF16 request, one GPU, TP=DP=1, full (non-chunked)
prefill and one generated token.  The engine capacity is kept at 16 sequences
for parity with the serving configuration. Dense and BTS run in fresh processes because enabling the
BTS plugin replaces a process-local FastDeploy backend class and is not safely
reversible.

Lengths denote total context budgets.  FastDeploy 2.5 requires
``input_tokens + min_tokens < max_model_len``, so one output position and one
engine-boundary position are reserved.  The measured input lengths are 32766,
65534 and 131070 by default.  The main
metric is median input-token throughput; for identical inputs its ratio is also
the inverse median end-to-end latency ratio.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import secrets
import signal
import statistics
import struct
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
SUBMISSION_ROOT = HERE.parents[1]
FASTDEPLOY_SRC = SUBMISSION_ROOT / "integrations" / "fastdeploy" / "src"
PROTOCOL = "fastdeploy_dense_vs_bts_prefill_fa2_v2"
DEFAULT_LENGTHS = (32768, 65536, 131072)
METHODS = ("dense", "bts")
FORMAL_RESTARTS = 5
ENGINE_BOUNDARY_POSITIONS = 1
BENCHMARK_RUN_TOKEN_ENV = "BTS_BENCHMARK_RUN_TOKEN"
FORMAL_WARMUP_RUNS = 1
FORMAL_MEASURED_RUNS = 5
FORMAL_BOOTSTRAP_SAMPLES = 10000
FORMAL_SEED = 20261110
FORMAL_TARGET_SPEEDUP = 1.5
FORMAL_FLASH_ATTN_VERSION = 2
FORMAL_MAX_NUM_SEQS = 16
EXPECTED_MODEL_SHAPE = {
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
EXPECTED_YARN4 = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        if temporary.exists():
            temporary.unlink()


def mark_manifest_failed(output: Path, error: BaseException) -> None:
    path = output / "manifest.json"
    if not path.is_file():
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            return
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["failure_type"] = type(error).__name__
        manifest["failure"] = str(error)
        manifest["failed_at"] = utc_now()
        atomic_write_json(path, manifest)
    except Exception:
        # Never hide the original benchmark error with a reporting failure.
        return


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, Any]:
    """Hash every submitted runtime source that can affect either arm."""
    candidates = {
        Path(__file__).resolve(),
        SUBMISSION_ROOT / "pyproject.toml",
        SUBMISSION_ROOT / "integrations" / "fastdeploy" / "pyproject.toml",
    }
    for directory in (
        SUBMISSION_ROOT / "bts",
        FASTDEPLOY_SRC / "bts_fastdeploy",
        SUBMISSION_ROOT / "integrations" / "fastdeploy" / "configs",
    ):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".json"}:
                candidates.add(path.resolve())
    files = {
        path.relative_to(SUBMISSION_ROOT).as_posix(): file_sha256(path)
        for path in sorted(candidates)
    }
    return {
        "schema_version": "bts_fastdeploy_runtime_sources_v1",
        "sha256": canonical_sha256(files),
        "files": files,
    }


def validate_model_contract(model_info: dict[str, Any], lengths: Sequence[int]) -> dict[str, Any]:
    """Validate the frozen Qwen3 BF16/PaddleFormers long-context contract."""
    errors = []
    if model_info.get("model_type") != "qwen3":
        errors.append("model_type must be qwen3")
    head_dim = model_info.get("head_dim")
    if head_dim is None:
        hidden_size = model_info.get("hidden_size")
        num_heads = model_info.get("num_attention_heads")
        if hidden_size is not None and num_heads and int(hidden_size) % int(num_heads) == 0:
            head_dim = int(hidden_size) // int(num_heads)
    actual_shape = {
        "hidden_size": int(model_info.get("hidden_size", -1)),
        "num_hidden_layers": int(model_info.get("num_hidden_layers", -1)),
        "num_attention_heads": int(model_info.get("num_attention_heads", -1)),
        "num_key_value_heads": int(model_info.get("num_key_value_heads", -1)),
        "head_dim": int(head_dim) if head_dim is not None else -1,
    }
    if actual_shape != EXPECTED_MODEL_SHAPE:
        errors.append(
            "model shape must be the frozen Qwen3-8B architecture "
            f"{EXPECTED_MODEL_SHAPE}; found {actual_shape}"
        )
    dtype = model_info.get("torch_dtype", model_info.get("dtype"))
    if str(dtype).lower() not in ("bfloat16", "bf16"):
        errors.append(f"model dtype must be bfloat16; found {dtype!r}")
    if any(
        model_info.get(field) not in (None, False, "", {})
        for field in ("quantization_config", "quantization", "quantization_method")
    ):
        errors.append("quantized model configs are not supported")
    if model_info.get("model_impl") not in (None, "paddleformers"):
        errors.append("model_impl must be paddleformers when specified")
    max_positions = int(model_info.get("max_position_embeddings", 0))
    if max(lengths) > max_positions:
        errors.append(
            f"requested context {max(lengths)} exceeds max_position_embeddings={max_positions}"
        )
    if max_positions != 131072:
        errors.append(
            f"max_position_embeddings must be 131072; found {max_positions}"
        )
    rope_scaling = model_info.get("rope_scaling")
    actual_yarn = {
        "rope_type": (rope_scaling or {}).get(
            "rope_type", (rope_scaling or {}).get("type")
        ),
        "factor": (rope_scaling or {}).get("factor"),
        "original_max_position_embeddings": (rope_scaling or {}).get(
            "original_max_position_embeddings"
        ),
    }
    if actual_yarn != EXPECTED_YARN4:
        errors.append(
            f"rope_scaling must be the frozen YaRN4 contract {EXPECTED_YARN4}; "
            f"found {actual_yarn}"
        )
    for field in ("vocab_size", "num_hidden_layers"):
        if int(model_info.get(field, 0)) <= 0:
            errors.append(f"{field} must be a positive integer")
    if errors:
        raise ValueError("invalid FastDeploy model contract: " + "; ".join(errors))
    return {
        "model_type": "qwen3",
        **actual_shape,
        "dtype": "bfloat16",
        "model_impl": "paddleformers",
        "max_position_embeddings": max_positions,
        "rope_scaling": rope_scaling,
        "vocab_size": int(model_info["vocab_size"]),
        "quantized": False,
    }


def parse_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--lengths must contain comma-separated integers") from error
    if not lengths:
        raise argparse.ArgumentTypeError("--lengths must contain at least one length")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("--lengths must not contain duplicates")
    if any(length <= 1 for length in lengths):
        raise argparse.ArgumentTypeError("every context length must be greater than one")
    return lengths


def deterministic_token_ids(length: int, vocab_size: int, seed: int) -> list[int]:
    """Return a stable synthetic prefix without tokenizer or special tokens."""
    if length <= 0:
        raise ValueError("prompt length must be positive")
    # Qwen special/control tokens occupy the high end of the vocabulary.  Keep
    # a conservative margin at both ends and use a specified 32-bit LCG so the
    # same IDs are generated across Python versions and processes.
    low = 256
    high = vocab_size - 1024
    if high <= low:
        raise ValueError("model vocabulary is too small for the benchmark generator")
    span = high - low
    state = seed & 0xFFFFFFFF
    result = []
    for _ in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        result.append(low + state % span)
    return result


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<I", int(token_id)))
    return digest.hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def summarize_samples(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples or any(value <= 0 or not math.isfinite(value) for value in samples):
        raise ValueError("timing samples must be finite and positive")
    return {
        "count": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "p95": percentile(samples, 0.95),
        "max": max(samples),
        "samples": samples,
    }


def paired_bootstrap_median_lower_bounds(
    speedups_by_length: dict[int, Sequence[float]],
    *,
    confidence: float = 0.95,
    samples: int = 10000,
    seed: int = 20261110,
) -> dict[int, float]:
    """Resample complete restart pairs, sharing indices across all lengths."""
    if not speedups_by_length:
        raise ValueError("paired bootstrap requires at least one length")
    counts = {len(values) for values in speedups_by_length.values()}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("paired bootstrap lengths must have equal sample counts >= 2")
    restart_count = next(iter(counts))
    rng = random.Random(seed)
    draws = {length: [] for length in speedups_by_length}
    for _ in range(samples):
        indices = [rng.randrange(restart_count) for _ in range(restart_count)]
        for length, values in speedups_by_length.items():
            draws[length].append(statistics.median(values[index] for index in indices))
    # Explicit lower empirical quantile; this is the analysis registered in the
    # JSON report and avoids library-version-dependent interpolation.
    index = max(0, math.floor((1.0 - confidence) * (samples - 1)))
    return {length: sorted(values)[index] for length, values in draws.items()}


def build_comparison(
    restart_reports: Sequence[dict[str, dict[str, Any]]],
    *,
    target_speedup: float,
    bootstrap_samples: int = FORMAL_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if not restart_reports:
        raise ValueError("comparison requires at least one paired restart")
    by_length: dict[int, list[dict[str, Any]]] = {}
    expected_lengths: set[int] | None = None
    for restart_index, reports in enumerate(restart_reports):
        if set(reports) != set(METHODS):
            raise ValueError("every restart requires both dense and BTS reports")
        dense_rows = {row["context_length"]: row for row in reports["dense"]["lengths"]}
        bts_rows = {row["context_length"]: row for row in reports["bts"]["lengths"]}
        if set(dense_rows) != set(bts_rows):
            raise ValueError("dense and BTS context lengths differ")
        if expected_lengths is None:
            expected_lengths = set(dense_rows)
        elif set(dense_rows) != expected_lengths:
            raise ValueError("context lengths differ across restarts")
        for context_length in sorted(dense_rows):
            dense = dense_rows[context_length]
            bts = bts_rows[context_length]
            for field in ("prompt_tokens", "prompt_sha256", "engine_contract_sha256"):
                if dense[field] != bts[field]:
                    raise ValueError(
                        f"dense and BTS {field} differ at {context_length}"
                    )
            by_length.setdefault(context_length, []).append(
                {
                    "restart": restart_index,
                    "dense_median_latency_seconds": dense["latency_seconds"]["median"],
                    "bts_median_latency_seconds": bts["latency_seconds"]["median"],
                    "dense_request_repetitions": dense["latency_seconds"]["count"],
                    "bts_request_repetitions": bts["latency_seconds"]["count"],
                    "throughput_speedup": (
                        dense["latency_seconds"]["median"]
                        / bts["latency_seconds"]["median"]
                    ),
                    "prompt_tokens": dense["prompt_tokens"],
                    "prompt_sha256": dense["prompt_sha256"],
                    "engine_contract_sha256": dense["engine_contract_sha256"],
                    "dense_median_input_tokens_per_second": dense[
                        "input_tokens_per_second"
                    ]["median"],
                    "bts_median_input_tokens_per_second": bts[
                        "input_tokens_per_second"
                    ]["median"],
                }
            )

    for context_length, pairs in by_length.items():
        if len(pairs) != len(restart_reports):
            raise ValueError(f"incomplete restart pairs at {context_length}")
        for field in ("prompt_tokens", "prompt_sha256", "engine_contract_sha256"):
            if len({pair[field] for pair in pairs}) != 1:
                raise ValueError(f"restart {field} mismatch at {context_length}")

    eligibility_failures = []
    if len(restart_reports) != FORMAL_RESTARTS:
        eligibility_failures.append(
            f"formal_restart_pairs_must_be_{FORMAL_RESTARTS}"
        )
    if set(by_length) != set(DEFAULT_LENGTHS):
        eligibility_failures.append(
            "formal_lengths_must_be_32768_65536_131072"
        )
    if not math.isclose(
        target_speedup, FORMAL_TARGET_SPEEDUP, rel_tol=0.0, abs_tol=1e-12
    ):
        eligibility_failures.append("formal_target_speedup_must_be_1.5")
    reports_flat = [
        report
        for restart in restart_reports
        for report in restart.values()
    ]
    if bootstrap_samples != FORMAL_BOOTSTRAP_SAMPLES:
        eligibility_failures.append(
            f"formal_bootstrap_samples_must_be_{FORMAL_BOOTSTRAP_SAMPLES}"
        )
    if any(report.get("warmup_runs") != FORMAL_WARMUP_RUNS for report in reports_flat):
        eligibility_failures.append(
            f"formal_warmup_runs_must_be_{FORMAL_WARMUP_RUNS}"
        )
    if any(report.get("measured_runs") != FORMAL_MEASURED_RUNS for report in reports_flat):
        eligibility_failures.append(
            f"formal_measured_runs_must_be_{FORMAL_MEASURED_RUNS}"
        )
    if any(report.get("seed") != FORMAL_SEED for report in reports_flat):
        eligibility_failures.append(f"formal_seed_must_be_{FORMAL_SEED}")
    if any(
        report.get("engine_contract", {}).get("max_new_tokens") != 1
        for report in reports_flat
    ):
        eligibility_failures.append("formal_max_new_tokens_must_be_1")
    if any(
        report.get("engine_contract", {}).get("paddle_flash_attn_version")
        != FORMAL_FLASH_ATTN_VERSION
        or report.get("engine_contract", {}).get("fastdeploy_flash_attn_version")
        != FORMAL_FLASH_ATTN_VERSION
        for report in reports_flat
    ):
        eligibility_failures.append("formal_dense_flash_attention_version_must_be_2")
    if any(
        report.get("runtime_native_attention_contract")
        != {
            "fastdeploy_flash_attn_version": FORMAL_FLASH_ATTN_VERSION,
            "paddle_flash_attn_version": FORMAL_FLASH_ATTN_VERSION,
            "dense_prefill_op": "flash_attn_unpadded",
            "decode_op": "append_attention",
        }
        for report in reports_flat
    ):
        eligibility_failures.append("formal_runtime_native_attention_contract_must_be_fa2")
    source_hashes = {
        report.get("source_manifest_sha256") for report in reports_flat
    }
    if None in source_hashes or len(source_hashes) != 1:
        eligibility_failures.append("formal_runtime_source_hash_must_match")
    config_hashes = {
        report.get("bts_config_sha256") for report in reports_flat
    }
    if None in config_hashes or len(config_hashes) != 1:
        eligibility_failures.append("formal_bts_config_hash_must_match")
    dependency_hashes = {
        report.get("dependency_contract_sha256") for report in reports_flat
    }
    if None in dependency_hashes or len(dependency_hashes) != 1:
        eligibility_failures.append(
            "formal_dependency_source_contract_hash_must_match"
        )
    if any(
        pair["dense_request_repetitions"] != FORMAL_MEASURED_RUNS
        or pair["bts_request_repetitions"] != FORMAL_MEASURED_RUNS
        for pairs in by_length.values()
        for pair in pairs
    ):
        eligibility_failures.append(
            f"formal_protocol_requires_{FORMAL_MEASURED_RUNS}_measured_requests_per_arm"
        )
    protocol_valid = not eligibility_failures
    speedups_by_length = {
        length: [pair["throughput_speedup"] for pair in pairs]
        for length, pairs in by_length.items()
    }
    lower_bounds = (
        paired_bootstrap_median_lower_bounds(
            speedups_by_length,
            samples=bootstrap_samples,
            seed=FORMAL_SEED,
        )
        if protocol_valid
        else {}
    )
    comparisons = []
    for context_length in sorted(by_length):
        pairs = by_length[context_length]
        speedups = [pair["throughput_speedup"] for pair in pairs]
        point_speedup = statistics.median(speedups)
        lower = lower_bounds.get(context_length)
        target_passed = (
            lower >= target_speedup if protocol_valid and lower is not None else None
        )
        comparisons.append(
            {
                "context_length": context_length,
                "prompt_tokens": pairs[0]["prompt_tokens"],
                "paired_restarts": pairs,
                "restart_speedups": speedups,
                "throughput_speedup": point_speedup,
                "bootstrap_one_sided_95pct_lower_bound": lower,
                "target_speedup": target_speedup,
                "point_estimate_meets_target": point_speedup >= target_speedup,
                "target_passed": target_passed,
                "target_basis": (
                    "paired_restart_bootstrap_lower_bound"
                    if protocol_valid else "protocol_ineligible"
                ),
            }
        )
    speedups = [row["throughput_speedup"] for row in comparisons]
    geometric_mean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    passed = (
        all(row["target_passed"] for row in comparisons)
        if protocol_valid else None
    )
    verdict = "PASS" if passed is True else ("FAIL" if protocol_valid else "INCONCLUSIVE")
    return {
        "lengths": comparisons,
        "statistics": {
            "schema_version": "paired_restarts_v1",
            "analysis_unit": "paired_fresh_process_restart",
            "restart_count": len(restart_reports),
            "formal_restart_count": FORMAL_RESTARTS,
            "statistic": "median_of_paired_restart_latency_ratios",
            "bootstrap": {
                "type": "paired_cluster_percentile",
                "shared_restart_indices_across_lengths": True,
                "confidence": 0.95,
                "replicates": bootstrap_samples,
                "seed": FORMAL_SEED,
                "quantile_method": "lower_floor_q_times_n_minus_1",
            },
        },
        "protocol_valid": protocol_valid,
        "protocol_eligibility_failures": eligibility_failures,
        "geometric_mean_throughput_speedup": geometric_mean,
        "all_lengths_passed": passed,
        "decision_reason": (
            "all_length_lower_bounds_meet_target"
            if passed is True
            else (
                "one_or_more_length_lower_bounds_below_target"
                if protocol_valid
                else ";".join(eligibility_failures)
            )
        ),
        "verdict": verdict,
    }


def engine_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "model_impl": "paddleformers",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "max_model_len": max(args.lengths),
        "max_num_seqs": FORMAL_MAX_NUM_SEQS,
        "max_num_batched_tokens": max(args.lengths),
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "use_cudagraph": False,
        "graph_opt_level": 0,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_custom_all_reduce": True,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.max_new_tokens,
        "engine_boundary_positions": ENGINE_BOUNDARY_POSITIONS,
        "paddle_flash_attn_version": args.paddle_flash_attn_version,
        "fastdeploy_flash_attn_version": FORMAL_FLASH_ATTN_VERSION,
        "dense_attention": "native_fastdeploy_flash_attention_2",
        "dense_prefill_op": "flash_attn_unpadded",
        "decode_op": "append_attention",
        "paddleformers_rope_compatibility_guard": True,
        "use_tqdm": True,
        "use_tqdm_reason": "FastDeploy 2.5 _run_engine requires num_requests initialized by tqdm path",
    }


def _completion_token_ids(outputs: Sequence[Any]) -> list[int]:
    token_ids = []
    for request_output in outputs:
        completion = request_output.outputs
        if completion is None:
            continue
        completions: Iterable[Any] = completion if isinstance(completion, (list, tuple)) else (completion,)
        for item in completions:
            token_ids.extend([] if item.token_ids is None else list(item.token_ids))
    return token_ids


def _package_versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "fastdeploy-gpu",
        "paddlepaddle-gpu",
        "paddleformers",
        "torch",
        "triton",
        "transformers",
        "bts",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _gpu_snapshot() -> list[dict[str, str]]:
    fields = ("index", "uuid", "name", "driver_version", "memory.total")
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return [{"error": str(error)}]
    return [
        dict(zip(fields, (item.strip() for item in line.split(","))))
        for line in raw.splitlines()
        if line.strip()
    ]


def selected_gpu_snapshot() -> dict[str, Any]:
    """Resolve the single visible physical GPU and list its compute processes."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU index or UUID"
        )
    fields = (
        "index",
        "uuid",
        "name",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "pstate",
        "clocks.sm",
        "compute_cap",
    )
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    gpus = [
        dict(zip(fields, (item.strip() for item in line.split(","))))
        for line in raw.splitlines()
        if line.strip()
    ]
    selected = [
        gpu
        for gpu in gpus
        if visible == gpu["index"]
        or visible == gpu["uuid"]
        or gpu["uuid"].startswith(visible)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"cannot resolve CUDA_VISIBLE_DEVICES={visible!r} uniquely")
    process_raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    processes = []
    for line in process_raw.splitlines():
        if not line.strip():
            continue
        items = [item.strip() for item in line.split(",", 3)]
        if len(items) == 4 and items[0] == selected[0]["uuid"]:
            processes.append(
                dict(zip(("gpu_uuid", "pid", "process_name", "used_memory"), items))
            )
    return {
        "cuda_visible_devices": visible,
        "physical_gpu": selected[0],
        "compute_processes": processes,
        "captured_at": utc_now(),
    }


def wait_for_selected_gpu_idle(timeout_seconds: float, poll_seconds: float = 5.0) -> dict[str, Any]:
    """Wait for driver access and exclusivity without terminating another job."""
    deadline = time.monotonic() + timeout_seconds
    announced_busy = False
    announced_unavailable = False
    while True:
        try:
            snapshot = selected_gpu_snapshot()
        except subprocess.CalledProcessError as error:
            if time.monotonic() >= deadline:
                detail = (error.output or "").strip()
                message = "NVIDIA GPU status is unavailable"
                if detail:
                    message += f": {detail}"
                raise RuntimeError(message) from error
            if not announced_unavailable:
                print(
                    "NVIDIA GPU status is unavailable; waiting for driver access to recover",
                    flush=True,
                )
                announced_unavailable = True
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
            continue
        if not snapshot["compute_processes"]:
            return snapshot
        if time.monotonic() >= deadline:
            pids = [process["pid"] for process in snapshot["compute_processes"]]
            raise RuntimeError(
                "selected GPU is not exclusive; existing compute PIDs="
                + ",".join(pids)
            )
        if not announced_busy:
            print(
                "selected GPU is busy; waiting without interfering with existing processes",
                flush=True,
            )
            announced_busy = True
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _descendant_pids(root_pid: int) -> list[int]:
    """Return descendants of a process using a read-only process-table snapshot."""
    raw = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid="], text=True, stderr=subprocess.DEVNULL
    )
    children: dict[int, list[int]] = {}
    for line in raw.splitlines():
        try:
            pid, parent = (int(value) for value in line.split())
        except (ValueError, TypeError):
            continue
        children.setdefault(parent, []).append(pid)
    result = []
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        result.append(pid)
        stack.extend(children.get(pid, ()))
    return result


def _pids_with_environment_token(token: str) -> list[int]:
    """Find same-user processes carrying the per-arm benchmark token."""
    expected = f"{BENCHMARK_RUN_TOKEN_ENV}={token}".encode("utf-8")
    result = []
    for environ_path in Path("/proc").glob("[0-9]*/environ"):
        try:
            values = environ_path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if expected in values:
            result.append(int(environ_path.parent.name))
    return result


def _terminate_owned_process_tree(
    process: subprocess.Popen[Any], token: str
) -> list[int]:
    """TERM then KILL only this arm, including nested ``setsid`` workers."""
    targets = set(_pids_with_environment_token(token))
    if process.poll() is None:
        targets.update(_descendant_pids(process.pid))
        targets.add(process.pid)
    terminated = set(targets)
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        newly_found = set(_pids_with_environment_token(token)) - targets
        for pid in newly_found:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        targets.update(newly_found)
        terminated.update(newly_found)
        alive = []
        for pid in targets:
            if pid == process.pid and process.poll() is not None:
                continue
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            return sorted(terminated)
        time.sleep(0.2)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass
    return sorted(terminated)


class _WorkerControlSignal(RuntimeError):
    pass


def run_worker_subprocess(
    command: Sequence[str],
    *,
    env: dict[str, str],
    log: Any,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    token = secrets.token_hex(16)
    process_env = dict(env)
    process_env[BENCHMARK_RUN_TOKEN_ENV] = token
    original_signal_handlers: dict[signal.Signals, Any] = {}

    def raise_control_signal(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        raise _WorkerControlSignal(f"benchmark controller received {name}")

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        original_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, raise_control_signal)

    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            command,
            env=process_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        next_gpu_check = 0.0
        monitor_started_at = utc_now()
        check_count = 0
        outsider_rechecks = 0
        owned_gpu_pids: set[int] = set()
        monitored_gpu: dict[str, str] | None = None
        last_checked_at: str | None = None

        def check_gpu_exclusivity() -> None:
            nonlocal check_count, outsider_rechecks, monitored_gpu, last_checked_at
            snapshot = selected_gpu_snapshot()
            check_count += 1
            last_checked_at = snapshot["captured_at"]
            if monitored_gpu is None:
                monitored_gpu = snapshot["physical_gpu"]
            owned = {
                process.pid,
                *_descendant_pids(process.pid),
                *_pids_with_environment_token(token),
            }
            outsiders = [
                item
                for item in snapshot["compute_processes"]
                if int(item["pid"]) not in owned
            ]
            if outsiders:
                outsider_rechecks += 1
                time.sleep(0.25)
                owned = {
                    process.pid,
                    *_descendant_pids(process.pid),
                    *_pids_with_environment_token(token),
                }
                snapshot = selected_gpu_snapshot()
                check_count += 1
                last_checked_at = snapshot["captured_at"]
                outsiders = [
                    item
                    for item in snapshot["compute_processes"]
                    if int(item["pid"]) not in owned
                ]
            if outsiders:
                pids = ",".join(item["pid"] for item in outsiders)
                raise RuntimeError(
                    "selected GPU lost exclusivity during the timed arm; "
                    f"external compute PIDs={pids}"
                )
            owned_gpu_pids.update(
                int(item["pid"])
                for item in snapshot["compute_processes"]
                if int(item["pid"]) in owned
            )

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            if time.monotonic() >= next_gpu_check:
                check_gpu_exclusivity()
                next_gpu_check = time.monotonic() + 1.0
            try:
                returncode = process.wait(timeout=min(0.25, remaining))
                # Close the final sampling window after the launcher exits but
                # while any token-marked nested GPU worker is still identifiable.
                check_gpu_exclusivity()
                lingering = set(_pids_with_environment_token(token))
                lingering.discard(process.pid)
                cleanup_pids = (
                    _terminate_owned_process_tree(process, token)
                    if returncode or lingering
                    else []
                )
                return returncode, {
                    "enabled": True,
                    "mode": "sampled_compute_process_monitor",
                    "poll_interval_seconds": 1.0,
                    "started_at": monitor_started_at,
                    "last_checked_at": last_checked_at,
                    "check_count": check_count,
                    "all_sampled_checks_exclusive": True,
                    "final_check_completed": True,
                    "outsider_rechecks": outsider_rechecks,
                    "benchmark_gpu_pids_observed": sorted(owned_gpu_pids),
                    "cleanup_pids": cleanup_pids,
                    "physical_gpu": monitored_gpu,
                }
            except subprocess.TimeoutExpired:
                pass
    except BaseException:
        if process is not None:
            _terminate_owned_process_tree(process, token)
        raise
    finally:
        for signum, handler in original_signal_handlers.items():
            signal.signal(signum, handler)


def wait_for_worker_audits(
    audit_dir: Path,
    *,
    timeout_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Read the live mmap audit after the FastDeploy GPU worker is ready."""
    from bts_fastdeploy.audit import read_audits

    deadline = time.monotonic() + timeout_seconds
    while True:
        audits = read_audits(audit_dir)
        if audits:
            return audits
        if time.monotonic() >= deadline:
            return []
        time.sleep(0.1)


def aggregate_audit_layers(
    audits: Sequence[dict[str, Any]], field: str
) -> dict[int, int]:
    result: dict[int, int] = {}
    for audit in audits:
        for raw_layer, count in audit.get(field, {}).items():
            layer = int(raw_layer)
            result[layer] = result.get(layer, 0) + int(count)
    return result


def audit_layer_deltas(
    baseline: Sequence[dict[str, Any]],
    final: Sequence[dict[str, Any]],
    field: str,
) -> dict[int, int]:
    baseline_pids = {int(audit["pid"]) for audit in baseline}
    final_pids = {int(audit["pid"]) for audit in final}
    if baseline_pids != final_pids:
        raise RuntimeError(
            "FastDeploy audit worker set changed during measurement: "
            f"baseline={sorted(baseline_pids)}, final={sorted(final_pids)}"
        )
    before = aggregate_audit_layers(baseline, field)
    after = aggregate_audit_layers(final, field)
    layers = set(before) | set(after)
    deltas = {layer: after.get(layer, 0) - before.get(layer, 0) for layer in layers}
    negative = {layer: count for layer, count in deltas.items() if count < 0}
    if negative:
        raise RuntimeError(f"FastDeploy audit counters decreased: {negative}")
    return deltas


def worker_main(args: argparse.Namespace) -> int:
    method = args.worker_method
    if method not in METHODS:
        raise ValueError(f"unknown worker method: {method}")
    # Fail closed against inherited research/plugin state, then install exactly
    # one submitted backend mode in this fresh process and its GPU worker.
    for name in tuple(os.environ):
        if name.startswith("BTS_FASTDEPLOY_") or name.startswith("PRISM_FASTDEPLOY_"):
            os.environ.pop(name, None)
    os.environ["BTS_FASTDEPLOY_ENABLE"] = "1"
    os.environ["BTS_FASTDEPLOY_MODE"] = (
        "bts" if method == "bts" else "dense_rope_compat"
    )
    if method == "bts":
        os.environ["BTS_FASTDEPLOY_CONFIG"] = str(args.config)
    model_info = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    model_contract = validate_model_contract(model_info, args.lengths)
    audit_key = hashlib.sha256(str(args.worker_output).encode("utf-8")).hexdigest()[:16]
    audit_dir = Path("/dev/shm") / f"bts-fastdeploy-audit-{os.getpid()}-{audit_key}"
    os.environ["BTS_FASTDEPLOY_AUDIT_DIR"] = str(audit_dir.resolve())
    os.environ["BTS_FASTDEPLOY_AUDIT_FLUSH"] = "mmap"
    os.environ["BTS_FASTDEPLOY_REQUIRE_DEPENDENCY_CONTRACT"] = "1"
    os.environ["BTS_FASTDEPLOY_AUDIT_NUM_LAYERS"] = str(
        int(model_info["num_hidden_layers"])
    )
    os.environ["FD_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["FD_DISABLE_CHUNKED_PREFILL"] = "1"
    os.environ["FD_PLUGINS"] = "bts"
    # The plugin also sets FastDeploy's independent module selector.  Set the
    # Paddle flag before importing either framework, then verify both below.
    os.environ["FLAGS_flash_attn_version"] = str(args.paddle_flash_attn_version)
    os.environ["FD_LOG_DIR"] = str((args.worker_output.parent / f"{method}-fastdeploy-logs").resolve())
    Path(os.environ["FD_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    # This prepares library paths and propagates the source checkout to every
    # FastDeploy worker before importing Paddle/FastDeploy.
    sys.path.insert(0, str(SUBMISSION_ROOT))
    sys.path.insert(0, str(FASTDEPLOY_SRC))
    from bts_fastdeploy.config import config_digest, load_config
    from bts_fastdeploy.environment import package_provenance, prepare_workers
    from bts_fastdeploy.upstream_contract import (
        dependency_source_contract,
        validate_runtime_dependency_imports,
    )

    provenance = prepare_workers()
    canonical_bts_config = load_config(args.config)
    expected_config_digest = config_digest(canonical_bts_config)
    runtime_sources = source_manifest()
    dependency_contract = dependency_source_contract()
    if (
        args.expected_dependency_contract_sha256 is not None
        and dependency_contract["sha256"]
        != args.expected_dependency_contract_sha256
    ):
        raise RuntimeError(
            "worker dependency-source contract differs from controller: "
            f"expected={args.expected_dependency_contract_sha256}, "
            f"actual={dependency_contract['sha256']}"
        )
    from fastdeploy import LLM, SamplingParams
    from bts_fastdeploy.plugin import register
    from fastdeploy.model_executor.layers.attention import flash_attn_backend

    register()
    runtime_attention_contract = getattr(
        flash_attn_backend.FlashAttentionBackend,
        "_bts_native_attention_contract",
        None,
    )
    if runtime_attention_contract != {
        "fastdeploy_flash_attn_version": FORMAL_FLASH_ATTN_VERSION,
        "paddle_flash_attn_version": FORMAL_FLASH_ATTN_VERSION,
        "dense_prefill_op": "flash_attn_unpadded",
        "decode_op": "append_attention",
    }:
        raise RuntimeError(
            "FastDeploy launcher did not enforce the frozen FA2/native-decode "
            f"contract: {runtime_attention_contract!r}"
        )
    post_register_dependency_contract = dependency_source_contract()
    if post_register_dependency_contract != dependency_contract:
        raise RuntimeError("approved dependency sources changed during worker startup")
    launcher_runtime_dependency_imports = validate_runtime_dependency_imports(
        dependency_contract
    )
    adapter_active = bool(
        getattr(flash_attn_backend.FlashAttentionBackend, "_bts_fastdeploy_adapter", False)
    )
    dense_compat_active = bool(
        getattr(
            flash_attn_backend.FlashAttentionBackend,
            "_bts_fastdeploy_dense_compat",
            False,
        )
    )
    if adapter_active != (method == "bts"):
        raise RuntimeError(
            f"backend activation mismatch: method={method}, adapter_active={adapter_active}"
        )
    if dense_compat_active != (method == "dense"):
        raise RuntimeError(
            "dense compatibility activation mismatch: "
            f"method={method}, dense_compat_active={dense_compat_active}"
        )
    expected_mode = "bts" if method == "bts" else "dense_rope_compat"
    actual_mode = getattr(
        flash_attn_backend.FlashAttentionBackend, "_bts_fastdeploy_mode", None
    )
    if actual_mode != expected_mode:
        raise RuntimeError(
            f"backend mode mismatch: expected={expected_mode}, actual={actual_mode}"
        )
    native_dense_preserved = bool(
        getattr(
            flash_attn_backend.FlashAttentionBackend,
            "_bts_native_flash_attn_preserved",
            False,
        )
    )
    if native_dense_preserved != (method == "dense"):
        raise RuntimeError(
            "native dense attention preservation marker mismatch: "
            f"method={method}, preserved={native_dense_preserved}"
        )

    contract = engine_contract(args)
    contract_hash = canonical_sha256(contract)
    engine_options = {
        "model": str(args.model),
        "model_impl": "paddleformers",
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "max_model_len": max(args.lengths),
        "max_num_seqs": FORMAL_MAX_NUM_SEQS,
        "max_num_batched_tokens": max(args.lengths),
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "graph_optimization_config": {"use_cudagraph": False, "graph_opt_level": 0},
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_custom_all_reduce": True,
    }
    vocab_size = int(model_info["vocab_size"])
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "running",
        "method": method,
        "started_at": utc_now(),
        "scope": "offline_sequential_prefill_with_max_num_seqs_16",
        "package_versions": _package_versions(),
        "package_provenance": provenance,
        "model_contract": model_contract,
        "gpu": _gpu_snapshot(),
        "adapter_active_in_launcher": adapter_active,
        "dense_rope_compat_active_in_launcher": dense_compat_active,
        "backend_mode_in_launcher": actual_mode,
        "native_dense_attention_preserved_in_launcher": native_dense_preserved,
        "runtime_native_attention_contract": runtime_attention_contract,
        "engine_contract": contract,
        "engine_contract_sha256": contract_hash,
        "seed": args.seed,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "bts_config": asdict(canonical_bts_config),
        "bts_config_sha256": expected_config_digest,
        "source_manifest_sha256": runtime_sources["sha256"],
        "source_manifest": runtime_sources,
        "dependency_contract_sha256": dependency_contract["sha256"],
        "dependency_contract": dependency_contract,
        "launcher_runtime_dependency_imports": launcher_runtime_dependency_imports,
        "lengths": [],
    }
    atomic_write_json(args.worker_output, report)
    llm = None
    audit_baseline: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    try:
        llm = LLM(**engine_options)
        # FastDeploy profiles the model during construction.  Snapshot the live
        # mmap counters only after that startup work so those calls cannot be
        # mistaken for benchmark requests.
        audit_baseline = wait_for_worker_audits(audit_dir)
        if not audit_baseline:
            raise RuntimeError(f"no {method} GPU-worker startup audit was produced")
        for context_length in args.lengths:
            prompt_tokens = (
                context_length - args.max_new_tokens - ENGINE_BOUNDARY_POSITIONS
            )
            if prompt_tokens + args.max_new_tokens >= max(args.lengths):
                raise RuntimeError(
                    "FastDeploy input/min-token boundary contract was violated"
                )
            prompt = deterministic_token_ids(prompt_tokens, vocab_size, args.seed + context_length)
            prompt_hash = token_ids_sha256(prompt)
            for _ in range(args.warmup_runs):
                warmup_output = llm.generate([prompt], sampling, use_tqdm=True)
                if len(_completion_token_ids(warmup_output)) != args.max_new_tokens:
                    raise RuntimeError("warmup did not return the requested output token count")
            latencies = []
            output_hashes = []
            for _ in range(args.measured_runs):
                started = time.perf_counter()
                outputs = llm.generate([prompt], sampling, use_tqdm=True)
                elapsed = time.perf_counter() - started
                generated = _completion_token_ids(outputs)
                if len(generated) != args.max_new_tokens:
                    raise RuntimeError(
                        f"expected {args.max_new_tokens} output tokens, got {len(generated)}"
                    )
                latencies.append(elapsed)
                output_hashes.append(token_ids_sha256(generated))
            latency_summary = summarize_samples(latencies)
            throughput_summary = summarize_samples(
                [prompt_tokens / elapsed for elapsed in latencies]
            )
            row = {
                "context_length": context_length,
                "prompt_tokens": prompt_tokens,
                "output_tokens_per_request": args.max_new_tokens,
                "prompt_sha256": prompt_hash,
                "engine_contract_sha256": contract_hash,
                "latency_seconds": latency_summary,
                "input_tokens_per_second": throughput_summary,
                "requests_per_second": summarize_samples([1.0 / value for value in latencies]),
                "output_token_sha256": output_hashes,
            }
            report["lengths"].append(row)
            atomic_write_json(args.worker_output, report)
            print(
                json.dumps(
                    {
                        "method": method,
                        "context_length": context_length,
                        "median_seconds": latency_summary["median"],
                        "median_input_tokens_per_second": throughput_summary["median"],
                    }
                ),
                flush=True,
            )
        # Read while the GPU worker is alive: mmap counters are the authoritative
        # request evidence and do not depend on an atexit hook.
        audits = wait_for_worker_audits(audit_dir)
        if not audits:
            raise RuntimeError(f"no {method} GPU-worker final audit was produced")
    finally:
        if llm is not None:
            llm.llm_engine._exit_sub_services()
    persisted_audit_dir = args.worker_output.parent / f"{method}-worker-audit"
    for audit in audits:
        atomic_write_json(
            persisted_audit_dir / f"worker-{int(audit['pid'])}.json", audit
        )
    expected_layers = set(range(int(model_info["num_hidden_layers"])))
    wrong_modes = [
        audit.get("backend_mode")
        for audit in audits
        if audit.get("backend_mode") != expected_mode
    ]
    if wrong_modes:
        raise RuntimeError(
            f"{method} worker audit mode mismatch: observed={wrong_modes}"
        )
    for audit in audits:
        worker_provenance = audit.get("package_provenance", {})
        for field in ("bts_file", "bts_fastdeploy_file", "required_source_root"):
            if worker_provenance.get(field) != provenance.get(field):
                raise RuntimeError(
                    f"{method} worker {field} differs from launcher provenance"
                )
        expected_audit_config = (
            expected_config_digest if method == "bts" else "not-applicable-dense"
        )
        if audit.get("config_sha256") != expected_audit_config:
            raise RuntimeError(
                f"{method} worker configuration digest mismatch: "
                f"expected={expected_audit_config}, "
                f"observed={audit.get('config_sha256')}"
            )
        if audit.get("dependency_contract_sha256") != dependency_contract["sha256"]:
            raise RuntimeError(
                f"{method} GPU executor dependency contract differs from launcher"
            )
        if audit.get("dependency_contract") != dependency_contract:
            raise RuntimeError(
                f"{method} GPU executor dependency manifest differs from launcher"
            )
        if audit.get("native_attention_contract") != runtime_attention_contract:
            raise RuntimeError(
                f"{method} GPU executor did not preserve the FA2/native-decode contract"
            )
        runtime_imports = audit.get("runtime_dependency_imports", {})
        if set(runtime_imports) != set(launcher_runtime_dependency_imports):
            raise RuntimeError(
                f"{method} GPU executor runtime dependency imports are incomplete"
            )
    expected_layer_calls = len(args.lengths) * (
        args.warmup_runs + args.measured_runs
    )
    if method == "bts":
        layer_calls = audit_layer_deltas(audit_baseline, audits, "layers")
        executed_layers = {layer for layer, count in layer_calls.items() if count > 0}
        if executed_layers != expected_layers:
            raise RuntimeError(
                "timed BTS worker did not prove execution in every model layer: "
                f"missing={sorted(expected_layers - executed_layers)}, "
                f"unexpected={sorted(executed_layers - expected_layers)}"
            )
        wrong_counts = {
            layer: layer_calls.get(layer, 0)
            for layer in expected_layers
            if layer_calls.get(layer, 0) != expected_layer_calls
        }
        if wrong_counts:
            raise RuntimeError(
                "BTS worker request deltas do not exactly match every requested "
                f"generation: expected={expected_layer_calls}, observed={wrong_counts}"
            )
        report["backend_evidence"] = {
            "adapter_active_in_launcher": True,
            "audit_counter_transport": "mmap_v1_tmpfs",
            "per_layer_file_io_during_timing": False,
            "per_layer_counter_store_during_timing": True,
            "per_request_audit_write_during_timing": False,
            "startup_profile_calls_excluded": True,
            "startup_audits": audit_baseline,
            "worker_audits": audits,
            "request_layer_call_deltas": {
                str(layer): layer_calls[layer] for layer in sorted(layer_calls)
            },
        }
    else:
        for audit in audits:
            native = audit.get("native_flash_attention", {})
            if native.get("preserved") is not True:
                raise RuntimeError(
                    "dense worker did not prove native flash_attn_func preservation"
                )
        rope_guard_calls = audit_layer_deltas(
            audit_baseline, audits, "rope_guard_layers"
        )
        executed_layers = {
            layer for layer, count in rope_guard_calls.items() if count > 0
        }
        if executed_layers != expected_layers:
            raise RuntimeError(
                "dense worker did not prove the semantic RoPE guard in every layer: "
                f"missing={sorted(expected_layers - executed_layers)}, "
                f"unexpected={sorted(executed_layers - expected_layers)}"
            )
        wrong_counts = {
            layer: rope_guard_calls.get(layer, 0)
            for layer in expected_layers
            if rope_guard_calls.get(layer, 0) != expected_layer_calls
        }
        if wrong_counts:
            raise RuntimeError(
                "dense worker request deltas do not exactly match every requested "
                f"generation: expected={expected_layer_calls}, observed={wrong_counts}"
            )
        report["backend_evidence"] = {
            "adapter_active_in_launcher": False,
            "dense_rope_compat_active_in_launcher": True,
            "native_dense_attention_unchanged": all(
                audit.get("native_flash_attention", {}).get("preserved") is True
                for audit in audits
            ),
            "audit_counter_transport": "mmap_v1_tmpfs",
            "per_layer_file_io_during_timing": False,
            "per_layer_counter_store_during_timing": True,
            "per_request_audit_write_during_timing": False,
            "startup_profile_calls_excluded": True,
            "startup_audits": audit_baseline,
            "worker_audits": audits,
            "request_rope_guard_layer_call_deltas": {
                str(layer): rope_guard_calls[layer]
                for layer in sorted(rope_guard_calls)
            },
        }
    report["status"] = "completed"
    report["finished_at"] = utc_now()
    atomic_write_json(args.worker_output, report)
    return 0


def subprocess_library_path(inherited: str | None = None) -> str:
    inherited = os.environ.get("LD_LIBRARY_PATH", "") if inherited is None else inherited
    candidates: list[Path] = []
    for entry in sys.path:
        base = Path(entry)
        candidates.append(base / "torch" / "lib")
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            candidates.extend(sorted(nvidia.glob("*/lib")))
    values = [str(path.resolve()) for path in candidates if path.is_dir()]
    values.extend(item for item in inherited.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(values))


def worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    python_paths = [str(SUBMISSION_ROOT), str(FASTDEPLOY_SRC)]
    python_paths.extend(item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
    env["LD_LIBRARY_PATH"] = subprocess_library_path(env.get("LD_LIBRARY_PATH", ""))
    env["BTS_SOURCE_ROOT"] = str(SUBMISSION_ROOT)
    env["OMP_NUM_THREADS"] = "1"
    return env


def method_order(restart: int) -> tuple[str, str]:
    return METHODS if restart % 2 == 0 else tuple(reversed(METHODS))


def _worker_command(args: argparse.Namespace, method: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        str(args.model),
        "--output",
        str(args.output),
        "--lengths",
        ",".join(str(value) for value in args.lengths),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measured-runs",
        str(args.measured_runs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--paddle-flash-attn-version",
        str(args.paddle_flash_attn_version),
        "--seed",
        str(args.seed),
        "--config",
        str(args.config),
        "--worker-method",
        method,
        "--worker-output",
        str(output),
    ]
    if args.expected_dependency_contract_sha256 is not None:
        command.extend(
            [
                "--expected-dependency-contract-sha256",
                args.expected_dependency_contract_sha256,
            ]
        )
    return command


def merge_restart_reports(
    restart_reports: Sequence[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Pool request-level measurements while preserving shared contracts."""
    merged: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        reports = [restart[method] for restart in restart_reports]
        first = reports[0]
        rows = []
        for first_row in first["lengths"]:
            length = first_row["context_length"]
            matching = [
                next(row for row in report["lengths"] if row["context_length"] == length)
                for report in reports
            ]
            for field in ("prompt_tokens", "prompt_sha256", "engine_contract_sha256"):
                if len({row[field] for row in matching}) != 1:
                    raise ValueError(f"restart {field} mismatch for {method} at {length}")
            latencies = [
                value
                for row in matching
                for value in row["latency_seconds"]["samples"]
            ]
            prompt_tokens = first_row["prompt_tokens"]
            rows.append(
                {
                    **{field: first_row[field] for field in (
                        "context_length",
                        "prompt_tokens",
                        "prompt_sha256",
                        "engine_contract_sha256",
                    )},
                    "latency_seconds": summarize_samples(latencies),
                    "input_tokens_per_second": summarize_samples(
                        [prompt_tokens / value for value in latencies]
                    ),
                    "requests_per_second": summarize_samples([1.0 / value for value in latencies]),
                }
            )
        merged[method] = {
            "method": method,
            "engine_contract": first["engine_contract"],
            "engine_contract_sha256": first["engine_contract_sha256"],
            "lengths": rows,
        }
    return merged


def controller_main(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(SUBMISSION_ROOT))
    sys.path.insert(0, str(FASTDEPLOY_SRC))
    from bts_fastdeploy.config import config_digest, load_config
    from bts_fastdeploy.upstream_contract import dependency_source_contract

    canonical_bts_config = load_config(args.config)
    expected_config_digest = config_digest(canonical_bts_config)
    runtime_sources = source_manifest()
    dependency_contract = dependency_source_contract()
    args.expected_dependency_contract_sha256 = dependency_contract["sha256"]
    model_info = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    model_contract = validate_model_contract(model_info, args.lengths)
    if args.max_new_tokens <= 0 or args.max_new_tokens >= min(args.lengths):
        raise ValueError("max-new-tokens must be positive and smaller than every context length")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("--output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL,
        "status": "dry_run" if args.dry_run else "running",
        "created_at": utc_now(),
        "scope": "single_gpu_single_request_offline_prefill_dominated",
        "non_claims": [
            "not concurrent serving throughput",
            "not FastDeploy scheduler throughput",
            "not vLLM throughput",
            "not task accuracy or quality",
            "not an average compute-pruning measurement",
        ],
        "quality_gate": {
            "status": "NOT_EVALUATED_BY_THIS_RUNNER",
            "required_for_full_accuracy_preserving_claim": True,
            "recommended_suites": ["RULER paired dense/BTS", "LongBench paired dense/BTS"],
        },
        "compute_pruning_gate": {
            "status": "NOT_EVALUATED_BY_THIS_RUNNER",
            "required_for_90pct_pruning_claim": True,
        },
        "model": str(args.model),
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "model_contract": model_contract,
        "source": {
            **runtime_sources,
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "bts_config_file_sha256": file_sha256(args.config),
            "bts_config_canonical_sha256": expected_config_digest,
        },
        "dependency_contract": dependency_contract,
        "lengths": list(args.lengths),
        "prompt_tokens": [
            value - args.max_new_tokens - ENGINE_BOUNDARY_POSITIONS
            for value in args.lengths
        ],
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "restarts": args.restarts,
        "target_speedup": args.target_speedup,
        "bts_config_file": json.loads(args.config.read_text(encoding="utf-8")),
        "bts_config_canonical": asdict(canonical_bts_config),
        "engine_contract": engine_contract(args),
        "engine_contract_sha256": canonical_sha256(engine_contract(args)),
        "restart_plan": [list(method_order(index)) for index in range(args.restarts)],
    }
    atomic_write_json(args.output / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    # A source-only PYTHONPATH makes BTS importable in the controller but does
    # not create the distribution entry point used by FastDeploy's fresh GPU
    # workers.  Check the exact PYTHON_BIN environment before waiting for an
    # idle GPU or loading model weights.
    from bts_fastdeploy.environment import (
        validate_fastdeploy_plugin_entry_point,
    )

    manifest["fastdeploy_plugin_entry_point"] = (
        validate_fastdeploy_plugin_entry_point()
    )
    atomic_write_json(args.output / "manifest.json", manifest)

    initial_gpu = wait_for_selected_gpu_idle(args.gpu_idle_timeout_seconds)
    physical_gpu = initial_gpu["physical_gpu"]
    if (
        "H100" not in physical_gpu.get("name", "").upper()
        or physical_gpu.get("compute_cap") != "9.0"
    ):
        raise RuntimeError(
            "formal FastDeploy comparison requires an NVIDIA H100 (compute capability "
            f"9.0); found name={physical_gpu.get('name')!r}, "
            f"compute_cap={physical_gpu.get('compute_cap')!r}"
        )
    manifest["gpu_exclusivity"] = {
        "required": True,
        "initial_idle_snapshot": initial_gpu,
        "boundary_snapshots": [],
        "runtime_monitors": [],
    }
    atomic_write_json(args.output / "manifest.json", manifest)
    env = worker_environment()
    restart_reports: list[dict[str, dict[str, Any]]] = []
    for restart in range(args.restarts):
        current: dict[str, dict[str, Any]] = {}
        for method in method_order(restart):
            if source_manifest() != runtime_sources:
                raise RuntimeError(
                    "submitted runtime sources changed after the benchmark controller "
                    "started; discard this run"
                )
            if dependency_source_contract() != dependency_contract:
                raise RuntimeError(
                    "approved FastDeploy/Paddle dependency sources changed after "
                    "the benchmark controller started; discard this run"
                )
            before_gpu = wait_for_selected_gpu_idle(0.0)
            manifest["gpu_exclusivity"]["boundary_snapshots"].append(
                {
                    "restart": restart,
                    "method": method,
                    "boundary": "before",
                    "snapshot": before_gpu,
                }
            )
            atomic_write_json(args.output / "manifest.json", manifest)
            method_dir = args.output / f"restart-{restart:03d}" / method
            method_dir.mkdir(parents=True, exist_ok=True)
            output = method_dir / "worker_report.json"
            command = _worker_command(args, method, output)
            atomic_write_json(
                method_dir / "command.json",
                {"argv": command, "method": method, "started_at": utc_now()},
            )
            print(
                f"[restart {restart + 1}/{args.restarts}] running {method}",
                flush=True,
            )
            with (method_dir / "worker.log").open("w", encoding="utf-8") as log:
                try:
                    returncode, gpu_monitor = run_worker_subprocess(
                        command,
                        env=env,
                        log=log,
                        timeout_seconds=args.worker_timeout_seconds,
                    )
                except subprocess.TimeoutExpired as error:
                    manifest["status"] = "failed"
                    manifest["failed_method"] = method
                    manifest["failed_restart"] = restart
                    manifest["failure"] = (
                        f"worker exceeded {args.worker_timeout_seconds} seconds"
                    )
                    atomic_write_json(args.output / "manifest.json", manifest)
                    raise RuntimeError(manifest["failure"]) from error
                except Exception as error:
                    manifest["status"] = "failed"
                    manifest["failed_method"] = method
                    manifest["failed_restart"] = restart
                    manifest["failure"] = str(error)
                    atomic_write_json(args.output / "manifest.json", manifest)
                    raise
            manifest["gpu_exclusivity"]["runtime_monitors"].append(
                {
                    "restart": restart,
                    "method": method,
                    **gpu_monitor,
                }
            )
            if gpu_monitor["check_count"] <= 0:
                raise RuntimeError(
                    f"{method} worker completed without a runtime GPU exclusivity check"
                )
            if source_manifest() != runtime_sources:
                raise RuntimeError(
                    "submitted runtime sources changed during a benchmark arm; "
                    "discard this run"
                )
            if dependency_source_contract() != dependency_contract:
                raise RuntimeError(
                    "approved FastDeploy/Paddle dependency sources changed during "
                    "a benchmark arm; discard this run"
                )
            if returncode:
                manifest["status"] = "failed"
                manifest["failed_method"] = method
                manifest["failed_restart"] = restart
                manifest["worker_exit_code"] = returncode
                atomic_write_json(args.output / "manifest.json", manifest)
                raise RuntimeError(
                    f"{method} worker failed with exit code {returncode}; "
                    f"see {method_dir / 'worker.log'}"
                )
            after_gpu = wait_for_selected_gpu_idle(30.0, poll_seconds=1.0)
            manifest["gpu_exclusivity"]["boundary_snapshots"].append(
                {
                    "restart": restart,
                    "method": method,
                    "boundary": "after",
                    "snapshot": after_gpu,
                }
            )
            atomic_write_json(args.output / "manifest.json", manifest)
            current[method] = json.loads(output.read_text(encoding="utf-8"))
            if current[method].get("status") != "completed":
                raise RuntimeError(f"{method} worker did not produce a completed report")
            if current[method].get("source_manifest") != runtime_sources:
                raise RuntimeError(
                    f"{method} worker runtime-source manifest differs from controller"
                )
            if current[method].get("bts_config_sha256") != expected_config_digest:
                raise RuntimeError(
                    f"{method} worker canonical BTS configuration differs from controller"
                )
            if current[method].get("dependency_contract") != dependency_contract:
                raise RuntimeError(
                    f"{method} worker dependency-source contract differs from controller"
                )
        restart_reports.append(current)

    pooled = merge_restart_reports(restart_reports)
    comparison = build_comparison(
        restart_reports,
        target_speedup=args.target_speedup,
        bootstrap_samples=args.bootstrap_samples,
    )
    report = {
        **manifest,
        "status": "completed",
        "finished_at": utc_now(),
        "pooled_methods": pooled,
        "pooled_methods_note": "descriptive only; not used for confidence or verdict",
        "comparison": comparison,
        "claim": (
            "Within FastDeploy 2.5, single-request prefill-dominated BTS "
            "throughput versus FastDeploy's native dense attention kernel; "
            "both PaddleFormers arms use the same semantic identity-RoPE guard."
        ),
    }
    atomic_write_json(args.output / "benchmark_report.json", report)
    manifest["status"] = "completed"
    manifest["verdict"] = comparison["verdict"]
    atomic_write_json(args.output / "manifest.json", manifest)
    print(json.dumps(comparison, indent=2, ensure_ascii=False), flush=True)
    if comparison["verdict"] == "PASS":
        return 0
    return 2 if comparison["verdict"] == "FAIL" else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen3 BF16 model")
    parser.add_argument("--output", type=Path, required=True, help="Output directory outside source")
    parser.add_argument("--lengths", type=parse_lengths, default=DEFAULT_LENGTHS)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=FORMAL_WARMUP_RUNS)
    parser.add_argument("--measured-runs", type=int, default=FORMAL_MEASURED_RUNS)
    parser.add_argument("--restarts", type=int, default=FORMAL_RESTARTS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--paddle-flash-attn-version",
        type=int,
        choices=(FORMAL_FLASH_ATTN_VERSION,),
        default=FORMAL_FLASH_ATTN_VERSION,
        help="Frozen dense baseline version: FlashAttention 2",
    )
    parser.add_argument("--worker-timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--gpu-idle-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "wait this long for NVIDIA driver access and for the selected GPU "
            "to have no compute processes"
        ),
    )
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--target-speedup", type=float, default=FORMAL_TARGET_SPEEDUP)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=FORMAL_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SUBMISSION_ROOT / "integrations" / "fastdeploy" / "configs" / "topk040_tail256.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--expected-dependency-contract-sha256",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")
    if args.measured_runs <= 0 or args.restarts <= 0:
        parser.error("--measured-runs and --restarts must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    if args.gpu_idle_timeout_seconds < 0:
        parser.error("--gpu-idle-timeout-seconds must be non-negative")
    if args.target_speedup <= 0:
        parser.error("--target-speedup must be positive")
    if not (args.model / "config.json").is_file():
        parser.error("--model must contain config.json")
    if not args.config.is_file():
        parser.error("--config does not exist")
    if args.worker_method:
        if args.worker_output is None:
            parser.error("internal worker requires --worker-output")
        return worker_main(args)
    if args.worker_output is not None:
        parser.error("--worker-output is internal and requires --worker-method")
    try:
        return controller_main(args)
    except ValueError as error:
        mark_manifest_failed(args.output, error)
        parser.error(str(error))
    except (Exception, KeyboardInterrupt) as error:
        mark_manifest_failed(args.output, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

