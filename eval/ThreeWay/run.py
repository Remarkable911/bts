#!/usr/bin/env python3
"""Validate and analyze a frozen four-arm, three-comparison benchmark.

This module deliberately does not launch either inference engine.  It accepts one
normalized report for each engine/attention arm, rejects reports that do not share
the exact same workload and benchmark contract, and performs a paired bootstrap
over five fresh-process restarts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bts_four_arm_input_v1"
ANALYSIS_SCHEMA_VERSION = "bts_three_comparison_analysis_v1"
PROTOCOL_NAME = "bts_four_arm_128k_prefill_v1"

FORMAL_CONTEXT_LENGTH = 131072
# FastDeploy needs one position beyond prompt + generated tokens.  Using 131069
# prompt tokens and two generated tokens is the largest common 128K case that
# also executes one real decode step in both engines.
FORMAL_INPUT_TOKENS = 131069
FORMAL_OUTPUT_TOKENS = 2
FORMAL_RESTARTS = 5
FORMAL_BOOTSTRAP_SAMPLES = 10000
FORMAL_BOOTSTRAP_SEED = 20261110
FORMAL_CONFIDENCE = 0.95
FORMAL_TARGET_SPEEDUP = 1.5

ARMS = ("fd_dense", "fd_bts", "vllm_dense", "vllm_bts")
ARM_SPECS = {
    "fd_dense": ("fastdeploy", "dense"),
    "fd_bts": ("fastdeploy", "bts"),
    "vllm_dense": ("vllm", "dense"),
    "vllm_bts": ("vllm", "bts"),
}
COMPARISONS = {
    "fd_bts_over_fd_dense": ("fd_bts", "fd_dense"),
    "vllm_bts_over_vllm_dense": ("vllm_bts", "vllm_dense"),
    "fd_bts_over_vllm_dense": ("fd_bts", "vllm_dense"),
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "arm",
    "shared_contract",
    "shared_contract_sha256",
    "workload",
    "workload_sha256",
    "runtime",
    "restarts",
}
_SHARED_CONTRACT_KEYS = {
    "protocol",
    "context_length",
    "input_tokens_per_request",
    "output_tokens_per_request",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "dtype",
    "kv_cache_dtype",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "enforce_eager",
    "prefix_caching",
    "chunked_prefill",
    "dense_attention_backend",
    "bts_dense_fallback_policy",
    "bts_optimization_scope",
    "offered_load",
    "timing_scope",
    "temperature",
    "min_output_tokens",
    "ignore_eos",
    "seed",
    "logical_model_sha256",
    "tokenizer_sha256",
    "bts_config_sha256",
    "gpu_uuid",
    "gpu_architecture",
    "gpu_count",
}
_WORKLOAD_KEYS = {
    "schema_version",
    "request_count",
    "context_length",
    "input_tokens_per_request",
    "output_tokens_per_request",
    "ordered_request_ids_sha256",
    "ordered_prompt_tokens_sha256",
}
_RUNTIME_KEYS = {
    "engine",
    "engine_version",
    "attention_mode",
    "prefill_backend",
    "requested_dense_attention_backend",
    "observed_dense_attention_backend",
    "decode_backend",
    "bts_prefill_enabled",
    "bts_decode_enabled",
    "engine_config_sha256",
    "source_manifest_sha256",
}
_RESTART_KEYS = {
    "restart",
    "measured_seconds",
    "completed_requests",
    "total_input_tokens",
    "total_output_tokens",
    "fresh_process",
    "gpu_exclusive",
    "request_order_sha256",
    "runtime_source_sha256",
    "dense_prefill_calls",
    "bts_prefill_calls",
    "dense_decode_calls",
    "bts_decode_calls",
}


class ProtocolError(ValueError):
    """Raised when an input cannot support the frozen comparison claim."""


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON representation used by the report contract."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ProtocolError(
            f"{label} keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolError(f"{label} must be a 64-character lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError(f"{label} must be a lowercase hexadecimal SHA-256")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{label} must be an integer >= {minimum}")
    return value


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ProtocolError(f"{label} must be a positive finite number")
    return result


def _expect_fixed(
    value: Mapping[str, Any], field: str, expected: Any, label: str
) -> None:
    actual = value[field]
    if isinstance(expected, bool):
        type_matches = isinstance(actual, bool)
    elif isinstance(expected, int):
        type_matches = isinstance(actual, int) and not isinstance(actual, bool)
    else:
        type_matches = isinstance(actual, type(expected))
    if not type_matches or actual != expected:
        raise ProtocolError(
            f"{label}.{field} must be {expected!r}, got {actual!r}"
        )


def _fixed_shared_contract_values() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "context_length": FORMAL_CONTEXT_LENGTH,
        "input_tokens_per_request": FORMAL_INPUT_TOKENS,
        "output_tokens_per_request": FORMAL_OUTPUT_TOKENS,
        "max_model_len": FORMAL_CONTEXT_LENGTH,
        "max_num_seqs": 1,
        "max_num_batched_tokens": FORMAL_CONTEXT_LENGTH,
        "gpu_memory_utilization": 0.90,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enforce_eager": True,
        "prefix_caching": False,
        "chunked_prefill": False,
        "dense_attention_backend": "flash_attention_2",
        "bts_dense_fallback_policy": "engine_native_dense",
        "bts_optimization_scope": "prefill_only",
        "offered_load": "infinite",
        "timing_scope": "end_to_end_generation_excludes_startup_and_warmup",
        "temperature": 0.0,
        "min_output_tokens": FORMAL_OUTPUT_TOKENS,
        "ignore_eos": True,
        "seed": FORMAL_BOOTSTRAP_SEED,
        "gpu_architecture": "sm90",
        "gpu_count": 1,
    }


def build_shared_contract(
    *,
    logical_model_sha256: str,
    tokenizer_sha256: str,
    bts_config_sha256: str,
    gpu_uuid: str,
) -> dict[str, Any]:
    """Build the fixed shared object that every arm must embed verbatim."""

    contract = _fixed_shared_contract_values()
    contract.update(
        {
            "logical_model_sha256": logical_model_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "bts_config_sha256": bts_config_sha256,
            "gpu_uuid": gpu_uuid,
        }
    )
    _validate_shared_contract(contract, "shared_contract")
    return contract


def build_workload_contract(
    *,
    request_count: int,
    ordered_request_ids_sha256: str,
    ordered_prompt_tokens_sha256: str,
) -> dict[str, Any]:
    """Build the normalized workload identity embedded by all four workers."""

    workload = {
        "schema_version": "uniform_128k_requests_v1",
        "request_count": request_count,
        "context_length": FORMAL_CONTEXT_LENGTH,
        "input_tokens_per_request": FORMAL_INPUT_TOKENS,
        "output_tokens_per_request": FORMAL_OUTPUT_TOKENS,
        "ordered_request_ids_sha256": ordered_request_ids_sha256,
        "ordered_prompt_tokens_sha256": ordered_prompt_tokens_sha256,
    }
    _validate_workload(workload, "workload")
    return workload


def build_arm_report(
    *,
    arm: str,
    shared_contract: Mapping[str, Any],
    workload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    restarts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble, hash, and validate a normalized report for a worker."""

    report = {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "shared_contract": dict(shared_contract),
        "shared_contract_sha256": canonical_sha256(shared_contract),
        "workload": dict(workload),
        "workload_sha256": canonical_sha256(workload),
        "runtime": dict(runtime),
        "restarts": [dict(restart) for restart in restarts],
    }
    validate_arm_report(report, expected_arm=arm)
    return report


def _validate_shared_contract(raw: Any, label: str) -> Mapping[str, Any]:
    contract = _object(raw, label)
    _exact_keys(contract, _SHARED_CONTRACT_KEYS, label)
    fixed = _fixed_shared_contract_values()
    for field, expected in fixed.items():
        _expect_fixed(contract, field, expected, label)
    for field in (
        "logical_model_sha256",
        "tokenizer_sha256",
        "bts_config_sha256",
    ):
        _sha256(contract[field], f"{label}.{field}")
    _nonempty_string(contract["gpu_uuid"], f"{label}.gpu_uuid")
    return contract


def _validate_workload(raw: Any, label: str) -> Mapping[str, Any]:
    workload = _object(raw, label)
    _exact_keys(workload, _WORKLOAD_KEYS, label)
    fixed = {
        "schema_version": "uniform_128k_requests_v1",
        "context_length": FORMAL_CONTEXT_LENGTH,
        "input_tokens_per_request": FORMAL_INPUT_TOKENS,
        "output_tokens_per_request": FORMAL_OUTPUT_TOKENS,
    }
    for field, expected in fixed.items():
        _expect_fixed(workload, field, expected, label)
    _integer(workload["request_count"], f"{label}.request_count", minimum=16)
    _sha256(
        workload["ordered_request_ids_sha256"],
        f"{label}.ordered_request_ids_sha256",
    )
    _sha256(
        workload["ordered_prompt_tokens_sha256"],
        f"{label}.ordered_prompt_tokens_sha256",
    )
    return workload


def _validate_runtime(raw: Any, arm: str, label: str) -> Mapping[str, Any]:
    runtime = _object(raw, label)
    _exact_keys(runtime, _RUNTIME_KEYS, label)
    engine, attention_mode = ARM_SPECS[arm]
    _expect_fixed(runtime, "engine", engine, label)
    _expect_fixed(runtime, "attention_mode", attention_mode, label)
    _nonempty_string(runtime["engine_version"], f"{label}.engine_version")
    _expect_fixed(
        runtime,
        "prefill_backend",
        "bts" if attention_mode == "bts" else "flash_attention_2",
        label,
    )
    for field in (
        "requested_dense_attention_backend",
        "observed_dense_attention_backend",
    ):
        _expect_fixed(runtime, field, "flash_attention_2", label)
    _expect_fixed(
        runtime,
        "decode_backend",
        (
            "native_append_attention"
            if engine == "fastdeploy"
            else "native_vllm_flash_attention_2"
        ),
        label,
    )
    _expect_fixed(runtime, "bts_prefill_enabled", attention_mode == "bts", label)
    _expect_fixed(runtime, "bts_decode_enabled", False, label)
    _sha256(runtime["engine_config_sha256"], f"{label}.engine_config_sha256")
    _sha256(runtime["source_manifest_sha256"], f"{label}.source_manifest_sha256")
    return runtime


def validate_arm_report(
    raw: Any, *, expected_arm: str | None = None, label: str = "report"
) -> Mapping[str, Any]:
    """Validate one arm and return it unchanged after fail-closed checks."""

    report = _object(raw, label)
    _exact_keys(report, _TOP_LEVEL_KEYS, label)
    _expect_fixed(report, "schema_version", SCHEMA_VERSION, label)
    arm = report["arm"]
    if not isinstance(arm, str):
        raise ProtocolError(f"{label}.arm must be a string")
    if arm not in ARM_SPECS:
        raise ProtocolError(f"{label}.arm must be one of {list(ARMS)}, got {arm!r}")
    if expected_arm is not None and arm != expected_arm:
        raise ProtocolError(f"{label}.arm must be {expected_arm!r}, got {arm!r}")

    contract = _validate_shared_contract(
        report["shared_contract"], f"{label}.shared_contract"
    )
    declared_contract_hash = _sha256(
        report["shared_contract_sha256"], f"{label}.shared_contract_sha256"
    )
    actual_contract_hash = canonical_sha256(contract)
    if declared_contract_hash != actual_contract_hash:
        raise ProtocolError(f"{label}.shared_contract_sha256 does not match its object")

    workload = _validate_workload(report["workload"], f"{label}.workload")
    declared_workload_hash = _sha256(
        report["workload_sha256"], f"{label}.workload_sha256"
    )
    actual_workload_hash = canonical_sha256(workload)
    if declared_workload_hash != actual_workload_hash:
        raise ProtocolError(f"{label}.workload_sha256 does not match its object")

    runtime = _validate_runtime(report["runtime"], arm, f"{label}.runtime")
    restarts = report["restarts"]
    if not isinstance(restarts, list) or len(restarts) != FORMAL_RESTARTS:
        raise ProtocolError(
            f"{label}.restarts must contain exactly {FORMAL_RESTARTS} restarts"
        )

    expected_request_count = workload["request_count"]
    expected_input_tokens = expected_request_count * FORMAL_INPUT_TOKENS
    expected_output_tokens = expected_request_count * FORMAL_OUTPUT_TOKENS
    expected_request_hash = workload["ordered_request_ids_sha256"]
    expected_source_hash = runtime["source_manifest_sha256"]
    restart_ids: set[int] = set()
    for position, raw_restart in enumerate(restarts):
        restart_label = f"{label}.restarts[{position}]"
        restart = _object(raw_restart, restart_label)
        _exact_keys(restart, _RESTART_KEYS, restart_label)
        restart_id = _integer(restart["restart"], f"{restart_label}.restart")
        restart_ids.add(restart_id)
        _positive_finite(restart["measured_seconds"], f"{restart_label}.measured_seconds")
        _expect_fixed(restart, "completed_requests", expected_request_count, restart_label)
        _expect_fixed(restart, "total_input_tokens", expected_input_tokens, restart_label)
        _expect_fixed(restart, "total_output_tokens", expected_output_tokens, restart_label)
        _expect_fixed(restart, "fresh_process", True, restart_label)
        _expect_fixed(restart, "gpu_exclusive", True, restart_label)
        _expect_fixed(restart, "request_order_sha256", expected_request_hash, restart_label)
        _expect_fixed(restart, "runtime_source_sha256", expected_source_hash, restart_label)

        counters = {
            field: _integer(restart[field], f"{restart_label}.{field}")
            for field in (
                "dense_prefill_calls",
                "bts_prefill_calls",
                "dense_decode_calls",
                "bts_decode_calls",
            )
        }
        if counters["dense_decode_calls"] <= 0:
            raise ProtocolError(
                f"{restart_label}.dense_decode_calls must prove a real dense decode step"
            )
        if counters["bts_decode_calls"] != 0:
            raise ProtocolError(f"{restart_label} used BTS during decode")
        if ARM_SPECS[arm][1] == "bts":
            if counters["bts_prefill_calls"] <= 0 or counters["dense_prefill_calls"] != 0:
                raise ProtocolError(
                    f"{restart_label} must use BTS, and only BTS, for measured prefill"
                )
        elif counters["dense_prefill_calls"] <= 0 or counters["bts_prefill_calls"] != 0:
            raise ProtocolError(
                f"{restart_label} must use FA2, and not BTS, for measured prefill"
            )

    expected_restart_ids = set(range(FORMAL_RESTARTS))
    if restart_ids != expected_restart_ids:
        raise ProtocolError(
            f"{label}.restart ids must be {sorted(expected_restart_ids)}, "
            f"got {sorted(restart_ids)}"
        )
    return report


def load_arm_report(path: Path, *, expected_arm: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load {path}: {error}") from error
    return validate_arm_report(raw, expected_arm=expected_arm, label=str(path))


def _lower_empirical_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ProtocolError("cannot summarize an empty bootstrap distribution")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.floor(probability * (len(ordered) - 1))))
    return ordered[index]


def analyze_arm_reports(reports: Mapping[str, Any]) -> dict[str, Any]:
    """Validate four arm reports and return the frozen three-comparison result."""

    if set(reports) != set(ARMS):
        raise ProtocolError(
            f"reports must contain exactly {list(ARMS)}, got {sorted(reports)}"
        )
    validated = {
        arm: validate_arm_report(reports[arm], expected_arm=arm, label=arm)
        for arm in ARMS
    }

    first = validated[ARMS[0]]
    for arm in ARMS[1:]:
        report = validated[arm]
        if report["shared_contract"] != first["shared_contract"]:
            raise ProtocolError(f"{arm} does not share the exact benchmark contract")
        if report["shared_contract_sha256"] != first["shared_contract_sha256"]:
            raise ProtocolError(f"{arm} shared contract hash differs")
        if report["workload"] != first["workload"]:
            raise ProtocolError(f"{arm} does not share the exact frozen workload")
        if report["workload_sha256"] != first["workload_sha256"]:
            raise ProtocolError(f"{arm} workload hash differs")

    for engine, dense_arm, bts_arm in (
        ("fastdeploy", "fd_dense", "fd_bts"),
        ("vllm", "vllm_dense", "vllm_bts"),
    ):
        dense_runtime = validated[dense_arm]["runtime"]
        bts_runtime = validated[bts_arm]["runtime"]
        for field in (
            "engine_version",
            "engine_config_sha256",
            "source_manifest_sha256",
        ):
            if dense_runtime[field] != bts_runtime[field]:
                raise ProtocolError(
                    f"{engine} dense/BTS {field} must match exactly"
                )

    throughputs: dict[str, dict[int, float]] = {}
    for arm, report in validated.items():
        rows = {row["restart"]: row for row in report["restarts"]}
        throughputs[arm] = {
            restart: row["total_input_tokens"] / float(row["measured_seconds"])
            for restart, row in rows.items()
        }

    comparison_rows: dict[str, list[dict[str, Any]]] = {}
    speedups: dict[str, list[float]] = {}
    for name, (numerator, denominator) in COMPARISONS.items():
        rows = []
        ratios = []
        for restart in range(FORMAL_RESTARTS):
            numerator_throughput = throughputs[numerator][restart]
            denominator_throughput = throughputs[denominator][restart]
            ratio = numerator_throughput / denominator_throughput
            if not math.isfinite(ratio) or ratio <= 0:
                raise ProtocolError(f"{name} produced an invalid restart ratio")
            rows.append(
                {
                    "restart": restart,
                    "numerator_input_tokens_per_second": numerator_throughput,
                    "denominator_input_tokens_per_second": denominator_throughput,
                    "throughput_speedup": ratio,
                }
            )
            ratios.append(ratio)
        comparison_rows[name] = rows
        speedups[name] = ratios

    rng = random.Random(FORMAL_BOOTSTRAP_SEED)
    bootstrap_draws = {name: [] for name in COMPARISONS}
    for _ in range(FORMAL_BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(FORMAL_RESTARTS) for _ in range(FORMAL_RESTARTS)]
        for name, ratios in speedups.items():
            bootstrap_draws[name].append(
                statistics.median(ratios[index] for index in indices)
            )

    comparisons: dict[str, dict[str, Any]] = {}
    for name, (numerator, denominator) in COMPARISONS.items():
        point = statistics.median(speedups[name])
        lower = _lower_empirical_quantile(
            bootstrap_draws[name], 1.0 - FORMAL_CONFIDENCE
        )
        comparisons[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "metric": "input_tokens_per_second",
            "paired_restart_speedups": speedups[name],
            "paired_restarts": comparison_rows[name],
            "median_throughput_speedup": point,
            "bootstrap_one_sided_95pct_lower_bound": lower,
            "target_speedup": FORMAL_TARGET_SPEEDUP,
            "point_estimate_meets_target": point >= FORMAL_TARGET_SPEEDUP,
            "target_passed": lower >= FORMAL_TARGET_SPEEDUP,
        }

    all_passed = all(row["target_passed"] for row in comparisons.values())
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "context_length": FORMAL_CONTEXT_LENGTH,
        "input_tokens_per_request": FORMAL_INPUT_TOKENS,
        "output_tokens_per_request": FORMAL_OUTPUT_TOKENS,
        "shared_contract_sha256": first["shared_contract_sha256"],
        "workload_sha256": first["workload_sha256"],
        "arm_source_manifest_sha256": {
            arm: validated[arm]["runtime"]["source_manifest_sha256"]
            for arm in ARMS
        },
        "statistics": {
            "analysis_unit": "paired_fresh_process_restart",
            "restart_count": FORMAL_RESTARTS,
            "point_statistic": "median_of_paired_restart_throughput_ratios",
            "bootstrap": {
                "type": "paired_cluster_percentile",
                "shared_restart_indices_across_all_three_comparisons": True,
                "confidence": FORMAL_CONFIDENCE,
                "replicates": FORMAL_BOOTSTRAP_SAMPLES,
                "seed": FORMAL_BOOTSTRAP_SEED,
                "quantile_method": "lower_floor_q_times_n_minus_1",
            },
        },
        "hard_gate": {
            "context_length": FORMAL_CONTEXT_LENGTH,
            "target_speedup": FORMAL_TARGET_SPEEDUP,
            "basis": "one_sided_95pct_paired_bootstrap_lower_bound",
            "all_three_comparisons_must_pass": True,
        },
        "comparisons": comparisons,
        "overall_status": "PASS" if all_passed else "FAIL",
    }


def analyze_paths(paths: Mapping[str, Path]) -> dict[str, Any]:
    reports = {
        arm: load_arm_report(paths[arm], expected_arm=arm)
        for arm in ARMS
    }
    return analyze_arm_reports(reports)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate four normalized arms and calculate three 128K ratios"
    )
    parser.add_argument("--fd-dense", type=Path, required=True)
    parser.add_argument("--fd-bts", type=Path, required=True)
    parser.add_argument("--vllm-dense", type=Path, required=True)
    parser.add_argument("--vllm-bts", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the analysis report here; refuses to overwrite an existing file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {
        "fd_dense": args.fd_dense,
        "fd_bts": args.fd_bts,
        "vllm_dense": args.vllm_dense,
        "vllm_bts": args.vllm_bts,
    }
    try:
        result = analyze_paths(paths)
        rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        print(rendered, end="")
    except (ProtocolError, FileExistsError) as error:
        print(f"protocol error: {error}", file=sys.stderr)
        return 4
    return 0 if result["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

