#!/usr/bin/env python3
"""Compare stock vLLM dense attention with the BTS vLLM backend.

Both arms use the same vLLM process, scheduler, model, KV-cache policy and
RULER-derived request trace.  Servers are restarted for every statistical
pair and arm order alternates.  The benchmark can measure either saturated
request throughput (``--request-rates inf``) or SLO goodput over a frozen,
common request-rate grid (``--slo-config``).

This two-arm protocol measures the complete BTS attention integration inside
the stock vLLM scheduler.  It does not by itself attribute a speedup to a new
scheduler; that requires dense/sparse x stock/new-scheduler ablations.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
SUBMISSION_ROOT = HERE.parents[1]
PROTOCOL = "vllm_dense_vs_bts_ruler_serving_fa2_v3"
WORKLOAD_PROTOCOL = "ruler_vllm_serving_workload_v1"
SUPPORTED_VLLM_VERSION = "0.17.0"
PLUGIN_ENTRY_POINT = "bts_vllm"
DEFAULT_LENGTHS = (32768, 65536, 131072)
EXPECTED_RULER_TASKS = (
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
FORMAL_SAMPLES_PER_TASK = 8
FORMAL_SEED = 42
FORMAL_BOOTSTRAP_SEED = 20261111
METHODS = ("dense", "bts")
MIN_FORMAL_RESTARTS = 5
RUN_TOKEN_ENV = "BTS_VLLM_RUN_TOKEN"
PYTORCH_ALLOC_CONF_ENV = "PYTORCH_ALLOC_CONF"
PYTORCH_CUDA_ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
FORMAL_PYTORCH_ALLOC_CONF = "expandable_segments:True"
FORMAL_FLASH_ATTN_VERSION = 2


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


def parse_lengths(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--lengths must contain integers") from error
    if not result or len(result) != len(set(result)) or any(item <= 1 for item in result):
        raise argparse.ArgumentTypeError(
            "--lengths must contain unique integers greater than one"
        )
    return result


def rate_label(rate: float) -> str:
    return "inf" if math.isinf(rate) else format(rate, ".12g")


def parse_rates(value: str) -> tuple[float, ...]:
    result = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        try:
            rate = math.inf if item in {"inf", "infinity"} else float(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "--request-rates must contain positive numbers or inf"
            ) from error
        if rate <= 0 or math.isnan(rate):
            raise argparse.ArgumentTypeError("request rates must be positive")
        result.append(rate)
    if not result or len({rate_label(rate) for rate in result}) != len(result):
        raise argparse.ArgumentTypeError("request rates must be nonempty and unique")
    finite = [rate for rate in result if math.isfinite(rate)]
    if finite != sorted(finite) or (any(math.isinf(rate) for rate in result) and not math.isinf(result[-1])):
        raise argparse.ArgumentTypeError(
            "request rates must be ascending, with inf last when present"
        )
    return tuple(result)


def parse_extra_args(value: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shell-style argument string: {error}")


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(float(value) for value in values)
    index = min(
        len(ordered) - 1,
        max(0, math.floor(probability * (len(ordered) - 1))),
    )
    return ordered[index]


def paired_bootstrap_lower_bounds(
    speedups: Mapping[int, Sequence[float]],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> dict[int, float]:
    if not speedups:
        raise ValueError("bootstrap requires at least one length")
    counts = {len(values) for values in speedups.values()}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("all lengths need the same number of at least two pairs")
    count = next(iter(counts))
    rng = random.Random(seed)
    draws = {length: [] for length in speedups}
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        for length, values in speedups.items():
            draws[length].append(statistics.median(values[index] for index in indices))
    probability = 1.0 - confidence
    return {
        length: percentile(values, probability) for length, values in draws.items()
    }


def load_slo_config(path: Path | None, lengths: Sequence[int]) -> dict[int, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"ttft", "tpot", "e2el"}
    result: dict[int, dict[str, float]] = {}
    for length in lengths:
        raw = payload.get(str(length))
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"SLO config is missing a nonempty {length} object")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"SLO config {length} has unknown metrics: {sorted(unknown)}")
        normalized = {name: float(value) for name, value in raw.items()}
        if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
            raise ValueError(f"SLO config {length} values must be finite and nonnegative")
        result[length] = normalized
    return result


def _read_jsonl_metadata(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"prompt", "output_tokens", "prompt_tokens", "context_length"}
            if not required <= set(row):
                raise ValueError(f"{path}:{line_number} lacks {sorted(required - set(row))}")
            rows.append(row)
    if not rows:
        raise ValueError(f"workload file is empty: {path}")
    return rows


def load_workload(
    root: Path,
    lengths: Sequence[int],
    num_prompts: int | None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json" if root.is_dir() else root
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != WORKLOAD_PROTOCOL:
        raise ValueError(
            f"workload protocol must be {WORKLOAD_PROTOCOL!r}; "
            f"found {manifest.get('protocol')!r}"
        )
    declared_sha = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if declared_sha != canonical_sha256(unhashed):
        raise ValueError("workload manifest_sha256 does not match its content")

    cases = {}
    for length in lengths:
        entry = manifest.get("lengths", {}).get(str(length))
        if not isinstance(entry, dict):
            raise ValueError(f"workload lacks context length {length}")
        path = (manifest_path.parent / str(entry["path"])).resolve()
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"workload file is missing or changed: {path}")
        rows = _read_jsonl_metadata(path)
        count = len(rows) if num_prompts is None else num_prompts
        if count <= 0 or count > len(rows):
            raise ValueError(
                f"--num-prompts must be in 1..{len(rows)} for length {length}"
            )
        selected = rows[:count]
        if any(int(row["context_length"]) != length for row in selected):
            raise ValueError(f"{path} contains a wrong context_length")
        cases[length] = {
            "path": path,
            "sha256": entry["sha256"],
            "rows": selected,
            "requests": count,
            "total_input_tokens": sum(int(row["prompt_tokens"]) for row in selected),
            "total_output_tokens": sum(int(row["output_tokens"]) for row in selected),
            "request_identity_sha256": canonical_sha256(
                [
                    {
                        "prompt_sha256": row.get("prompt_sha256"),
                        "prompt_tokens": int(row["prompt_tokens"]),
                        "output_tokens": int(row["output_tokens"]),
                    }
                    for row in selected
                ]
            ),
        }
    return manifest, cases


def check_runtime() -> dict[str, Any]:
    versions = {}
    for name in ("vllm", "torch", "transformers", "triton", "bts"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if versions["vllm"] != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"this adapter requires vLLM {SUPPORTED_VLLM_VERSION}; "
            f"found {versions['vllm']!r} in {sys.executable}"
        )
    entries = importlib.metadata.entry_points(group="vllm.general_plugins")
    matches = [entry for entry in entries if entry.name == PLUGIN_ENTRY_POINT]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one vllm.general_plugins entry point named "
            f"{PLUGIN_ENTRY_POINT!r}; reinstall this submission into {sys.executable}"
        )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": versions,
        "plugin_entry_point": {
            "name": matches[0].name,
            "value": matches[0].value,
        },
    }


def model_contract(model: str, max_length: int, trust_remote_code: bool) -> dict[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    config_dict = config.to_dict()
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    values = {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", -1)),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
        "head_dim": int(head_dim),
        "max_position_embeddings": int(
            getattr(config, "max_position_embeddings", -1)
        ),
        "torch_dtype": str(
            config_dict.get("dtype", config_dict.get("torch_dtype"))
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
        "max_position_embeddings": 131072,
    }
    failed = [name for name, expected_value in expected.items() if values[name] != expected_value]
    rope = values["rope_scaling"]
    if not isinstance(rope, dict):
        failed.append("rope_scaling")
    else:
        if rope.get("rope_type", rope.get("type")) != "yarn":
            failed.append("rope_scaling.rope_type")
        if float(rope.get("factor", -1)) != 4.0:
            failed.append("rope_scaling.factor")
        if int(rope.get("original_max_position_embeddings", -1)) != 32768:
            failed.append("rope_scaling.original_max_position_embeddings")
    if max_length > values["max_position_embeddings"]:
        failed.append("requested_max_length")
    if values["torch_dtype"] not in {"bfloat16", "torch.bfloat16"}:
        failed.append("torch_dtype")
    if failed:
        raise RuntimeError(
            "vLLM BTS benchmark requires Qwen3-8B BF16 long-context config; "
            "mismatched fields: " + ", ".join(failed)
        )
    return {
        "requested_model": model,
        **values,
        "config_sha256": canonical_sha256(config_dict),
        "huggingface_commit": getattr(config, "_commit_hash", None),
    }


def formal_eligibility_failures(
    args: argparse.Namespace,
    workload_manifest: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> list[str]:
    """Return every deviation from the frozen 1.5x acceptance protocol."""

    failures = []
    if args.num_prompts is not None:
        failures.append("formal_protocol_requires_the_complete_workload")
    if tuple(workload_manifest.get("tasks", ())) != EXPECTED_RULER_TASKS:
        failures.append("formal_protocol_requires_all_13_ruler_tasks")
    if int(workload_manifest.get("samples_per_task", -1)) != FORMAL_SAMPLES_PER_TASK:
        failures.append(
            f"formal_protocol_requires_{FORMAL_SAMPLES_PER_TASK}_samples_per_task"
        )
    if int(workload_manifest.get("seed", -1)) != FORMAL_SEED or args.seed != FORMAL_SEED:
        failures.append(f"formal_protocol_requires_seed_{FORMAL_SEED}")
    if workload_manifest.get("fixed_output_tokens") is not None:
        failures.append("formal_protocol_requires_task_native_output_budgets")
    if args.num_warmups != 1:
        failures.append("formal_protocol_requires_one_client_warmup")
    if not math.isclose(args.target_speedup, 1.5, rel_tol=0.0, abs_tol=1e-12):
        failures.append("formal_protocol_requires_target_speedup_1.5")
    if not math.isclose(args.confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
        failures.append("formal_protocol_requires_one_sided_confidence_0.95")
    if args.bootstrap_samples < 10000:
        failures.append("formal_protocol_requires_at_least_10000_bootstrap_samples")
    if args.bootstrap_seed != FORMAL_BOOTSTRAP_SEED:
        failures.append(
            f"formal_protocol_requires_bootstrap_seed_{FORMAL_BOOTSTRAP_SEED}"
        )
    if args.max_num_seqs != 16:
        failures.append("formal_protocol_requires_max_num_seqs_16")
    if args.max_num_batched_tokens != max(DEFAULT_LENGTHS):
        failures.append("formal_protocol_requires_max_num_batched_tokens_131072")
    if args.max_concurrency is not None:
        failures.append("formal_protocol_requires_unbounded_client_concurrency")
    if not math.isclose(
        args.gpu_memory_utilization, 0.90, rel_tol=0.0, abs_tol=1e-12
    ):
        failures.append("formal_protocol_requires_gpu_memory_utilization_0.90")
    if args.pytorch_alloc_conf != FORMAL_PYTORCH_ALLOC_CONF:
        failures.append(
            "formal_protocol_requires_pytorch_expandable_segments_allocator"
        )
    if (
        args.server_extra_args
        or args.dense_server_extra_args
        or args.bts_server_extra_args
        or args.client_extra_args
    ):
        failures.append("formal_protocol_disallows_extra_cli_arguments")
    if args.trust_remote_code:
        failures.append("formal_protocol_disallows_trust_remote_code")
    if args.model is None or not Path(args.model).expanduser().is_dir():
        failures.append("formal_protocol_requires_a_local_frozen_model_directory")
    if "H100" not in str(gpu.get("name", "")):
        failures.append("formal_protocol_requires_an_H100_GPU")
    if str(gpu.get("compute_capability")) != "9.0":
        failures.append("formal_protocol_requires_compute_capability_9.0")
    return failures


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    rows = []
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) != 5:
            continue
        index, uuid, name, memory, compute_capability = (
            field.strip() for field in fields
        )
        rows.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(memory),
                "compute_capability": compute_capability,
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return rows


def select_gpu() -> dict[str, Any]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
    if len(selectors) != 1:
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU index or UUID"
        )
    selector = selectors[0]
    inventory = gpu_inventory()
    if selector.isdigit():
        matches = [row for row in inventory if row["physical_index"] == int(selector)]
    elif selector.startswith("GPU-"):
        matches = [row for row in inventory if str(row["uuid"]).startswith(selector)]
    else:
        raise RuntimeError("MIG and ambiguous CUDA_VISIBLE_DEVICES selectors are unsupported")
    if len(matches) != 1:
        raise RuntimeError(f"could not resolve CUDA_VISIBLE_DEVICES={raw!r}")
    return {**matches[0], "cuda_visible_devices": raw}


def gpu_processes(gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--id={gpu['physical_index']}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    rows = []
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) != 3:
            continue
        pid, name, memory = (field.strip() for field in fields)
        if pid.isdigit():
            rows.append(
                {
                    "pid": int(pid),
                    "process_name": name,
                    "used_gpu_memory_mib": int(memory) if memory.isdigit() else memory,
                }
            )
    return rows


def descendant_pids(root_pid: int) -> set[int]:
    """Return a read-only snapshot of every descendant of ``root_pid``."""

    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        try:
            pid, parent = (int(value) for value in line.split())
        except (TypeError, ValueError):
            continue
        children.setdefault(parent, []).append(pid)
    descendants: set[int] = set()
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        stack.extend(children.get(pid, ()))
    return descendants


def pids_with_environment_token(token: str) -> set[int]:
    expected = f"{RUN_TOKEN_ENV}={token}".encode("utf-8")
    result = set()
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            values = path.read_bytes().split(b"\0")
        except (FileNotFoundError, OSError, PermissionError, ProcessLookupError):
            continue
        if expected in values:
            result.add(int(path.parent.name))
    return result


def run_client_with_gpu_monitor(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log: Any,
    timeout_seconds: float,
    server_process: subprocess.Popen[Any],
    gpu: Mapping[str, Any],
    run_token: str,
) -> tuple[int, dict[str, Any]]:
    """Run one client and sample GPU exclusivity without touching outsiders."""

    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(environment),
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout_seconds
    check_count = 0
    outsider_rechecks = 0
    owned_gpu_pids: set[int] = set()
    last_checked_at: str | None = None

    def check_exclusive() -> None:
        nonlocal check_count, outsider_rechecks, last_checked_at
        if server_process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited during benchmark with {server_process.returncode}"
            )
        allowed = {
            server_process.pid,
            process.pid,
            *descendant_pids(server_process.pid),
            *descendant_pids(process.pid),
            *pids_with_environment_token(run_token),
        }
        observed = gpu_processes(gpu)
        check_count += 1
        last_checked_at = utc_now()
        outsiders = [row for row in observed if int(row["pid"]) not in allowed]
        if outsiders:
            outsider_rechecks += 1
            time.sleep(0.25)
            allowed.update(descendant_pids(server_process.pid))
            allowed.update(descendant_pids(process.pid))
            allowed.update(pids_with_environment_token(run_token))
            observed = gpu_processes(gpu)
            check_count += 1
            last_checked_at = utc_now()
            outsiders = [row for row in observed if int(row["pid"]) not in allowed]
        if outsiders:
            raise RuntimeError(
                "selected GPU lost exclusivity during the benchmark; external "
                f"compute processes={outsiders}"
            )
        owned_gpu_pids.update(
            int(row["pid"]) for row in observed if int(row["pid"]) in allowed
        )

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            check_exclusive()
            try:
                returncode = process.wait(timeout=min(1.0, remaining))
            except subprocess.TimeoutExpired:
                continue
            check_exclusive()
            return returncode, {
                "mode": "sampled_compute_process_monitor",
                "poll_interval_seconds": 1.0,
                "check_count": check_count,
                "outsider_rechecks": outsider_rechecks,
                "all_sampled_checks_exclusive": True,
                "last_checked_at": last_checked_at,
                "benchmark_gpu_pids_observed": sorted(owned_gpu_pids),
            }
    except BaseException:
        stop_process_group(process, timeout=10.0)
        raise


def require_idle_gpu(gpu: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    processes = gpu_processes(gpu)
    record = {"at": utc_now(), "label": label, "processes": processes}
    if processes:
        raise RuntimeError(f"selected GPU is occupied at {label}: {processes}")
    return record


def choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(process: subprocess.Popen[Any], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited during startup with {process.returncode}")
        try:
            with urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, URLError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(f"vLLM server was not ready after {timeout}s: {last_error}")


def stop_process_group(process: subprocess.Popen[Any], timeout: float = 60.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=30)


def source_manifest() -> dict[str, Any]:
    roots = (
        SUBMISSION_ROOT / "bts",
        SUBMISSION_ROOT / "integrations" / "vllm",
        HERE,
    )
    files = {
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".json", ".sh"}
    }
    files.add(SUBMISSION_ROOT / "pyproject.toml")
    entries = {
        str(path.relative_to(SUBMISSION_ROOT)): file_sha256(path) for path in sorted(files)
    }
    return {"files": entries, "manifest_sha256": canonical_sha256(entries)}


def build_server_command(args: argparse.Namespace, method: str, port: int) -> list[str]:
    backend = "FLASH_ATTN" if method == "dense" else "CUSTOM"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        # vLLM 0.17's reshape_and_cache_flash dispatch expects the selector
        # "auto" for an unquantized cache.  With this BF16 model the resolved
        # KV storage dtype is still torch.bfloat16; passing the literal
        # "bfloat16" reaches the same storage allocation but crashes the C++
        # cache-update op during engine warmup.
        "auto",
        "--max-model-len",
        str(max(args.lengths)),
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--attention-backend",
        backend,
        "--attention-config",
        json.dumps(
            {"flash_attn_version": FORMAL_FLASH_ATTN_VERSION},
            separators=(",", ":"),
        ),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--generation-config",
        "vllm",
        "--disable-log-stats",
        "--disable-uvicorn-access-log",
        "--uvicorn-log-level",
        "warning",
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(args.server_extra_args)
    command.extend(args.dense_server_extra_args if method == "dense" else args.bts_server_extra_args)
    return command


def validate_resolved_flash_attention(log_path: Path) -> dict[str, Any]:
    """Require the model worker to report the frozen FA2 implementation."""
    versions = [
        int(value)
        for value in re.findall(
            r"Using FlashAttention version\s+(\d+)",
            log_path.read_text(encoding="utf-8", errors="replace"),
        )
    ]
    if not versions:
        raise RuntimeError("vLLM worker did not report its resolved FlashAttention version")
    if set(versions) != {FORMAL_FLASH_ATTN_VERSION}:
        raise RuntimeError(
            "vLLM worker did not use the frozen FlashAttention 2 contract: "
            f"observed={versions}"
        )
    return {
        "requested_version": FORMAL_FLASH_ATTN_VERSION,
        "observed_versions": versions,
        "dense_kernel": "vllm_flash_attention_2",
        "decode_fallback": "native_vllm_flash_attention_2",
    }


def build_server_environment(
    args: argparse.Namespace,
    *,
    method: str,
    run_token: str,
    audit_dir: Path,
) -> dict[str, str]:
    """Build the identical allocator/runtime environment for both arms."""

    environment = os.environ.copy()
    environment[RUN_TOKEN_ENV] = run_token
    environment[PYTORCH_ALLOC_CONF_ENV] = args.pytorch_alloc_conf
    # torch 2.10 accepts both names and gives the legacy CUDA-specific name
    # precedence when it is inherited.  Pin both to prevent a parent-shell
    # setting from silently disabling the frozen allocator contract.
    environment[PYTORCH_CUDA_ALLOC_CONF_ENV] = args.pytorch_alloc_conf
    environment["VLLM_USE_V1"] = "1"
    environment["VLLM_PLUGINS"] = (
        "__bts_vllm_disabled__" if method == "dense" else PLUGIN_ENTRY_POINT
    )
    if method == "bts":
        environment["BTS_VLLM_AUDIT_DIR"] = str(audit_dir)
    else:
        environment.pop("BTS_VLLM_AUDIT_DIR", None)
    return environment


def build_client_command(
    args: argparse.Namespace,
    *,
    method: str,
    restart: int,
    length: int,
    rate: float,
    port: int,
    workload: Mapping[str, Any],
    slo: Mapping[str, float],
    result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(workload["path"]),
        "--custom-output-len",
        "-1",
        "--num-prompts",
        str(workload["requests"]),
        "--num-warmups",
        str(args.num_warmups),
        "--ready-check-timeout-sec",
        "0",
        "--request-rate",
        rate_label(rate),
        "--seed",
        str(args.seed),
        "--temperature",
        "0",
        "--skip-chat-template",
        "--ignore-eos",
        "--disable-shuffle",
        "--no-oversample",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,95,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
        "--label",
        method,
        "--metadata",
        f"protocol={PROTOCOL}",
        f"method={method}",
        f"restart={restart}",
        f"context_length={length}",
        f"workload_sha256={workload['sha256']}",
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.max_concurrency is not None:
        command.extend(("--max-concurrency", str(args.max_concurrency)))
    if slo:
        command.append("--goodput")
        command.extend(f"{name}:{value}" for name, value in sorted(slo.items()))
    command.extend(args.client_extra_args)
    return command


def expected_benchmark_prompt_requests(
    args: argparse.Namespace, workloads: Mapping[int, Mapping[str, Any]]
) -> int:
    """Count requests that reach the server after the audit baseline."""

    return sum(
        len(args.request_rates)
        * (int(workloads[length]["requests"]) + args.num_warmups)
        for length in args.lengths
    )


def validate_benchmark_result(
    result: Mapping[str, Any], workload: Mapping[str, Any]
) -> dict[str, Any]:
    requests = int(workload["requests"])
    if int(result.get("completed", -1)) != requests or int(result.get("failed", 0)) != 0:
        raise RuntimeError(
            f"benchmark completed/failed mismatch: {result.get('completed')}/"
            f"{result.get('failed')} for {requests} requests"
        )
    expected_inputs = [int(row["prompt_tokens"]) for row in workload["rows"]]
    expected_outputs = [int(row["output_tokens"]) for row in workload["rows"]]
    observed_inputs = [int(value) for value in result.get("input_lens", [])]
    observed_outputs = [int(value) for value in result.get("output_lens", [])]
    if observed_inputs != expected_inputs:
        raise RuntimeError("vLLM client tokenized prompts differently from the workload manifest")
    if observed_outputs != expected_outputs:
        raise RuntimeError("vLLM did not generate every frozen output-token budget")
    if any(result.get("errors", [])):
        raise RuntimeError("vLLM client reported request errors")
    for field in ("duration", "request_throughput", "total_token_throughput"):
        value = float(result.get(field, 0))
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"invalid benchmark field {field}={value!r}")
    return dict(result)


def read_audit_baseline(directory: Path, expected_layers: int) -> list[dict[str, Any]]:
    """Capture post-startup counters before benchmark requests are submitted."""

    from bts_vllm.audit import read_audits

    audits = [dict(audit) for audit in read_audits(directory)]
    if not audits or any(audit.get("backend_mode") != "bts" for audit in audits):
        raise RuntimeError("BTS plugin did not produce valid startup audit records")
    impl_count = sum(int(audit.get("impl_count", 0)) for audit in audits)
    model_workers = [audit for audit in audits if int(audit.get("impl_count", 0))]
    if impl_count != expected_layers or len(model_workers) != 1:
        raise RuntimeError(
            "BTS backend startup audit must contain one TP=1 model worker with "
            f"{expected_layers} layer implementations; found "
            f"{len(model_workers)} worker(s) and {impl_count} implementations"
        )
    return audits


def read_and_validate_audit(
    directory: Path,
    *,
    baseline: Sequence[Mapping[str, Any]],
    expected_layers: int,
    expected_sparse_segments: int,
) -> dict[str, Any]:
    """Validate the measured audit delta, ignoring vLLM startup/profile work."""

    from bts_vllm.audit import COUNTER_NAMES, read_audits

    final = [dict(audit) for audit in read_audits(directory)]
    if not final or any(audit.get("backend_mode") != "bts" for audit in final):
        raise RuntimeError("BTS plugin did not produce valid final audit records")
    final_by_pid = {int(audit["pid"]): audit for audit in final}
    baseline_by_pid = {int(audit["pid"]): audit for audit in baseline}
    if not set(baseline_by_pid) <= set(final_by_pid):
        raise RuntimeError("a BTS audit process disappeared between startup and shutdown")

    totals = {
        name: sum(int(audit.get(name, 0)) for audit in final)
        for name in COUNTER_NAMES
    }
    baseline_totals = {
        name: sum(int(audit.get(name, 0)) for audit in baseline)
        for name in COUNTER_NAMES
    }
    delta = {name: totals[name] - baseline_totals[name] for name in COUNTER_NAMES}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("BTS audit counters decreased during the benchmark")
    model_workers = [audit for audit in final if int(audit.get("impl_count", 0))]
    if totals["impl_instances"] != expected_layers or len(model_workers) != 1:
        raise RuntimeError(
            "BTS final audit does not contain exactly one complete TP=1 model worker"
        )
    sparse_calls = delta["sparse_calls"]
    if sparse_calls < expected_layers or sparse_calls % expected_layers:
        raise RuntimeError(
            f"measured BTS sparse call count {sparse_calls} is not a positive "
            "layer multiple"
        )
    if delta["sparse_segments"] != expected_sparse_segments:
        raise RuntimeError(
            "BTS did not execute exactly one sparse full-prefill segment per "
            f"submitted request and layer: observed {delta['sparse_segments']}, "
            f"expected {expected_sparse_segments}"
        )
    if delta["sparse_segments"] < sparse_calls:
        raise RuntimeError("BTS audit sparse segment count is inconsistent")
    return {
        "baseline_records": list(baseline),
        "final_records": final,
        "totals": totals,
        "measured_delta": delta,
        "expected_sparse_segments": expected_sparse_segments,
        "model_worker_pids": [int(audit["pid"]) for audit in model_workers],
    }


def run_arm(
    args: argparse.Namespace,
    *,
    method: str,
    restart: int,
    output: Path,
    workloads: Mapping[int, Mapping[str, Any]],
    slo_config: Mapping[int, Mapping[str, float]],
    gpu: Mapping[str, Any],
    expected_layers: int,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    arm_dir = output / f"restart-{restart:02d}" / method
    arm_dir.mkdir(parents=True, exist_ok=False)
    audit_dir = arm_dir / "audit"
    port = choose_port()
    run_token = secrets.token_hex(16)
    server_command = build_server_command(args, method, port)
    environment = build_server_environment(
        args,
        method=method,
        run_token=run_token,
        audit_dir=audit_dir,
    )
    if method == "bts":
        audit_dir.mkdir(parents=True, exist_ok=False)

    idle = require_idle_gpu(gpu, label=f"restart-{restart}-{method}-before_server")
    source_before = source_manifest()["manifest_sha256"]
    if source_before != expected_source_manifest_sha256:
        raise RuntimeError("runtime source changed before starting an arm")
    manifest = {
        "protocol": PROTOCOL,
        "status": "starting",
        "method": method,
        "restart": restart,
        "port": port,
        "server_command": server_command,
        "environment": {
            "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
            "VLLM_USE_V1": environment["VLLM_USE_V1"],
            "VLLM_PLUGINS": environment["VLLM_PLUGINS"],
            PYTORCH_ALLOC_CONF_ENV: environment[PYTORCH_ALLOC_CONF_ENV],
            PYTORCH_CUDA_ALLOC_CONF_ENV: environment[
                PYTORCH_CUDA_ALLOC_CONF_ENV
            ],
            "BTS_VLLM_AUDIT_DIR": environment.get("BTS_VLLM_AUDIT_DIR"),
            RUN_TOKEN_ENV: run_token,
        },
        "gpu_idle_check": idle,
        "source_manifest_sha256_before": source_before,
        "started_at": utc_now(),
    }
    atomic_write_json(arm_dir / "manifest.json", manifest)

    log_path = arm_dir / "server.log"
    process: subprocess.Popen[Any] | None = None
    audit_baseline: list[dict[str, Any]] = []
    points = []
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                server_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
                start_new_session=True,
            )
        wait_for_server(process, port, args.server_startup_timeout_seconds)
        resolved_flash_attention = validate_resolved_flash_attention(log_path)
        if method == "bts":
            audit_baseline = read_audit_baseline(audit_dir, expected_layers)
        manifest["status"] = "running"
        manifest["server_pid"] = process.pid
        manifest["ready_at"] = utc_now()
        manifest["audit_baseline"] = audit_baseline if method == "bts" else None
        manifest["resolved_flash_attention"] = resolved_flash_attention
        atomic_write_json(arm_dir / "manifest.json", manifest)

        length_order = list(args.lengths)
        if restart % 2:
            length_order.reverse()
        for length in length_order:
            workload = workloads[length]
            for rate in args.request_rates:
                label = rate_label(rate).replace(".", "p")
                point_dir = arm_dir / f"length-{length}" / f"rate-{label}"
                point_dir.mkdir(parents=True, exist_ok=False)
                result_path = point_dir / "result.json"
                command = build_client_command(
                    args,
                    method=method,
                    restart=restart,
                    length=length,
                    rate=rate,
                    port=port,
                    workload=workload,
                    slo=slo_config.get(length, {}),
                    result_path=result_path,
                )
                atomic_write_json(point_dir / "command.json", {"command": command})
                client_log = point_dir / "client.log"
                client_environment = dict(environment)
                # Only the server/model worker needs the attention plugin.
                # Keeping the benchmark client out of the audit directory
                # prevents activation-only client records from obscuring the
                # measured model-worker delta.
                client_environment["VLLM_PLUGINS"] = "__bts_vllm_disabled__"
                client_environment.pop("BTS_VLLM_AUDIT_DIR", None)
                with client_log.open("w", encoding="utf-8") as log:
                    returncode, gpu_monitor = run_client_with_gpu_monitor(
                        command,
                        environment=client_environment,
                        log=log,
                        timeout_seconds=args.client_timeout_seconds,
                        server_process=process,
                        gpu=gpu,
                        run_token=run_token,
                    )
                atomic_write_json(point_dir / "gpu_monitor.json", gpu_monitor)
                if returncode:
                    raise RuntimeError(
                        f"vLLM bench failed for {method}/{length}/{rate_label(rate)} "
                        f"with exit {returncode}"
                    )
                result = validate_benchmark_result(
                    json.loads(result_path.read_text(encoding="utf-8")), workload
                )
                points.append(
                    {
                        "context_length": length,
                        "request_rate": rate_label(rate),
                        "result_path": str(result_path.relative_to(output)),
                        "result_sha256": file_sha256(result_path),
                        "gpu_monitor": gpu_monitor,
                        "completed": int(result["completed"]),
                        "duration": float(result["duration"]),
                        "request_throughput": float(result["request_throughput"]),
                        "request_goodput": (
                            float(result["request_goodput"])
                            if result.get("request_goodput") is not None
                            else None
                        ),
                        "total_token_throughput": float(
                            result["total_token_throughput"]
                        ),
                        "total_input_tokens": int(result["total_input_tokens"]),
                        "total_output_tokens": int(result["total_output_tokens"]),
                        "p50_ttft_ms": result.get("p50_ttft_ms"),
                        "p95_ttft_ms": result.get("p95_ttft_ms"),
                        "p99_ttft_ms": result.get("p99_ttft_ms"),
                        "p50_tpot_ms": result.get("p50_tpot_ms"),
                        "p95_tpot_ms": result.get("p95_tpot_ms"),
                        "p99_tpot_ms": result.get("p99_tpot_ms"),
                        "p95_e2el_ms": result.get("p95_e2el_ms"),
                        "p99_e2el_ms": result.get("p99_e2el_ms"),
                    }
                )
    except BaseException as error:
        manifest.update(
            status="failed",
            failure_type=type(error).__name__,
            failure=str(error),
            failed_at=utc_now(),
        )
        atomic_write_json(arm_dir / "manifest.json", manifest)
        raise
    finally:
        if process is not None:
            stop_process_group(process)

    source_after = source_manifest()["manifest_sha256"]
    if source_after != expected_source_manifest_sha256:
        manifest.update(
            status="failed",
            failure_type="RuntimeError",
            failure="runtime source changed while an arm was running",
            failed_at=utc_now(),
            source_manifest_sha256_after=source_after,
        )
        atomic_write_json(arm_dir / "manifest.json", manifest)
        raise RuntimeError("runtime source changed while an arm was running")

    # vLLM bench is invoked with the endpoint ready check disabled, so the
    # server receives only the explicit warmups plus the measured prompts.
    # Keep this count aligned with build_client_command(); the client log's
    # "initial single prompt" message does not imply a request when the ready
    # check timeout is zero.
    expected_prompt_requests = expected_benchmark_prompt_requests(args, workloads)
    audit = (
        read_and_validate_audit(
            audit_dir,
            baseline=audit_baseline,
            expected_layers=expected_layers,
            expected_sparse_segments=expected_layers * expected_prompt_requests,
        )
        if method == "bts"
        else None
    )
    report = {
        "method": method,
        "restart": restart,
        "points": sorted(
            points,
            key=lambda point: (
                int(point["context_length"]),
                math.inf
                if point["request_rate"] == "inf"
                else float(point["request_rate"]),
            ),
        ),
        "audit": audit,
        "resolved_flash_attention": resolved_flash_attention,
        "source_manifest_sha256_after": source_after,
    }
    atomic_write_json(arm_dir / "report.json", report)
    manifest.update(status="completed", completed_at=utc_now())
    atomic_write_json(arm_dir / "manifest.json", manifest)
    return report


def select_capacity(
    points: Sequence[Mapping[str, Any]], *, use_goodput: bool
) -> dict[str, Any]:
    field = "request_goodput" if use_goodput else "request_throughput"
    candidates = []
    for point in points:
        value = point.get(field)
        if value is None:
            raise RuntimeError(f"point lacks required {field}")
        candidates.append((float(value), point))
    capacity, selected = max(candidates, key=lambda item: item[0])
    rates = [point["request_rate"] for point in points]
    if use_goodput:
        highest = points[-1]
        if highest["request_rate"] == "inf":
            grid_covered = True
        else:
            offered = float(highest["request_rate"])
            # The last point must be visibly overloaded or above the observed
            # capacity by 10%; otherwise the finite capacity grid is unbounded.
            grid_covered = (
                offered >= 1.10 * capacity
                and float(highest[field]) <= 0.95 * offered
            )
    else:
        grid_covered = "inf" in rates
    return {
        "metric": field,
        "capacity_requests_per_second": capacity,
        "selected_request_rate": selected["request_rate"],
        "grid_covered": grid_covered,
    }


def build_comparison(
    paired_reports: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    lengths: Sequence[int],
    target_speedup: float,
    use_goodput: bool,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    formal_eligibility_failures: Sequence[str] = (),
) -> dict[str, Any]:
    speedups: dict[int, list[float]] = {length: [] for length in lengths}
    length_rows: dict[int, list[dict[str, Any]]] = {length: [] for length in lengths}
    all_grids_covered = True
    for restart, reports in enumerate(paired_reports):
        if set(reports) != set(METHODS):
            raise RuntimeError("each restart must contain dense and BTS reports")
        points_by_method = {
            method: {
                length: [
                    point
                    for point in reports[method]["points"]
                    if int(point["context_length"]) == length
                ]
                for length in lengths
            }
            for method in METHODS
        }
        for length in lengths:
            dense_points = points_by_method["dense"][length]
            bts_points = points_by_method["bts"][length]
            if [point["request_rate"] for point in dense_points] != [
                point["request_rate"] for point in bts_points
            ]:
                raise RuntimeError("dense and BTS request-rate grids differ")
            for dense, bts in zip(dense_points, bts_points, strict=True):
                for field in (
                    "completed",
                    "total_input_tokens",
                    "total_output_tokens",
                ):
                    if dense[field] != bts[field]:
                        raise RuntimeError(
                            f"dense/BTS workload field {field} differs at {length}"
                        )
            dense_capacity = select_capacity(dense_points, use_goodput=use_goodput)
            bts_capacity = select_capacity(bts_points, use_goodput=use_goodput)
            speedup = (
                bts_capacity["capacity_requests_per_second"]
                / dense_capacity["capacity_requests_per_second"]
            )
            speedups[length].append(speedup)
            all_grids_covered &= bool(
                dense_capacity["grid_covered"] and bts_capacity["grid_covered"]
            )
            length_rows[length].append(
                {
                    "restart": restart,
                    "dense": dense_capacity,
                    "bts": bts_capacity,
                    "throughput_speedup": speedup,
                }
            )

    lower_bounds = (
        paired_bootstrap_lower_bounds(
            speedups,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        if len(paired_reports) >= 2
        else {length: None for length in lengths}
    )
    formal_scale = (
        len(paired_reports) >= MIN_FORMAL_RESTARTS
        and tuple(lengths) == DEFAULT_LENGTHS
        and all_grids_covered
        and not formal_eligibility_failures
    )
    summaries = {}
    for length in lengths:
        lower = lower_bounds[length]
        summaries[str(length)] = {
            "restart_pairs": length_rows[length],
            "paired_speedups": speedups[length],
            "median_speedup": statistics.median(speedups[length]),
            "one_sided_confidence": confidence,
            "lower_bound": lower,
            "target_speedup": target_speedup,
            "passes_target": (
                bool(lower is not None and lower >= target_speedup)
                if formal_scale
                else None
            ),
        }
    if not formal_scale:
        status = "INCONCLUSIVE"
    elif all(summary["passes_target"] for summary in summaries.values()):
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "analysis_unit": "paired_fresh_vllm_server_restart",
        "capacity_metric": "request_goodput" if use_goodput else "request_throughput",
        "statistics": {
            "restart_statistic": "median_of_paired_capacity_ratios",
            "bootstrap_type": "paired_cluster_percentile",
            "shared_restart_indices_across_lengths": True,
            "replicates": bootstrap_samples,
            "seed": bootstrap_seed,
            "one_sided_confidence": confidence,
            "quantile_method": "lower_floor_q_times_n_minus_1",
        },
        "minimum_formal_restarts": MIN_FORMAL_RESTARTS,
        "restart_pairs": len(paired_reports),
        "common_rate_grid_covered": all_grids_covered,
        "formal_scale": formal_scale,
        "formal_eligibility_failures": list(formal_eligibility_failures),
        "lengths": summaries,
        "overall_status": status,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", help="Local Qwen3-8B YaRN4 model path")
    result.add_argument("--workload", type=Path, help="Workload manifest or directory")
    result.add_argument("--output", type=Path)
    result.add_argument("--lengths", type=parse_lengths, default=DEFAULT_LENGTHS)
    result.add_argument("--num-prompts", type=int)
    result.add_argument("--request-rates", type=parse_rates, default=(math.inf,))
    result.add_argument(
        "--slo-config",
        type=Path,
        help='JSON mapping each length to TTFT/TPOT/E2EL limits, e.g. {"32768":{"ttft":5000}}',
    )
    result.add_argument("--restarts", type=int, default=5)
    result.add_argument("--num-warmups", type=int, default=1)
    result.add_argument("--target-speedup", type=float, default=1.5)
    result.add_argument("--confidence", type=float, default=0.95)
    result.add_argument("--bootstrap-samples", type=int, default=10000)
    result.add_argument("--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED)
    result.add_argument("--seed", type=int, default=FORMAL_SEED)
    result.add_argument("--max-num-seqs", type=int, default=16)
    result.add_argument("--max-num-batched-tokens", type=int)
    result.add_argument("--max-concurrency", type=int)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument(
        "--pytorch-alloc-conf",
        default=FORMAL_PYTORCH_ALLOC_CONF,
        help=(
            "PyTorch CUDA allocator configuration applied identically to both "
            "servers; the formal protocol requires expandable_segments:True"
        ),
    )
    result.add_argument("--served-model-name", default="bts-vllm-eval")
    result.add_argument("--server-startup-timeout-seconds", type=float, default=900)
    result.add_argument("--client-timeout-seconds", type=float, default=14400)
    result.add_argument("--trust-remote-code", action="store_true")
    result.add_argument("--server-extra-args", type=parse_extra_args, default=())
    result.add_argument("--dense-server-extra-args", type=parse_extra_args, default=())
    result.add_argument("--bts-server-extra-args", type=parse_extra_args, default=())
    result.add_argument("--client-extra-args", type=parse_extra_args, default=())
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.restarts <= 0 or args.num_warmups < 0:
        raise SystemExit("--restarts must be positive and --num-warmups nonnegative")
    if args.num_prompts is not None and args.num_prompts <= 0:
        raise SystemExit("--num-prompts must be positive")
    if args.max_num_seqs <= 0:
        raise SystemExit("--max-num-seqs must be positive")
    if not (0 < args.gpu_memory_utilization < 1):
        raise SystemExit("--gpu-memory-utilization must be in (0, 1)")
    if not args.pytorch_alloc_conf.strip():
        raise SystemExit("--pytorch-alloc-conf must be nonempty")
    if not (0.5 < args.confidence < 1) or args.bootstrap_samples <= 0:
        raise SystemExit("invalid confidence/bootstrap sample count")
    if args.target_speedup <= 0:
        raise SystemExit("--target-speedup must be positive")
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = max(args.lengths)
    if args.max_num_batched_tokens < max(args.lengths):
        raise SystemExit(
            "--max-num-batched-tokens must cover the longest full prefill when "
            "chunked prefill is disabled"
        )
    if args.slo_config is None and args.request_rates != (math.inf,):
        raise SystemExit(
            "without --slo-config, the formal saturated-throughput protocol "
            "requires --request-rates inf"
        )
    slo_config = load_slo_config(args.slo_config, args.lengths)

    plan = {
        "protocol": PROTOCOL,
        "scope": (
            "same-vLLM dense-vs-BTS serving throughput; attention/system effect, "
            "not standalone scheduler attribution"
        ),
        "model": args.model,
        "workload": str(args.workload) if args.workload else None,
        "output": str(args.output) if args.output else None,
        "lengths": list(args.lengths),
        "request_rates": [rate_label(rate) for rate in args.request_rates],
        "slo_config": slo_config,
        "restarts": args.restarts,
        "method_order_by_restart": [
            list(METHODS if restart % 2 == 0 else reversed(METHODS))
            for restart in range(args.restarts)
        ],
        "server_contract": {
            "vllm_version": SUPPORTED_VLLM_VERSION,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "dtype": "bfloat16",
            "kv_cache_dtype_selector": "auto",
            "kv_cache_storage_dtype": "bfloat16",
            "prefix_caching": False,
            "chunked_prefill": False,
            "flash_attention_version": FORMAL_FLASH_ATTN_VERSION,
            "enforce_eager": True,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "pytorch_alloc_conf": args.pytorch_alloc_conf,
        },
        "acceptance": {
            "target_speedup": args.target_speedup,
            "one_sided_confidence": args.confidence,
            "minimum_paired_restarts": MIN_FORMAL_RESTARTS,
            "all_default_lengths_must_pass": True,
        },
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not args.model or args.workload is None or args.output is None:
        raise SystemExit("--model, --workload and --output are required for a real run")
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    runtime = check_runtime()
    model = model_contract(args.model, max(args.lengths), args.trust_remote_code)
    workload_manifest, workloads = load_workload(
        args.workload, args.lengths, args.num_prompts
    )
    if workload_manifest["model"]["config_sha256"] != model["config_sha256"]:
        raise RuntimeError("workload and server model config hashes differ")
    gpu = select_gpu()
    eligibility_failures = formal_eligibility_failures(
        args, workload_manifest, gpu
    )
    sources = source_manifest()
    run_manifest = {
        **plan,
        "status": "running",
        "started_at": utc_now(),
        "runtime": runtime,
        "model_contract": model,
        "workload_manifest_sha256": workload_manifest["manifest_sha256"],
        "workload_cases": {
            str(length): {
                key: value
                for key, value in case.items()
                if key not in {"rows", "path"}
            }
            | {"path": str(case["path"])}
            for length, case in workloads.items()
        },
        "gpu": gpu,
        "formal_eligibility_failures": eligibility_failures,
        "source_manifest": sources,
    }
    run_manifest["run_identity_sha256"] = canonical_sha256(run_manifest)
    atomic_write_json(output / "manifest.json", run_manifest)

    pairs = []
    try:
        for restart in range(args.restarts):
            reports = {}
            order = METHODS if restart % 2 == 0 else tuple(reversed(METHODS))
            for method in order:
                reports[method] = run_arm(
                    args,
                    method=method,
                    restart=restart,
                    output=output,
                    workloads=workloads,
                    slo_config=slo_config,
                    gpu=gpu,
                    expected_layers=int(model["num_hidden_layers"]),
                    expected_source_manifest_sha256=sources["manifest_sha256"],
                )
            pairs.append(reports)
            atomic_write_json(
                output / "progress.json",
                {"completed_restart_pairs": len(pairs), "updated_at": utc_now()},
            )
        comparison = build_comparison(
            pairs,
            lengths=args.lengths,
            target_speedup=args.target_speedup,
            use_goodput=bool(slo_config),
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            formal_eligibility_failures=eligibility_failures,
        )
        report = {
            "protocol": PROTOCOL,
            "run_identity_sha256": run_manifest["run_identity_sha256"],
            "claim_scope": (
                "BTS sparse-attention system inside the stock vLLM scheduler "
                "versus stock vLLM dense attention"
            ),
            "scheduler_attribution": False,
            "comparison": comparison,
            "completed_at": utc_now(),
        }
        atomic_write_json(output / "benchmark_report.json", report)
        run_manifest.update(status="completed", completed_at=utc_now())
        atomic_write_json(output / "manifest.json", run_manifest)
    except BaseException as error:
        run_manifest.update(
            status="failed",
            failure_type=type(error).__name__,
            failure=str(error),
            failed_at=utc_now(),
        )
        atomic_write_json(output / "manifest.json", run_manifest)
        raise

    status = report["comparison"]["overall_status"]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return {"PASS": 0, "FAIL": 2, "INCONCLUSIVE": 3}[status]


if __name__ == "__main__":
    raise SystemExit(main())
