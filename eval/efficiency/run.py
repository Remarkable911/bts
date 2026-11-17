#!/usr/bin/env python3
"""Re-run the frozen attention-module efficiency experiment.

This runner compares two forwards of the *same* loaded Qwen3 attention module:

* dense: the original Transformers forward, configured for SDPA and executed
  while PyTorch permits only ``SDPBackend.FLASH_ATTENTION``;
* candidate: BTS (default), FlashAttention-2, FlexPrefill, MInference, or
  XAttention, using the same module object and immutable inputs.

Inputs are deterministic synthetic token-ID prefixes.  Their hidden states are
computed with the checkpoint's real embedding table and the selected layer's
input RMSNorm.  For the default layer zero this is the actual hidden-state
construction at the first attention-module boundary.  For a nonzero layer it
is deliberately labelled embedding-derived rather than a real activation of
that deeper layer.

The measured scope is one attention-module prefill forward (QKV projections,
Q/K normalization, RoPE, attention, and output projection; BTS additionally
includes all sparse selection and packing).  It is NOT an end-to-end model,
generation, request-throughput, or serving-system benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import MethodType
from typing import Any, Callable, Iterator, Sequence

import torch


HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
SUBMISSION_ROOT = EVAL_ROOT.parent
SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SUBMISSION_ROOT))

from baselines.registry import DENSITY_DEFINITIONS

METHODS = ("dense_flash", "bts")
EFFICIENCY_METHODS = (
    "bts",
    "flashattention",
    "flexprefill",
    "minference",
    "xattention",
)
METHOD_DISPLAY_NAME = "BTS"
DEFAULT_LENGTHS = (16384, 32768, 65536, 131072)
STANDARD_PROTOCOL = {
    "lengths": DEFAULT_LENGTHS,
    "layer": 0,
    "seed": 20261109,
    "warmup": 4,
    "rounds": 5,
    "samples": 9,
    "probe_tokens": 128,
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "compute_capability": (9, 0),
}
SOURCE_SUFFIXES = {".py", ".cu", ".cpp", ".h", ".cuh"}
PROTOCOL = "qwen3_synthetic_token_attention_module_efficiency_v1"
SCOPE = (
    "One Qwen3 attention-module prefill forward: QKV projections, Q/K norm, "
    "RoPE application, attention, and output projection; sparse candidates "
    "also include their selection, mask/index construction, and sparse kernel."
)
NON_CLAIM = (
    "Attention-module latency only; not full-model latency, generation speed, "
    "request throughput, scheduler throughput, or serving throughput."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace ``path`` with one complete JSON document."""
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
                default=str,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--lengths must be comma-separated integers") from error
    if not lengths:
        raise argparse.ArgumentTypeError("--lengths must contain at least one length")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("--lengths must not contain duplicates")
    if any(length <= 0 or length > 131072 for length in lengths):
        raise argparse.ArgumentTypeError("every length must be in [1, 131072]")
    return lengths


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def summarize(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "std": statistics.pstdev(samples),
        "min": min(samples),
        "p10": percentile(samples, 0.10),
        "p90": percentile(samples, 0.90),
        "max": max(samples),
        "samples": samples,
    }


def measurement_order(
    round_index: int,
    sample_index: int,
    candidate_name: str = "bts",
) -> tuple[str, str]:
    """Return a balanced AB/BA order without consuming an RNG."""
    methods = ("dense_flash", candidate_name)
    return methods if (round_index + sample_index) % 2 == 0 else tuple(reversed(methods))


def probe_indices(length: int, count: int) -> tuple[int, ...]:
    """Choose deterministic, evenly spread output-token positions."""
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if count >= length:
        return tuple(range(length))
    if count == 1:
        return (length - 1,)
    return tuple(
        sorted({(index * (length - 1)) // (count - 1) for index in range(count)})
    )


def _update_tensor_digest(digest: Any, tensor: torch.Tensor) -> None:
    native = tensor.detach().cpu().contiguous()
    digest.update(str(native.dtype).encode("ascii"))
    digest.update(json.dumps(list(native.shape), separators=(",", ":")).encode("ascii"))
    byte_view = native.view(torch.uint8).reshape(-1)
    chunk_bytes = 8 * 1024 * 1024
    for first in range(0, byte_view.numel(), chunk_bytes):
        chunk = byte_view[first : first + chunk_bytes].numpy()
        digest.update(memoryview(chunk))


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    _update_tensor_digest(digest, tensor)
    return digest.hexdigest()


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash the exact, materialized state used by this benchmark."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        _update_tensor_digest(digest, state[name])
        digest.update(b"\0")
    return digest.hexdigest()


def source_hashes(root: Path = SUBMISSION_ROOT) -> dict[str, Any]:
    files = [SCRIPT_PATH]
    for source_root in (root / "bts", root / "baselines"):
        files.extend(
            path
            for path in sorted(source_root.rglob("*"))
            if path.is_file() and path.suffix in SOURCE_SUFFIXES
        )
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in files}
    return {"files": hashes, "manifest_sha256": canonical_sha256(hashes)}


def numerical_comparison(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    """Compare small CPU probes; this is diagnostic, not an accuracy gate."""
    if reference.shape != actual.shape:
        raise ValueError("numerical probes must have the same shape")
    left = reference.detach().cpu().float().reshape(-1).double()
    right = actual.detach().cpu().float().reshape(-1).double()
    delta = right - left
    left_norm_sq = float(torch.dot(left, left))
    right_norm_sq = float(torch.dot(right, right))
    delta_norm_sq = float(torch.dot(delta, delta))
    denominator = math.sqrt(max(left_norm_sq * right_norm_sq, 0.0))
    return {
        "probe_shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "all_finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
        "relative_l2": math.sqrt(delta_norm_sq / max(left_norm_sq, 1e-30)),
        "cosine": float(torch.dot(left, right)) / max(denominator, 1e-30),
        "interpretation": (
            "Expected sparse-vs-dense approximation difference on sampled output "
            "tokens; reported for audit and not used as an accuracy acceptance gate."
        ),
    }


def parse_csv_rows(raw: str, fields: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",", len(fields) - 1)]
        if len(values) != len(fields):
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(dict(zip(fields, values)))
    return rows


def standard_protocol_audit(
    args: argparse.Namespace,
    *,
    model_metadata: dict[str, Any],
    compute_capability: Sequence[int],
) -> dict[str, Any]:
    """Distinguish a complete custom diagnostic from the formal protocol."""
    observed = {
        "lengths": tuple(args.lengths),
        "layer": args.layer,
        "seed": args.seed,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "samples": args.samples,
        "probe_tokens": args.probe_tokens,
        "model_type": model_metadata.get("model_type"),
        "hidden_size": model_metadata.get("hidden_size"),
        "num_hidden_layers": model_metadata.get("num_hidden_layers"),
        "num_attention_heads": model_metadata.get("num_attention_heads"),
        "num_key_value_heads": model_metadata.get("num_key_value_heads"),
        "head_dim": model_metadata.get("head_dim"),
        "compute_capability": tuple(int(item) for item in compute_capability),
    }
    checks = {
        name: observed[name] == expected
        for name, expected in STANDARD_PROTOCOL.items()
    }
    deviations = {
        name: {
            "expected": list(expected) if isinstance(expected, tuple) else expected,
            "observed": (
                list(observed[name])
                if isinstance(observed[name], tuple)
                else observed[name]
            ),
        }
        for name, expected in STANDARD_PROTOCOL.items()
        if not checks[name]
    }
    return {
        "name": "submission_attention_efficiency_formal_v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "deviations": deviations,
        "policy": (
            "Only the canonical Qwen3-8B layer-0, four-length, H100-SM90 "
            "protocol receives formal_valid=true. Successful custom/smoke runs "
            "remain complete but are explicitly non-formal."
        ),
    }


def _nvidia_smi(arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"nvidia-smi query failed: {error}") from error
    return completed.stdout.strip()


GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "memory.total",
    "compute_cap",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "power.limit",
)


def gpu_inventory() -> list[dict[str, str]]:
    raw = _nvidia_smi(
        [f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"]
    )
    return parse_csv_rows(raw, GPU_FIELDS)


def resolve_physical_gpu(
    logical_index: int,
    explicit_index: int | None,
    inventory: Sequence[dict[str, str]],
    cuda_visible_devices: str | None,
) -> int:
    """Resolve and validate CUDA logical-to-physical GPU mapping."""
    by_index = {int(row["index"]): row for row in inventory}
    if logical_index < 0:
        raise ValueError("CUDA logical device index must be nonnegative")
    if cuda_visible_devices is None:
        resolved = logical_index
    else:
        entries = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
        if logical_index >= len(entries):
            raise ValueError("logical CUDA device is outside CUDA_VISIBLE_DEVICES")
        selected = entries[logical_index]
        if selected.isdigit():
            resolved = int(selected)
        elif selected.startswith("GPU-"):
            matches = [
                index for index, row in by_index.items() if row["uuid"].startswith(selected)
            ]
            if len(matches) != 1:
                raise ValueError(f"could not uniquely resolve CUDA UUID {selected!r}")
            resolved = matches[0]
        else:
            raise ValueError(
                "MIG/nonstandard CUDA_VISIBLE_DEVICES entries are unsupported; "
                "use a full GPU UUID or numeric physical index"
            )
    if resolved not in by_index:
        raise ValueError(f"physical GPU {resolved} is absent from nvidia-smi")
    if explicit_index is not None and explicit_index != resolved:
        raise ValueError(
            f"logical cuda:{logical_index} maps to physical GPU {resolved}, "
            f"not --physical-gpu {explicit_index}"
        )
    return resolved


def own_process_ids() -> set[int]:
    identifiers = {os.getpid()}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("NSpid:"):
                identifiers.update(int(item) for item in line.split()[1:])
    return identifiers


def gpu_processes(gpu_uuid: str) -> list[dict[str, Any]]:
    fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    raw = _nvidia_smi(
        [f"--query-compute-apps={','.join(fields)}", "--format=csv,noheader,nounits"]
    )
    parsed = parse_csv_rows(raw, fields)
    rows: list[dict[str, Any]] = []
    for row in parsed:
        if row["gpu_uuid"] != gpu_uuid:
            continue
        converted: dict[str, Any] = dict(row)
        if row["pid"].isdigit():
            converted["pid"] = int(row["pid"])
        if row["used_gpu_memory"].isdigit():
            converted["used_gpu_memory_mib"] = int(row["used_gpu_memory"])
        rows.append(converted)
    return rows


def check_gpu_exclusive(
    checkpoints: list[dict[str, Any]],
    *,
    gpu_uuid: str,
    stage: str,
    allow_busy_gpu: bool,
) -> None:
    processes = gpu_processes(gpu_uuid)
    own = own_process_ids()
    foreign = [row for row in processes if row.get("pid") not in own]
    checkpoints.append(
        {
            "stage": stage,
            "checked_at": utc_now(),
            "processes": processes,
            "foreign_processes": foreign,
        }
    )
    if foreign and not allow_busy_gpu:
        pids = sorted({row.get("pid") for row in foreign})
        raise RuntimeError(
            f"GPU interference at {stage}: foreign compute processes {pids}; "
            "retry on an exclusive GPU or use --allow-busy-gpu for a result "
            "that is explicitly marked non-formal"
        )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _device_index(value: str) -> int:
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return torch.cuda.current_device() if device.index is None else device.index


def _model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    import transformers

    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    major = int(transformers.__version__.split(".", 1)[0])
    kwargs["dtype" if major >= 5 else "torch_dtype"] = torch.bfloat16
    if args.revision:
        kwargs["revision"] = args.revision
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    return kwargs


def load_components(args: argparse.Namespace):
    """Load once, retain only the embedding/norm/selected attention module."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, **_model_load_kwargs(args))
    if getattr(model.config, "model_type", None) != "qwen3":
        raise TypeError(
            f"expected a Qwen3 text checkpoint, got model_type={model.config.model_type!r}"
        )
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise TypeError("loaded model does not expose the expected Qwen3 decoder")
    if not 0 <= args.layer < len(backbone.layers):
        raise ValueError(
            f"--layer {args.layer} is outside the checkpoint's {len(backbone.layers)} layers"
        )
    decoder_layer = backbone.layers[args.layer]
    embedding = backbone.embed_tokens
    input_norm = decoder_layer.input_layernorm
    attention = decoder_layer.self_attn
    config = model.config
    if type(attention).__name__ != "Qwen3Attention":
        raise TypeError(f"expected Qwen3Attention, got {type(attention).__name__}")
    if getattr(attention, "sliding_window", None) is not None:
        raise RuntimeError("the frozen BTS submission does not support sliding attention")
    if attention.head_dim != 128:
        raise RuntimeError("the frozen BTS submission requires head_dim=128")
    if getattr(attention.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("dense baseline was not configured for the SDPA interface")

    # Child modules do not retain parent references.  Delete the unused model
    # before any long hidden state is materialized.
    del decoder_layer, backbone, model
    gc.collect()
    return config, embedding.eval(), input_norm.eval(), attention.eval()


@torch.inference_mode()
def build_hidden_states(
    embedding: torch.nn.Module,
    input_norm: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    if token_ids.device.type != "cpu" or token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("token_ids must have CPU shape [1, tokens]")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    hidden_size = int(embedding.weight.shape[1])
    hidden = torch.empty(
        (1, token_ids.shape[1], hidden_size),
        dtype=embedding.weight.dtype,
        device="cpu",
    )
    for first in range(0, token_ids.shape[1], chunk_tokens):
        last = min(first + chunk_tokens, token_ids.shape[1])
        chunk = embedding(token_ids[:, first:last])
        hidden[:, first:last].copy_(input_norm(chunk))
    return hidden


def extract_output(value: Any) -> torch.Tensor:
    output = value[0] if isinstance(value, tuple) else value
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"attention forward returned {type(output).__name__}")
    return output


def _sdp_backend_flags() -> dict[str, bool | None]:
    cuda_backend = torch.backends.cuda
    names = (
        "flash_sdp_enabled",
        "math_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "cudnn_sdp_enabled",
    )
    return {
        name: bool(getattr(cuda_backend, name)()) if hasattr(cuda_backend, name) else None
        for name in names
    }


@contextmanager
def forced_flash_sdp() -> Iterator[dict[str, bool | None]]:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        flags = _sdp_backend_flags()
        if flags["flash_sdp_enabled"] is not True:
            raise RuntimeError("PyTorch FlashAttention SDP backend is disabled")
        for name in (
            "math_sdp_enabled",
            "mem_efficient_sdp_enabled",
            "cudnn_sdp_enabled",
        ):
            if flags[name] is True:
                raise RuntimeError(f"dense fallback backend remained enabled: {name}")
        yield flags


def output_probe(
    output: torch.Tensor,
    positions: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    index = torch.tensor(tuple(positions), dtype=torch.long, device=output.device)
    sample = output.index_select(1, index).detach().cpu().contiguous()
    metadata = {
        "full_shape": list(output.shape),
        "dtype": str(output.dtype),
        "full_all_finite": bool(torch.isfinite(output).all().item()),
        "sample_positions": list(positions),
        "sample_sha256": tensor_sha256(sample),
        "sample_mean": float(sample.float().mean()),
        "sample_std": float(sample.float().std(unbiased=False)),
        "sample_l2": float(torch.linalg.vector_norm(sample.float())),
    }
    return sample, metadata


@contextmanager
def density_collection(collector: Any):
    if collector is None:
        yield None
        return
    previous = {
        "collect_density": collector.collect_density,
        "collect_select_time": collector.collect_select_time,
        "collect_stage_time": collector.collect_stage_time,
    }
    collector.reset()
    collector.collect_density = True
    collector.collect_select_time = False
    collector.collect_stage_time = False
    try:
        yield collector
    finally:
        collector.reset()
        for name, value in previous.items():
            setattr(collector, name, value)


@torch.inference_mode()
def validate_outputs_and_density(
    methods: dict[str, Callable[[], torch.Tensor]],
    *,
    length: int,
    probe_token_count: int,
    candidate_name: str = "bts",
    collector: Any = None,
) -> dict[str, Any]:
    positions = probe_indices(length, probe_token_count)
    torch.cuda.synchronize()
    dense_output = methods["dense_flash"]()
    torch.cuda.synchronize()
    dense_sample, dense_metadata = output_probe(dense_output, positions)
    del dense_output

    with density_collection(collector) as active_collector:
        candidate_output = methods[candidate_name]()
        torch.cuda.synchronize()
        candidate_sample, candidate_metadata = output_probe(
            candidate_output, positions
        )
        density_summary = (
            active_collector.summary() if active_collector is not None else None
        )
        del candidate_output
    if collector is None:
        density = 1.0
        collector_method = "implicit_full_attention"
        collector_count = 1
    else:
        if density_summary is None or density_summary.get("count") != 1:
            raise RuntimeError(
                f"{candidate_name} did not emit exactly one density record"
            )
        density = float(density_summary["total"])
        collector_method = density_summary["method"]
        collector_count = density_summary["count"]
    if not 0.0 < density <= 1.0:
        raise RuntimeError(f"{candidate_name} emitted invalid density {density}")
    if (
        not dense_metadata["full_all_finite"]
        or not candidate_metadata["full_all_finite"]
    ):
        raise FloatingPointError("attention output contains a non-finite value")
    comparison = numerical_comparison(dense_sample, candidate_sample)
    del dense_sample, candidate_sample
    return {
        "probe_only_not_accuracy_gate": True,
        "dense_flash": dense_metadata,
        candidate_name: candidate_metadata,
        "candidate_vs_dense": comparison,
        "sparse_vs_dense": comparison,
        "density": {
            "dense_attention_density": 1.0,
            "candidate": candidate_name,
            "candidate_logical_density": density,
            "candidate_pruned_fraction": 1.0 - density,
            "collector_method": collector_method,
            "collector_count": collector_count,
            "definition": DENSITY_DEFINITIONS[candidate_name],
            "semantics": (
                "Method-specific logical block density divided by all causal "
                "dense block slots for this one forward; native full attention "
                "is defined as density 1.0."
            ),
        },
    }


@torch.inference_mode()
def time_methods(
    methods: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    rounds: int,
    samples: int,
    checkpoints: list[dict[str, Any]],
    gpu_uuid: str,
    allow_busy_gpu: bool,
    candidate_name: str = "bts",
) -> dict[str, Any]:
    method_names = ("dense_flash", candidate_name)
    warmup_orders: list[list[str]] = []
    for index in range(warmup):
        order = list(measurement_order(0, index, candidate_name))
        warmup_orders.append(order)
        for name in order:
            output = methods[name]()
            del output
    torch.cuda.synchronize()

    round_rows: list[dict[str, Any]] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for round_index in range(rounds):
            check_gpu_exclusive(
                checkpoints,
                gpu_uuid=gpu_uuid,
                stage=f"timing_round_{round_index}_before",
                allow_busy_gpu=allow_busy_gpu,
            )
            event_samples = {name: [] for name in method_names}
            wall_samples = {name: [] for name in method_names}
            orders: list[list[str]] = []
            for sample_index in range(samples):
                order = list(
                    measurement_order(round_index, sample_index, candidate_name)
                )
                orders.append(order)
                for name in order:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    start_event.record()
                    output = methods[name]()
                    end_event.record()
                    end_event.synchronize()
                    wall_ms = (time.perf_counter() - wall_start) * 1000.0
                    event_samples[name].append(float(start_event.elapsed_time(end_event)))
                    wall_samples[name].append(wall_ms)
                    del output
            check_gpu_exclusive(
                checkpoints,
                gpu_uuid=gpu_uuid,
                stage=f"timing_round_{round_index}_after",
                allow_busy_gpu=allow_busy_gpu,
            )
            round_rows.append(
                {
                    "round": round_index,
                    "orders": orders,
                    "methods": {
                        name: {
                            "cuda_event_ms": summarize(event_samples[name]),
                            "synchronized_wall_ms": summarize(wall_samples[name]),
                        }
                        for name in method_names
                    },
                }
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    aggregate: dict[str, Any] = {}
    for name in method_names:
        event_values = [
            value
            for row in round_rows
            for value in row["methods"][name]["cuda_event_ms"]["samples"]
        ]
        wall_values = [
            value
            for row in round_rows
            for value in row["methods"][name]["synchronized_wall_ms"]["samples"]
        ]
        aggregate[name] = {
            "cuda_event_ms": summarize(event_values),
            "synchronized_wall_ms": summarize(wall_values),
            "round_cuda_event_medians_ms": [
                row["methods"][name]["cuda_event_ms"]["median"] for row in round_rows
            ],
        }
    ratios = [
        row["methods"]["dense_flash"]["cuda_event_ms"]["median"]
        / row["methods"][candidate_name]["cuda_event_ms"]["median"]
        for row in round_rows
    ]
    speedup = {
        "candidate": candidate_name,
        "round_median_ratios": ratios,
        "median_of_round_ratios": statistics.median(ratios),
        "ratio_of_aggregate_medians": (
            aggregate["dense_flash"]["cuda_event_ms"]["median"]
            / aggregate[candidate_name]["cuda_event_ms"]["median"]
        ),
    }
    result = {
        "clock": "torch.cuda.Event elapsed_time in milliseconds",
        "scope": SCOPE,
        "non_claim": NON_CLAIM,
        "protocol": (
            "Alternating AB/BA calls of the same module and immutable arguments; "
            "foreign compute-process checks before and after every timing round; "
            "validation/density calls and warmups excluded from samples; no CUDA Graph."
        ),
        "warmup_orders": warmup_orders,
        "rounds": round_rows,
        "aggregate": aggregate,
        "candidate_speedup_vs_dense_flash": speedup,
    }
    if candidate_name == "bts":
        result["bts_speedup_vs_dense_flash"] = speedup
    return result


def make_token_ids(max_length: int, vocab_size: int, seed: int) -> torch.Tensor:
    if max_length <= 0 or vocab_size <= 0:
        raise ValueError("max_length and vocab_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(1, max_length),
        generator=generator,
        dtype=torch.long,
        device="cpu",
    )


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def load_efficiency_method(
    name: str,
) -> tuple[Callable[..., Any] | None, Any, dict[str, Any]]:
    """Load one candidate after the selected GPU passes the busy-process gate."""
    from baselines.registry import (
        configure_environment,
        dependency_report,
        load_forward,
        method_contract,
    )

    configure_environment(name)
    contract = method_contract(name)
    if name == "bts":
        from bts.bts import STAT_COLLECTOR, bts_attention_forward

        return bts_attention_forward, STAT_COLLECTOR, contract
    if name == "flashattention":
        dependency_report(name, strict=True)
        from baselines.FlashAttention import preflight_backend

        observed = preflight_backend()
        if observed.get("callable") is not True:
            raise RuntimeError("FlashAttention-2 preflight returned no callable backend")
        contract["backend_preflight"] = observed
        return None, None, contract
    return load_forward(name)


@torch.inference_mode()
def execute(args: argparse.Namespace, report: dict[str, Any]) -> None:
    inventory = gpu_inventory()
    logical_index = _device_index(args.device)
    physical_index = resolve_physical_gpu(
        logical_index,
        args.physical_gpu,
        inventory,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    hardware = next(row for row in inventory if int(row["index"]) == physical_index)
    gpu_uuid = hardware["uuid"]
    report["environment"]["gpu"] = {
        "logical_device": f"cuda:{logical_index}",
        "physical_index": physical_index,
        "nvidia_smi": hardware,
    }
    report["gpu_exclusivity"]["selected_gpu_uuid"] = gpu_uuid
    atomic_write_json(args.output, report)

    checkpoints = report["gpu_exclusivity"]["checkpoints"]
    check_gpu_exclusive(
        checkpoints,
        gpu_uuid=gpu_uuid,
        stage="before_cuda_initialization",
        allow_busy_gpu=args.allow_busy_gpu,
    )
    candidate_forward, candidate_collector, candidate_contract = (
        load_efficiency_method(args.method)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(logical_index)
    device = torch.device("cuda", logical_index)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    config, embedding, input_norm, attention = load_components(args)
    check_gpu_exclusive(
        checkpoints,
        gpu_uuid=gpu_uuid,
        stage="after_checkpoint_load",
        allow_busy_gpu=args.allow_busy_gpu,
    )
    report["environment"]["gpu"].update(
        {
            "torch_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )

    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    config_dict = config.to_dict()
    report["model"] = {
        "requested": args.model,
        "requested_revision": args.revision,
        "resolved_name_or_path": _config_value(config, "_name_or_path"),
        "resolved_commit_hash": _config_value(config, "_commit_hash"),
        "model_type": _config_value(config, "model_type"),
        "config_sha256": canonical_sha256(config_dict),
        "layer": args.layer,
        "num_hidden_layers": _config_value(config, "num_hidden_layers"),
        "hidden_size": _config_value(config, "hidden_size"),
        "num_attention_heads": _config_value(config, "num_attention_heads"),
        "num_key_value_heads": _config_value(config, "num_key_value_heads"),
        "head_dim": attention.head_dim,
        "vocab_size": _config_value(config, "vocab_size"),
        "max_position_embeddings": _config_value(config, "max_position_embeddings"),
        "rope_parameters": _config_value(config, "rope_parameters"),
        "rope_scaling": _config_value(config, "rope_scaling"),
        "attention_class": f"{type(attention).__module__}.{type(attention).__name__}",
        "original_forward": (
            f"{attention.forward.__func__.__module__}."
            f"{attention.forward.__func__.__qualname__}"
        ),
        "component_state_sha256": {
            "embed_tokens": module_state_sha256(embedding),
            "input_layernorm": module_state_sha256(input_norm),
            "self_attn": module_state_sha256(attention),
        },
    }
    report["standard_protocol"] = standard_protocol_audit(
        args,
        model_metadata=report["model"],
        compute_capability=report["environment"]["gpu"]["compute_capability"],
    )
    report["method"] = {
        "candidate": candidate_contract,
        "dense": {
            "transformers_attention_interface": "sdpa",
            "kernel_enforcement": (
                "torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)"
            ),
            "fallback_backends_permitted": [],
        },
        "same_attention_module_object_and_arguments": True,
        "batch_size": 1,
        "dtype": "torch.bfloat16",
        "statistics_during_timing": "disabled",
    }
    atomic_write_json(args.output, report)

    # Environment variables can enable the frozen collector at import time.
    # Formal latency must not inherit scalar copies, auxiliary CUDA events, or
    # retained density samples from the caller's shell.
    if candidate_collector is not None:
        candidate_collector.collect_density = False
        candidate_collector.collect_select_time = False
        candidate_collector.collect_stage_time = False
        candidate_collector.reset()

    original_dense_forward = attention.forward
    if candidate_forward is not None:
        attention.forward = MethodType(candidate_forward, attention)
    attention.to(device=device, dtype=torch.bfloat16).eval()
    rotary = Qwen3RotaryEmbedding(config, device=device).to(device).eval()
    if getattr(attention, "layer_idx", None) != args.layer:
        raise RuntimeError("selected attention module has an unexpected layer index")

    token_ids = make_token_ids(max(args.lengths), int(config.vocab_size), args.seed)
    report["input_protocol"] = {
        "kind": "deterministic synthetic token-ID prefixes",
        "seed": args.seed,
        "generator": "torch.Generator(device='cpu').manual_seed(seed); torch.randint",
        "maximum_token_ids_sha256": tensor_sha256(token_ids),
        "prefix_lengths": list(args.lengths),
        "hidden_transform": (
            f"checkpoint embed_tokens then layer {args.layer} input_layernorm; "
            "computed outside timing"
        ),
        "position_transform": (
            "checkpoint Qwen3RotaryEmbedding over positions [0, length); computed "
            "outside timing"
        ),
        "rope_policy": (
            "Use the checkpoint config unchanged; this runner never inserts or "
            "modifies YaRN/rope scaling."
        ),
        "lengths_above_config_max_position_embeddings": [
            length
            for length in args.lengths
            if length > int(getattr(config, "max_position_embeddings", length))
        ],
        "long_context_quality_caveat": (
            "Lengths above the checkpoint's configured context are still useful for "
            "this synthetic latency protocol, but do not establish long-context "
            "quality. Supply a separately prepared YaRN checkpoint when that policy "
            "is required."
        ),
        "is_actual_first_layer_attention_input": args.layer == 0,
        "deeper_layer_caveat": (
            None
            if args.layer == 0
            else "Embedding-derived synthetic input is not a real upstream activation "
            f"of decoder layer {args.layer}."
        ),
        "shared_immutable_inputs_for_both_methods": True,
    }
    atomic_write_json(args.output, report)

    for length in args.lengths:
        print(f"[efficiency] preparing length={length}", flush=True)
        check_gpu_exclusive(
            checkpoints,
            gpu_uuid=gpu_uuid,
            stage=f"length_{length}_before",
            allow_busy_gpu=args.allow_busy_gpu,
        )
        hidden_cpu = build_hidden_states(
            embedding,
            input_norm,
            token_ids[:, :length],
            chunk_tokens=args.input_chunk_tokens,
        )
        input_record = {
            "length": length,
            "token_ids_sha256": tensor_sha256(token_ids[:, :length]),
            "hidden_states_sha256": tensor_sha256(hidden_cpu),
            "hidden_shape": list(hidden_cpu.shape),
            "hidden_dtype": str(hidden_cpu.dtype),
        }
        hidden = hidden_cpu.to(device=device, dtype=torch.bfloat16)
        del hidden_cpu
        position_ids = torch.arange(length, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = rotary(hidden, position_ids)
        input_record["position_ids_sha256"] = tensor_sha256(position_ids)
        input_record["argument_fingerprint_sha256"] = canonical_sha256(input_record)

        dense_backend_flags: dict[str, bool | None] = {}

        def original_call(attn_implementation: str) -> torch.Tensor:
            previous = attention.config._attn_implementation
            attention.config._attn_implementation = attn_implementation
            try:
                return extract_output(
                    original_dense_forward(
                        hidden,
                        position_embeddings,
                        None,
                        past_key_values=None,
                    )
                )
            finally:
                attention.config._attn_implementation = previous

        def dense_call() -> torch.Tensor:
            nonlocal dense_backend_flags
            with forced_flash_sdp() as flags:
                dense_backend_flags = flags
                return original_call("sdpa")

        if args.method == "flashattention":
            def candidate_call() -> torch.Tensor:
                return original_call("flash_attention_2")
        else:
            def candidate_call() -> torch.Tensor:
                return extract_output(
                    attention(
                        hidden,
                        position_embeddings,
                        None,
                        past_key_values=None,
                    )
                )

        methods = {"dense_flash": dense_call, args.method: candidate_call}
        torch.cuda.reset_peak_memory_stats(device)
        print(f"[efficiency] validating outputs/density length={length}", flush=True)
        validation = validate_outputs_and_density(
            methods,
            length=length,
            probe_token_count=args.probe_tokens,
            candidate_name=args.method,
            collector=candidate_collector,
        )
        print(f"[efficiency] timing length={length}", flush=True)
        timing = time_methods(
            methods,
            warmup=args.warmup,
            rounds=args.rounds,
            samples=args.samples,
            checkpoints=checkpoints,
            gpu_uuid=gpu_uuid,
            allow_busy_gpu=args.allow_busy_gpu,
            candidate_name=args.method,
        )
        check_gpu_exclusive(
            checkpoints,
            gpu_uuid=gpu_uuid,
            stage=f"length_{length}_after",
            allow_busy_gpu=args.allow_busy_gpu,
        )
        result = {
            "length": length,
            "input": input_record,
            "dense_backend_flags_inside_forced_context": dense_backend_flags,
            "validation": validation,
            "timing": timing,
            "peak_cuda_memory": {
                "allocated_bytes": torch.cuda.max_memory_allocated(device),
                "reserved_bytes": torch.cuda.max_memory_reserved(device),
                "scope": (
                    "Process peak after this length's validation, warmup, and timing; "
                    "not a method-isolated memory benchmark."
                ),
            },
        }
        report["results"].append(result)
        atomic_write_json(args.output, report)

        del methods, hidden, position_ids, position_embeddings
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()

    del attention, original_dense_forward, rotary, embedding, input_norm, token_ids
    del candidate_forward, candidate_collector
    gc.collect()
    torch.cuda.empty_cache()
    check_gpu_exclusive(
        checkpoints,
        gpu_uuid=gpu_uuid,
        stage="after_all_lengths",
        allow_busy_gpu=args.allow_busy_gpu,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen3 text checkpoint path or Hub ID")
    parser.add_argument("--revision", help="Optional immutable Hugging Face revision/commit")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--method",
        choices=EFFICIENCY_METHODS,
        default="bts",
        help=(
            "Candidate paired against forced PyTorch Flash SDPA: BTS, "
            "FlashAttention-2, FlexPrefill, MInference, or XAttention"
        ),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=DEFAULT_LENGTHS,
        help="Comma-separated token lengths (default: 16384,32768,65536,131072)",
    )
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--warmup", type=int, default=4, help="Warmup pairs per length")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=9, help="Samples per method per round")
    parser.add_argument("--probe-tokens", type=int, default=128)
    parser.add_argument("--input-chunk-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.layer < 0:
        parser.error("--layer must be nonnegative")
    for name in ("warmup", "rounds", "samples", "probe_tokens", "input_chunk_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        _device_index(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.physical_gpu is not None and args.physical_gpu < 0:
        parser.error("--physical-gpu must be nonnegative")
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists() and not args.overwrite:
        parser.error("--output already exists; pass --overwrite to replace it")
    return args


def initial_report(args: argparse.Namespace) -> dict[str, Any]:
    from baselines.registry import get_spec

    return {
        "schema_version": 1,
        "method_display_name": get_spec(args.method).display_name,
        "status": "running",
        "formal_valid": False,
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "non_claim": NON_CLAIM,
        "started_at": utc_now(),
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "triton": _package_version("triton"),
            "flash_attn": _package_version("flash-attn"),
            "einops": _package_version("einops"),
            "minference": _package_version("minference"),
            "block_sparse_attn": _package_version("block-sparse-attn"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "source": source_hashes(),
        "gpu_exclusivity": {
            "allow_busy_gpu": args.allow_busy_gpu,
            "formal_requires_no_observed_foreign_compute_processes": True,
            "checkpoints": [],
        },
        "results": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from baselines.registry import configure_environment

    configure_environment(args.method)
    report = initial_report(args)
    atomic_write_json(args.output, report)
    try:
        execute(args, report)
        observed_exclusive = all(
            not row["foreign_processes"]
            for row in report["gpu_exclusivity"]["checkpoints"]
        )
        report["gpu_exclusivity"]["observed_exclusive"] = observed_exclusive
        report["formal_valid"] = bool(
            not args.allow_busy_gpu
            and observed_exclusive
            and report["standard_protocol"]["accepted"]
            and len(report["results"]) == len(args.lengths)
        )
        report["status"] = "complete" if report["formal_valid"] else "complete_nonformal"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "repr": repr(error),
        }
        raise
    finally:
        report["finished_at"] = utc_now()
        atomic_write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
