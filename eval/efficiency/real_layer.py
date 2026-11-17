#!/usr/bin/env python3
"""Measure one selected Qwen3 attention layer across frozen methods.

The stable submission launcher points this runner at the same immutable
``replay_{length}.pt`` captures used by the best complete efficiency run in the
parent experiment.  A non-default layer can instead be captured from the real
model or loaded from an explicitly supplied matching replay.  The default
protocol is 32K/64K/128K, three independent
rounds, three warmups and eight measured forwards per round.  Forced PyTorch
Flash SDPA is the dense baseline; MInference, FlexPrefill, XAttention and BTS
replay the same module object and immutable arguments.

The historical measurements are provenance references, never pre-filled as a
new result.  A new run remains non-formal until its own output is inspected,
because hardware, dependencies and submission source are not byte-identical to
the historical development checkout.  Without ``--attention-inputs-dir`` this
file supports explicit real-model capture for a selected layer; that remains a
non-reference diagnostic rather than the default audited submission protocol.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
SUBMISSION_ROOT = HERE.parent.parent
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from baselines.registry import DENSITY_DEFINITIONS, method_contract
from eval.efficiency.run import (
    _device_index,
    _model_load_kwargs,
    _package_version,
    atomic_write_json,
    canonical_sha256,
    check_gpu_exclusive,
    density_collection,
    extract_output,
    file_sha256,
    forced_flash_sdp,
    gpu_inventory,
    load_efficiency_method,
    make_token_ids,
    module_state_sha256,
    numerical_comparison,
    output_probe,
    parse_lengths,
    probe_indices,
    resolve_physical_gpu,
    source_hashes,
    summarize,
    tensor_sha256,
)


PROTOCOL = "qwen3_layer12_frozen_replay_efficiency_v1"
SELECTED_LAYER_PROTOCOL = "qwen3_selected_layer_attention_efficiency_v1"
METHODS = ("bts", "flashattention", "flexprefill", "minference", "xattention")
DEFAULT_METHODS = ("minference", "xattention", "bts")
DEFAULT_LENGTHS = (32768, 65536, 131072)
REFERENCE_MODEL = "Qwen/Qwen3-8B"
REFERENCE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
REFERENCE_LAYER = 12
REFERENCE_INPUT_SHA256 = {
    32768: "4db748990b094bd25d50a841d0ee75287a1d7f25e83dee462df1c7a56cb16088",
    65536: "9cc07c61a1f3a15d579ad7e4bc5efa917e11f44d71a4a452b6e2d8f180316df4",
    131072: "6770614bb8328d16382047bbf6ca9f9c782e147a454828f08d658c6311d509c0",
}
REFERENCE_RESULT = {
    "path": (
        "/home/lsf/experiments/prism-token-sparse/work/"
        "efficiency_current_fourway_strict_20261106/summary.json"
    ),
    "sha256": "d7bf3f227a15d9844ac2644af684a2a3f98924220e2b7c0849c7c224ac71c423",
    "reproduced_by_current_run": False,
    "bts_latency_ms_median": {
        "32768": 22.60690402984619,
        "65536": 51.51089572906494,
        "131072": 140.15753555297852,
    },
    "bts_speedup_vs_flash_median": {
        "32768": 2.1871817985706343,
        "65536": 3.3546792508781813,
        "131072": 4.881255906645183,
    },
}
SCOPE = (
    "One captured Qwen3 attention-module prefill forward: QKV projections, "
    "Q/K norm, RoPE, attention, and output projection. Sparse methods also "
    "include selection, index/mask construction, packing, and their kernel."
)
NON_CLAIM = (
    "Captured target-layer attention-module only; not full-model latency, generation "
    "speed, request throughput, or serving throughput. Historical reference "
    "numbers are provenance and are not results of the current run."
)


class _CaptureComplete(Exception):
    """Private control-flow signal raised before the selected attention runs."""


def _uses_reference_replay(args: argparse.Namespace) -> bool:
    return bool(
        args.attention_inputs_dir is not None
        and args.model == REFERENCE_MODEL
        and args.revision == REFERENCE_REVISION
        and args.layer == REFERENCE_LAYER
    )


def _expected_reference_input_sha256(
    args: argparse.Namespace, length: int
) -> str | None:
    if not _uses_reference_replay(args):
        return None
    return REFERENCE_INPUT_SHA256.get(length)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_methods(value: str) -> tuple[str, ...]:
    aliases = {
        "flash-attention": "flashattention",
        "flash_attention": "flashattention",
        "flashattention2": "flashattention",
        "fa2": "flashattention",
        "flex-prefill": "flexprefill",
        "x-attention": "xattention",
        "xattn": "xattention",
    }
    methods: list[str] = []
    for raw in value.split(","):
        name = aliases.get(raw.strip().lower(), raw.strip().lower())
        if not name:
            continue
        if name not in METHODS:
            expected = ", ".join(METHODS)
            raise argparse.ArgumentTypeError(
                f"unsupported method {raw!r}; expected one of: {expected}"
            )
        if name in methods:
            raise argparse.ArgumentTypeError(f"--methods contains duplicate {name!r}")
        methods.append(name)
    if not methods:
        raise argparse.ArgumentTypeError("--methods must contain at least one method")
    return tuple(methods)


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    return value


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _tensor_manifest(value: Any, prefix: str = "root") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, torch.Tensor):
        rows.append(
            {
                "path": prefix,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
        )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            rows.extend(_tensor_manifest(item, f"{prefix}[{index}]"))
    elif isinstance(value, dict):
        for key in sorted(value):
            rows.extend(_tensor_manifest(value[key], f"{prefix}.{key}"))
    return rows


def _hidden_states(
    positional: Sequence[Any], keywords: Mapping[str, Any]
) -> torch.Tensor:
    hidden = keywords.get("hidden_states")
    if hidden is None and positional:
        hidden = positional[0]
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError("captured attention call has no tensor hidden_states")
    return hidden


def _model_backbone(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise TypeError("loaded model does not expose model.layers")
    return backbone


def _selected_attention(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    backbone = _model_backbone(model)
    if not 0 <= layer < len(backbone.layers):
        raise ValueError(
            f"--layer {layer} is outside the checkpoint's {len(backbone.layers)} layers"
        )
    attention = backbone.layers[layer].self_attn
    if type(attention).__name__ != "Qwen3Attention":
        raise TypeError(f"expected Qwen3Attention, got {type(attention).__name__}")
    if getattr(attention, "layer_idx", layer) != layer:
        raise RuntimeError("selected attention module has an unexpected layer index")
    return attention


@torch.inference_mode()
def capture_attention_call(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    layer: int,
    device: torch.device,
) -> tuple[torch.nn.Module, tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """Capture one module invocation and stop before its attention executes."""
    if token_ids.device.type != "cpu" or token_ids.ndim != 2:
        raise ValueError("token_ids must be a CPU rank-2 tensor")
    attention = _selected_attention(model, layer)
    captured: dict[str, Any] = {}

    def hook(
        module: torch.nn.Module,
        positional: tuple[Any, ...],
        keywords: dict[str, Any],
    ) -> None:
        del module
        captured["positional"] = _clone_to_cpu(positional)
        captured["keywords"] = _clone_to_cpu(keywords)
        raise _CaptureComplete()

    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        try:
            model(token_ids.to(device=device), use_cache=False)
        except _CaptureComplete:
            pass
    finally:
        handle.remove()
    if "positional" not in captured or "keywords" not in captured:
        raise RuntimeError(f"model forward never reached attention layer {layer}")

    positional = tuple(captured["positional"])
    keywords = dict(captured["keywords"])
    hidden = _hidden_states(positional, keywords)
    if tuple(hidden.shape[:2]) != tuple(token_ids.shape):
        raise RuntimeError(
            "captured hidden-state batch/length differs from the token input: "
            f"hidden={tuple(hidden.shape[:2])}, tokens={tuple(token_ids.shape)}"
        )
    metadata = {
        "layer": layer,
        "hidden_shape": list(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
        "hidden_states_sha256": tensor_sha256(hidden),
        "tensor_arguments": _tensor_manifest(
            {"positional": positional, "keywords": keywords}
        ),
    }
    metadata["argument_manifest_sha256"] = canonical_sha256(
        metadata["tensor_arguments"]
    )
    return attention, positional, keywords, metadata


def _hf_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    return kwargs


def _prompt_from_json(path: Path, index: int) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        try:
            sample = payload[index]
        except IndexError as error:
            raise ValueError(f"data index {index} is outside {path}") from error
    elif isinstance(payload, dict):
        key = str(index)
        if key not in payload:
            raise ValueError(f"data index {index} is absent from {path}")
        sample = payload[key]
    else:
        raise TypeError("--data-path JSON must contain a list or object")
    if not isinstance(sample, dict):
        raise TypeError("selected data sample must be an object")

    prompt: Any = sample.get("prompt", sample.get("text"))
    if prompt is None:
        origin = sample.get("origin_prompt")
        if isinstance(origin, list) and origin and isinstance(origin[0], dict):
            prompt = origin[0].get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "selected data sample must contain prompt/text or "
            "origin_prompt[0].prompt"
        )
    return prompt


def fit_token_ids(
    encoded: torch.Tensor, maximum_length: int
) -> tuple[torch.Tensor, str]:
    """Apply the historical truncate-or-repeat policy to one token row."""
    if encoded.device.type != "cpu" or encoded.ndim != 2 or encoded.shape[0] != 1:
        raise ValueError("encoded token IDs must have CPU shape [1, tokens]")
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    original_count = int(encoded.shape[1])
    if original_count <= 0:
        raise ValueError("the selected prompt tokenized to an empty sequence")
    if original_count >= maximum_length:
        return encoded[:, :maximum_length].contiguous(), "truncate"
    repeats = (maximum_length + original_count - 1) // original_count
    return (
        encoded.repeat(1, repeats)[:, :maximum_length].contiguous(),
        "repeat_then_truncate",
    )


def build_input_ids(
    args: argparse.Namespace,
    *,
    maximum_length: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build one deterministic maximum-length token sequence on CPU."""
    from transformers import AutoConfig, AutoTokenizer

    if args.data_path is not None:
        prompt = _prompt_from_json(args.data_path, args.data_index)
        source = {
            "kind": "json_prompt",
            "path": str(args.data_path),
            "file_sha256": file_sha256(args.data_path),
            "data_index": args.data_index,
        }
    elif args.prompt is not None:
        prompt = args.prompt
        source = {"kind": "command_line_prompt"}
    else:
        config = AutoConfig.from_pretrained(args.model, **_hf_config_kwargs(args))
        ids = make_token_ids(maximum_length, int(config.vocab_size), args.seed)
        return ids, {
            "kind": "deterministic_random_token_ids",
            "seed": args.seed,
            "original_token_count": maximum_length,
            "length_policy": "prefix",
            "maximum_token_ids_sha256": tensor_sha256(ids),
        }

    tokenizer = AutoTokenizer.from_pretrained(args.model, **_hf_config_kwargs(args))
    encoded = tokenizer(prompt, return_tensors="pt").input_ids.to(
        device="cpu", dtype=torch.long
    )
    original_count = int(encoded.shape[1])
    ids, policy = fit_token_ids(encoded, maximum_length)
    source.update(
        {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "original_token_count": original_count,
            "length_policy": policy,
            "maximum_token_ids_sha256": tensor_sha256(ids),
        }
    )
    return ids, source


def capture_real_layer(
    args: argparse.Namespace,
    token_ids: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """Load a full model, capture one layer call, and release other weights."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, **_model_load_kwargs(args)
    ).to(device=device)
    model.eval()
    if getattr(model.config, "model_type", None) != "qwen3":
        raise TypeError(
            f"expected a Qwen3 text checkpoint, got {model.config.model_type!r}"
        )
    model.config.use_cache = False
    attention, positional, keywords, metadata = capture_attention_call(
        model,
        token_ids,
        layer=args.layer,
        device=device,
    )
    metadata.update(
        {
            "model_type": model.config.model_type,
            "resolved_name_or_path": getattr(model.config, "_name_or_path", None),
            "resolved_commit_hash": getattr(model.config, "_commit_hash", None),
            "max_position_embeddings": getattr(
                model.config, "max_position_embeddings", None
            ),
            "attention_class": f"{type(attention).__module__}.{type(attention).__name__}",
            "attention_state_sha256": module_state_sha256(attention),
        }
    )

    # Child modules do not retain their parent.  Keeping only this reference
    # gives the replay benchmark the same layer weights without retaining the
    # embedding and other decoder layers in the process peak.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    positional = _move_to_device(positional, device)
    keywords = _move_to_device(keywords, device)
    return attention, positional, keywords, metadata


def load_replay_layer(
    args: argparse.Namespace,
    length: int,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """Load and validate one trusted parent-experiment layer capture."""
    if args.attention_inputs_dir is None:
        raise ValueError("--attention-inputs-dir is required for replay loading")
    path = args.attention_inputs_dir / f"replay_{length}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen replay input: {path}")

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"replay payload must be an object: {path}")
    required = {"module", "args", "kwargs", "input_sha256", "length"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"replay payload is missing {missing}: {path}")

    attention = state["module"]
    positional = state["args"]
    keywords = state["kwargs"]
    if not isinstance(attention, torch.nn.Module):
        raise TypeError(f"replay module is not torch.nn.Module: {path}")
    if not isinstance(positional, tuple) or not isinstance(keywords, dict):
        raise TypeError(f"replay args/kwargs have an unexpected type: {path}")
    hidden = _hidden_states(positional, keywords)
    if int(state["length"]) != length or tuple(hidden.shape[:2]) != (1, length):
        raise ValueError(
            "replay sequence length or batch differs from the request: "
            f"state={state['length']}, hidden={tuple(hidden.shape[:2])}, requested={length}"
        )
    if hidden.dtype != torch.bfloat16:
        raise TypeError(f"replay hidden states must be BF16, got {hidden.dtype}")
    if type(attention).__name__ != "Qwen3Attention":
        raise TypeError(
            f"replay attention must be Qwen3Attention, got {type(attention).__name__}"
        )
    if int(getattr(attention, "layer_idx", -1)) != args.layer:
        raise ValueError(
            f"replay layer {getattr(attention, 'layer_idx', None)} differs from "
            f"requested layer {args.layer}"
        )
    if getattr(attention, "_prism_config", None) is not None:
        raise ValueError("replay attention carries a fixed _prism_config override")

    config = getattr(attention, "config", None)
    cached_model = getattr(config, "_name_or_path", None)
    cached_revision = getattr(config, "_commit_hash", None)
    if cached_model and cached_model != args.model:
        raise ValueError(
            f"replay model {cached_model!r} differs from requested model {args.model!r}"
        )
    if args.revision and cached_revision != args.revision:
        raise ValueError(
            f"replay revision {cached_revision!r} differs from requested "
            f"revision {args.revision!r}"
        )

    observed_input_sha256 = str(state["input_sha256"])
    expected_input_sha256 = _expected_reference_input_sha256(args, length)
    if (
        expected_input_sha256 is not None
        and observed_input_sha256 != expected_input_sha256
    ):
        raise ValueError(
            f"replay input hash mismatch for {length}: "
            f"expected {expected_input_sha256}, got {observed_input_sha256}"
        )

    metadata = {
        "kind": (
            "frozen_parent_layer_replay"
            if _uses_reference_replay(args)
            else "external_selected_layer_replay"
        ),
        "path": str(path.resolve()),
        "length": length,
        "layer": args.layer,
        "input_sha256": observed_input_sha256,
        "expected_input_sha256": expected_input_sha256,
        "input_sha256_matches_reference": (
            expected_input_sha256 is not None
            and observed_input_sha256 == expected_input_sha256
        ),
        "hidden_shape": list(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
        "model_type": getattr(config, "model_type", None),
        "resolved_name_or_path": cached_model,
        "resolved_commit_hash": cached_revision,
        "attention_class": (f"{type(attention).__module__}.{type(attention).__name__}"),
        "attention_state_sha256": module_state_sha256(attention),
        "tensor_arguments": _tensor_manifest(
            {"positional": positional, "keywords": keywords}
        ),
    }
    metadata["argument_manifest_sha256"] = canonical_sha256(
        metadata["tensor_arguments"]
    )

    attention = attention.to(device=device).eval()
    positional = _move_to_device(positional, device)
    keywords = _move_to_device(keywords, device)
    del state
    return attention, positional, keywords, metadata


@contextmanager
def collector_disabled(collector: Any) -> Iterator[None]:
    if collector is None:
        yield
        return
    names = (
        "collect_density",
        "collect_select_time",
        "collect_stage_time",
    )
    previous = {
        name: getattr(collector, name) for name in names if hasattr(collector, name)
    }
    collector.reset()
    for name in previous:
        setattr(collector, name, False)
    try:
        yield
    finally:
        collector.reset()
        for name, value in previous.items():
            setattr(collector, name, value)


def _original_call(
    attention: torch.nn.Module,
    original_forward: Callable[..., Any],
    positional: tuple[Any, ...],
    keywords: dict[str, Any],
    implementation: str,
    *,
    force_flash_sdpa: bool,
) -> torch.Tensor:
    previous = attention.config._attn_implementation
    attention.config._attn_implementation = implementation
    try:
        if force_flash_sdpa:
            with forced_flash_sdp():
                return extract_output(original_forward(*positional, **keywords))
        return extract_output(original_forward(*positional, **keywords))
    finally:
        attention.config._attn_implementation = previous


def build_calls(
    attention: torch.nn.Module,
    positional: tuple[Any, ...],
    keywords: dict[str, Any],
    prepared: Mapping[str, tuple[Callable[..., Any] | None, Any, dict[str, Any]]],
) -> tuple[dict[str, Callable[[], torch.Tensor]], dict[str, Any]]:
    original_forward = attention.forward
    calls: dict[str, Callable[[], torch.Tensor]] = {
        "dense_flash": lambda: _original_call(
            attention,
            original_forward,
            positional,
            keywords,
            "sdpa",
            force_flash_sdpa=True,
        )
    }
    collectors: dict[str, Any] = {"dense_flash": None}
    for name, (forward, collector, _contract) in prepared.items():
        collectors[name] = collector
        if name == "flashattention":
            calls[name] = lambda: _original_call(
                attention,
                original_forward,
                positional,
                keywords,
                "flash_attention_2",
                force_flash_sdpa=False,
            )
            continue
        if forward is None:
            raise RuntimeError(f"method {name} has no replay forward")

        def candidate_call(
            candidate_forward: Callable[..., Any] = forward,
        ) -> torch.Tensor:
            return extract_output(candidate_forward(attention, *positional, **keywords))

        calls[name] = candidate_call
    return calls, collectors


@torch.inference_mode()
def benchmark_call(
    call: Callable[[], torch.Tensor],
    *,
    collector: Any,
    warmup: int,
    runs: int,
    rounds: int,
    device: torch.device,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    cuda_rounds: list[list[float]] = []
    wall_rounds: list[list[float]] = []
    allocated_peaks: list[int] = []
    reserved_peaks: list[int] = []
    with collector_disabled(collector):
        for _round_index in range(rounds):
            for _ in range(warmup):
                output = call()
                del output
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.reset_peak_memory_stats(device)
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(runs)
            ]
            wall_samples: list[float] = []
            for start, end in events:
                wall_start = time.perf_counter()
                start.record()
                output = call()
                end.record()
                end.synchronize()
                wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
                del output
            cuda_samples = [float(start.elapsed_time(end)) for start, end in events]
            cuda_rounds.append(cuda_samples)
            wall_rounds.append(wall_samples)
            allocated_peaks.append(int(torch.cuda.max_memory_allocated(device)))
            reserved_peaks.append(int(torch.cuda.max_memory_reserved(device)))

    cuda_flat = [sample for round_samples in cuda_rounds for sample in round_samples]
    wall_flat = [sample for round_samples in wall_rounds for sample in round_samples]
    cuda_summary = summarize(cuda_flat)
    wall_summary = summarize(wall_flat)
    cuda_round_means = [summarize(samples)["mean"] for samples in cuda_rounds]
    wall_round_means = [summarize(samples)["mean"] for samples in wall_rounds]
    cuda_summary.update(
        {
            "round_means": cuda_round_means,
            "median_of_round_means": summarize(cuda_round_means)["median"],
        }
    )
    wall_summary.update(
        {
            "round_means": wall_round_means,
            "median_of_round_means": summarize(wall_round_means)["median"],
        }
    )

    return {
        "clock": "torch.cuda.Event elapsed_time in milliseconds",
        "rounds": rounds,
        "round_aggregation": "median of per-round means",
        "warmup_runs_per_round": warmup,
        "measured_samples_per_round": runs,
        "cuda_event_ms": cuda_summary,
        "synchronized_wall_ms": wall_summary,
        "peak_cuda_memory": {
            "allocated_bytes": max(allocated_peaks),
            "reserved_bytes": max(reserved_peaks),
            "allocated_gib": max(allocated_peaks) / (1024**3),
            "reserved_gib": max(reserved_peaks) / (1024**3),
            "scope": (
                "Maximum of per-round peaks after each round's warmup reset and "
                "during clean replay forwards; common attention weights and "
                "captured inputs remain resident."
            ),
        },
    }


@torch.inference_mode()
def diagnose_call(
    call: Callable[[], torch.Tensor],
    *,
    collector: Any,
    method: str,
    length: int,
    probe_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with density_collection(collector) as active_collector:
        output = call()
        torch.cuda.synchronize()
        sample, output_metadata = output_probe(
            output, probe_indices(length, probe_tokens)
        )
        density_summary = (
            active_collector.summary() if active_collector is not None else None
        )
        del output
    if collector is None:
        density = 1.0
        collector_count = 1
        collector_method = "implicit_full_attention"
    else:
        if density_summary is None or density_summary.get("count") != 1:
            raise RuntimeError(f"{method} did not emit exactly one density record")
        density = float(density_summary["total"])
        collector_count = int(density_summary["count"])
        collector_method = density_summary["method"]
    definition = (
        "forced full causal PyTorch Flash SDPA"
        if method == "dense_flash"
        else DENSITY_DEFINITIONS[method]
    )
    return sample, {
        "output": output_metadata,
        "density": {
            "logical_density": density,
            "collector_count": collector_count,
            "collector_method": collector_method,
            "definition": definition,
        },
    }


def _display_name(method: str) -> str:
    if method == "dense_flash":
        return "Dense Flash SDPA"
    return method_contract(method)["display_name"]


def _summary_row(
    method: str,
    length: int,
    result: Mapping[str, Any],
    *,
    dense_mean: float | None,
    dense_round_means: Sequence[float] | None = None,
) -> dict[str, Any]:
    if "timing" not in result:
        return {
            "method": _display_name(method),
            "method_id": method,
            "seq_len": length,
            "status": result.get("status"),
            "attn_latency_ms_mean": None,
            "attn_latency_ms_std": None,
            "attn_accel": None,
            "peak_memory_gb": None,
            "density": None,
        }
    timing = result["timing"]
    mean = float(
        timing["cuda_event_ms"].get(
            "median_of_round_means", timing["cuda_event_ms"]["mean"]
        )
    )
    diagnostic = result.get("diagnostic") or {}
    density = (diagnostic.get("density") or {}).get("logical_density")
    method_round_means = timing["cuda_event_ms"].get("round_means")
    paired_speedups = None
    if (
        dense_round_means is not None
        and method_round_means is not None
        and len(dense_round_means) == len(method_round_means)
        and all(float(value) > 0 for value in method_round_means)
    ):
        paired_speedups = [
            float(dense_value) / float(method_value)
            for dense_value, method_value in zip(dense_round_means, method_round_means)
        ]
    acceleration = (
        float(summarize(paired_speedups)["median"])
        if paired_speedups is not None
        else None if dense_mean is None or mean <= 0 else dense_mean / mean
    )
    return {
        "method": _display_name(method),
        "method_id": method,
        "seq_len": length,
        "status": result.get("status", "ok"),
        "attn_latency_ms_mean": mean,
        "attn_latency_ms_std": float(timing["cuda_event_ms"]["std"]),
        "select_latency_ms_mean": None,
        "select_latency_ms_std": None,
        "attn_accel": acceleration,
        "paired_round_speedups": paired_speedups,
        "peak_memory_gb": float(timing["peak_cuda_memory"]["allocated_gib"]),
        "mem_overhead_gb": None,
        "density": density,
    }


def _terminal_metric(
    value: Any,
    *,
    digits: int = 4,
    suffix: str = "",
) -> str:
    """Format one measured terminal-table value without inventing a result."""
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def _terminal_mean_std(
    row: Mapping[str, Any],
    mean_key: str,
    std_key: str,
) -> str:
    mean = row.get(mean_key)
    if mean is None:
        return "-"
    std = row.get(std_key)
    if std is None:
        return _terminal_metric(mean)
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def render_terminal_summary(report: Mapping[str, Any], output: Path) -> str:
    """Render the current run's measured rows in the parent table style."""
    columns = (
        ("Method", 48),
        ("SeqLen", 7),
        ("Attn Latency (ms)", 21),
        ("Accel", 9),
        ("Mem GiB", 9),
        ("Density", 11),
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
        "Efficiency summary (measurements from this run)",
        divider,
        table_row(header for header, _width in columns),
        divider,
    ]
    rows = report.get("summary_rows") or []
    if not rows:
        lines.append("No measured summary rows were produced.")
    for row in rows:
        density = row.get("density")
        lines.append(
            table_row(
                (
                    row.get("method", row.get("method_id", "unknown")),
                    row.get("seq_len", "-"),
                    _terminal_mean_std(
                        row,
                        "attn_latency_ms_mean",
                        "attn_latency_ms_std",
                    ),
                    _terminal_metric(row.get("attn_accel"), digits=3, suffix="x"),
                    _terminal_metric(row.get("peak_memory_gb"), digits=3),
                    (
                        _terminal_metric(
                            100.0 * float(density),
                            digits=4,
                            suffix="%",
                        )
                        if density is not None
                        else "-"
                    ),
                )
            )
        )
    run_status = str(report.get("status", "unknown"))
    if report.get("formal_valid") is True:
        protocol_validity = "FORMAL"
    elif run_status == "complete_nonformal":
        protocol_validity = "NON-FORMAL"
    else:
        protocol_validity = "INVALID"
    lines.extend(
        (
            divider,
            f"Run status: {run_status}",
            f"Protocol validity: {protocol_validity}",
            f"Saved efficiency results to {output.resolve()}",
        )
    )
    return "\n".join(lines)


def _runner_source_hashes() -> dict[str, Any]:
    record = source_hashes()
    relative = str(Path(__file__).resolve().relative_to(SUBMISSION_ROOT))
    record["files"][relative] = file_sha256(Path(__file__).resolve())
    record["manifest_sha256"] = canonical_sha256(record["files"])
    return record


def _historical_reference() -> dict[str, Any]:
    reference = dict(REFERENCE_RESULT)
    path = Path(str(reference["path"]))
    observed = file_sha256(path) if path.is_file() else None
    reference.update(
        {
            "source_present": path.is_file(),
            "observed_sha256": observed,
            "source_hash_verified": observed == reference["sha256"],
        }
    )
    return reference


def _reference_protocol_audit(args: argparse.Namespace) -> dict[str, Any]:
    from bts.config import describe as describe_bts

    bts = describe_bts()
    expected = {
        "model": REFERENCE_MODEL,
        "revision": REFERENCE_REVISION,
        "layer": REFERENCE_LAYER,
        "lengths": list(DEFAULT_LENGTHS),
        "methods": list(DEFAULT_METHODS),
        "rounds": 3,
        "warmup": 3,
        "runs": 8,
        "batch_size": 1,
        "dense_baseline": "forced PyTorch Flash SDPA",
        "bts_profile": {
            "block_thresholds": [0.93, 0.93],
            "token_ratio": 0.4,
            "token_score": "qk_block_mean",
            "query_block_size": 128,
            "dense_tail_tokens": 256,
        },
    }
    observed = {
        "model": args.model,
        "revision": args.revision,
        "layer": args.layer,
        "lengths": list(args.lengths),
        "methods": list(args.methods),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "runs": args.runs,
        "batch_size": 1,
        "bts_profile": {
            "block_thresholds": [
                bts.get("low_freq_threshold"),
                bts.get("high_freq_threshold"),
            ],
            "token_ratio": bts.get("token_ratio"),
            "token_score": bts.get("token_score"),
            "query_block_size": bts.get("query_block_size"),
            "dense_tail_tokens": bts.get("dense_tail_tokens"),
        },
        "attention_inputs_dir": (
            str(args.attention_inputs_dir)
            if args.attention_inputs_dir is not None
            else None
        ),
    }
    comparable = {
        key: observed[key]
        for key in (
            "model",
            "revision",
            "layer",
            "lengths",
            "methods",
            "rounds",
            "warmup",
            "runs",
            "batch_size",
            "bts_profile",
        )
    }
    expected_comparable = {key: expected[key] for key in comparable}
    replay_files = {}
    if args.attention_inputs_dir is not None:
        replay_files = {
            str(length): {
                "path": str(args.attention_inputs_dir / f"replay_{length}.pt"),
                "exists": (args.attention_inputs_dir / f"replay_{length}.pt").is_file(),
                "expected_input_sha256": _expected_reference_input_sha256(args, length),
            }
            for length in args.lengths
        }
    replay_files_complete = set(replay_files) == {
        str(length) for length in DEFAULT_LENGTHS
    } and all(row["exists"] for row in replay_files.values())
    return {
        "matches_defaults": (
            comparable == expected_comparable and replay_files_complete
        ),
        "expected": expected,
        "observed": observed,
        "replay_files": replay_files,
        "note": (
            "Configuration equivalence does not copy or guarantee historical "
            "measurements; each real run writes its own timing values."
        ),
    }


def _initial_report(args: argparse.Namespace) -> dict[str, Any]:
    reference_audit = _reference_protocol_audit(args)
    uses_reference_replay = _uses_reference_replay(args)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL if uses_reference_replay else SELECTED_LAYER_PROTOCOL,
        "status": "running",
        "formal_valid": False,
        "formal_valid_reason": (
            (
                "This run replays the audited historical inputs, but uses the "
                "submission source and current environment; it cannot inherit the "
                "historical result's validity or numbers."
            )
            if uses_reference_replay
            else (
                "This run uses a non-reference layer or input source and cannot "
                "inherit the layer-12 historical protocol's validity or numbers."
            )
        ),
        "scope": SCOPE,
        "non_claim": NON_CLAIM,
        "started_at": utc_now(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
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
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "source": _runner_source_hashes(),
        "method_contracts": {
            method: method_contract(method) for method in args.methods
        },
        "gpu_exclusivity": {
            "allow_busy_gpu": args.allow_busy_gpu,
            "checkpoints": [],
        },
        "input_protocol": {
            "kind": (
                "audited frozen parent layer replay"
                if uses_reference_replay
                else (
                    "external selected-layer replay"
                    if args.attention_inputs_dir is not None
                    else "full-model real target-layer activation"
                )
            ),
            "upstream_attention": "dense SDPA",
            "capture_point": "immediately before the selected self-attention call",
            "capture_storage": "detached CPU tensors replayed on the selected GPU",
            "shared_immutable_arguments_for_all_methods": True,
        },
        "reference_protocol_match": reference_audit,
        "historical_reference": _historical_reference(),
        "results": [],
        "summary_rows": [],
    }


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
    checkpoints = report["gpu_exclusivity"]["checkpoints"]
    check_gpu_exclusive(
        checkpoints,
        gpu_uuid=gpu_uuid,
        stage="before_cuda_initialization",
        allow_busy_gpu=args.allow_busy_gpu,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(logical_index)
    device = torch.device("cuda", logical_index)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    prepared = {method: load_efficiency_method(method) for method in args.methods}
    report["method_contracts"] = {
        method: prepared[method][2] for method in args.methods
    }
    token_ids: torch.Tensor | None = None
    if args.attention_inputs_dir is None:
        token_ids, input_source = build_input_ids(
            args, maximum_length=max(args.lengths)
        )
        report["input_protocol"].update(input_source)
    else:
        report["input_protocol"].update(
            {
                "attention_inputs_dir": str(args.attention_inputs_dir),
                "expected_input_sha256": {
                    str(length): _expected_reference_input_sha256(args, length)
                    for length in args.lengths
                },
            }
        )
    atomic_write_json(args.output, report)

    saw_failure = False
    for length in args.lengths:
        length_record: dict[str, Any] = {
            "length": length,
            "status": "running",
            "methods": [],
        }
        if token_ids is not None:
            length_record["token_ids_sha256"] = tensor_sha256(token_ids[:, :length])
        report["results"].append(length_record)
        atomic_write_json(args.output, report)
        check_gpu_exclusive(
            checkpoints,
            gpu_uuid=gpu_uuid,
            stage=f"length_{length}_before_capture",
            allow_busy_gpu=args.allow_busy_gpu,
        )
        action = "loading" if args.attention_inputs_dir else "capturing"
        print(f"{action} layer={args.layer} length={length}", flush=True)
        try:
            if args.attention_inputs_dir is not None:
                attention, positional, keywords, capture = load_replay_layer(
                    args, length, device=device
                )
            else:
                if token_ids is None:
                    raise AssertionError("capture mode has no token IDs")
                attention, positional, keywords, capture = capture_real_layer(
                    args,
                    token_ids[:, :length],
                    device=device,
                )
        except torch.cuda.OutOfMemoryError as error:
            length_record.update(
                {
                    "status": "oom_during_capture",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
            saw_failure = True
            torch.cuda.empty_cache()
            atomic_write_json(args.output, report)
            continue
        length_record["capture"] = capture
        check_gpu_exclusive(
            checkpoints,
            gpu_uuid=gpu_uuid,
            stage=f"length_{length}_after_capture",
            allow_busy_gpu=args.allow_busy_gpu,
        )
        calls, collectors = build_calls(attention, positional, keywords, prepared)
        method_order = ("dense_flash", *args.methods)
        result_by_method: dict[str, dict[str, Any]] = {}
        for method in method_order:
            print(f"timing method={method} length={length}", flush=True)
            check_gpu_exclusive(
                checkpoints,
                gpu_uuid=gpu_uuid,
                stage=f"length_{length}_{method}_before",
                allow_busy_gpu=args.allow_busy_gpu,
            )
            method_result: dict[str, Any] = {
                "method": method,
                "display_name": _display_name(method),
                "status": "running",
            }
            length_record["methods"].append(method_result)
            result_by_method[method] = method_result
            try:
                method_result["timing"] = benchmark_call(
                    calls[method],
                    collector=collectors[method],
                    warmup=args.warmup,
                    runs=args.runs,
                    rounds=args.rounds,
                    device=device,
                )
                method_result["status"] = "ok"
            except torch.cuda.OutOfMemoryError as error:
                method_result.update(
                    {
                        "status": "oom",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                saw_failure = True
                torch.cuda.empty_cache()
            except Exception as error:
                method_result.update(
                    {
                        "status": "failed",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "repr": repr(error),
                        },
                    }
                )
                saw_failure = True
                gc.collect()
                torch.cuda.empty_cache()
            check_gpu_exclusive(
                checkpoints,
                gpu_uuid=gpu_uuid,
                stage=f"length_{length}_{method}_after",
                allow_busy_gpu=args.allow_busy_gpu,
            )
            atomic_write_json(args.output, report)

        dense_probe: torch.Tensor | None = None
        for method in method_order:
            method_result = result_by_method[method]
            if method_result["status"] != "ok":
                continue
            try:
                probe, diagnostic = diagnose_call(
                    calls[method],
                    collector=collectors[method],
                    method=method,
                    length=length,
                    probe_tokens=args.probe_tokens,
                )
                if method == "dense_flash":
                    dense_probe = probe
                elif dense_probe is not None:
                    diagnostic["candidate_vs_dense_flash"] = numerical_comparison(
                        dense_probe, probe
                    )
                method_result["diagnostic"] = diagnostic
                del probe
            except torch.cuda.OutOfMemoryError as error:
                method_result["status"] = "diagnostic_oom"
                method_result["diagnostic"] = {
                    "status": "oom",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
                saw_failure = True
                torch.cuda.empty_cache()
            except Exception as error:
                method_result["status"] = "diagnostic_failed"
                method_result["diagnostic"] = {
                    "status": "failed",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "repr": repr(error),
                    },
                }
                saw_failure = True
                gc.collect()
                torch.cuda.empty_cache()

        dense_result = result_by_method["dense_flash"]
        dense_mean = (
            float(
                dense_result["timing"]["cuda_event_ms"].get(
                    "median_of_round_means",
                    dense_result["timing"]["cuda_event_ms"]["mean"],
                )
            )
            if "timing" in dense_result
            else None
        )
        dense_round_means = (
            dense_result["timing"]["cuda_event_ms"].get("round_means")
            if "timing" in dense_result
            else None
        )
        for method in method_order:
            report["summary_rows"].append(
                _summary_row(
                    method,
                    length,
                    result_by_method[method],
                    dense_mean=dense_mean,
                    dense_round_means=dense_round_means,
                )
            )
        length_record["status"] = (
            "complete"
            if all(row["status"] == "ok" for row in length_record["methods"])
            else "complete_with_failures"
        )
        atomic_write_json(args.output, report)
        del calls, collectors, positional, keywords, attention
        gc.collect()
        torch.cuda.empty_cache()

    report["gpu_exclusivity"]["observed_exclusive"] = all(
        not row["foreign_processes"] for row in checkpoints
    )
    report["status"] = "complete_with_failures" if saw_failure else "complete_nonformal"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Qwen3 checkpoint path or Hub ID"
    )
    parser.add_argument("--revision", help="Optional immutable Hugging Face revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=DEFAULT_METHODS,
        help="Comma-separated candidates; default: minference,xattention,bts",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=REFERENCE_LAYER,
        help="Zero-based decoder layer whose real attention inputs are captured (default: 12)",
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=DEFAULT_LENGTHS,
        help="Comma-separated lengths (default: 32768,65536,131072)",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--attention-inputs-dir",
        type=Path,
        help=(
            "Directory containing trusted replay_{length}.pt layer captures; "
            "the stable launcher supplies the audited parent artifact directory"
        ),
    )
    input_group.add_argument(
        "--data-path",
        type=Path,
        help=(
            "Optional JSON prompt input. Supports prompt/text and the historical "
            "origin_prompt[0].prompt layout."
        ),
    )
    input_group.add_argument("--prompt", help="Optional literal prompt")
    parser.add_argument("--data-index", type=int, default=0, help="JSON sample index")
    parser.add_argument(
        "--seed",
        type=int,
        default=20261109,
        help="Seed for deterministic token IDs when no prompt input is supplied",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed replay forwards per method and length (default: 3)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=8,
        help="Timed replay forwards per method, length and round (default: 8)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Independent timing rounds (default: 3)",
    )
    parser.add_argument(
        "--probe-tokens",
        type=int,
        default=128,
        help="Evenly spaced output positions retained for diagnostics",
    )
    parser.add_argument("--device", default="cuda:0", help="Logical CUDA device")
    parser.add_argument(
        "--physical-gpu",
        type=int,
        help="Physical nvidia-smi index used to audit CUDA_VISIBLE_DEVICES mapping",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Allow foreign compute processes; results remain explicitly non-formal",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="One combined JSON; an existing file requires --overwrite",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output JSON",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.layer < 0:
        parser.error("--layer must be nonnegative")
    if args.data_index < 0:
        parser.error("--data-index must be nonnegative")
    for name in ("warmup", "runs", "rounds", "probe_tokens"):
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
    if args.data_path is not None:
        args.data_path = args.data_path.expanduser().resolve()
        if not args.data_path.is_file():
            parser.error(f"--data-path is not a file: {args.data_path}")
    if args.attention_inputs_dir is not None:
        args.attention_inputs_dir = args.attention_inputs_dir.expanduser().resolve()
        if not args.attention_inputs_dir.is_dir():
            parser.error(
                "--attention-inputs-dir is not a directory: "
                f"{args.attention_inputs_dir}"
            )
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists() and not args.overwrite:
        parser.error("--output already exists; pass --overwrite to replace it")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = _initial_report(args)
    atomic_write_json(args.output, report)
    try:
        execute(args, report)
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
    print(render_terminal_summary(report, args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
