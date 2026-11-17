"""Strict 128K Density replay primitives.

This module is intentionally smaller than the user-facing Density runner.  It
ports the measurement path used by the parent experiment:

* read the exact model prompt recorded in an official lm-eval sample log;
* tokenize it once and repeat/truncate that token row to exactly 131072 tokens;
* run the unpatched dense model through layers 0..34 while capturing the input
  to every attention layer, then stop immediately before dense layer 35 runs;
* replay BTS attention once for each of those 36 dense-upstream inputs and use
  the collector's arithmetic mean as the per-prompt density.

There is deliberately no generation call here.  Density is an attention-only
diagnostic over inputs captured from one dense full-model prefill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

MODEL_ID = "Qwen/Qwen3-8B"
SEQUENCE_LENGTH = 131072
EXPECTED_LAYERS = 36


class _DenseCaptureComplete(Exception):
    """Private control-flow signal raised before the final attention executes."""


@dataclass(frozen=True)
class OfficialPrompt:
    """One exact prompt and its immutable source/output row identities."""

    task: str
    index: int
    prompt: str
    source_path: str
    source_line_1_based: int
    source_row_sha256: str
    official_output_path: str
    official_output_line_1_based: int
    official_output_row_sha256: str
    prompt_sha256: str
    prompt_characters: int
    source_declared_length: int | None

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        # The full prompt is an input, not useful duplicated result metadata.
        value.pop("prompt")
        return value


@dataclass
class CapturedAttentionCall:
    """A dense-upstream invocation ready for attention-only replay."""

    layer_index: int
    module: torch.nn.Module
    positional: tuple[Any, ...]
    keywords: dict[str, Any]
    hidden_shape: tuple[int, ...]
    hidden_dtype: str
    capture_device: str


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _selected_jsonl_row(path: Path, index: int) -> tuple[bytes, dict[str, Any]]:
    """Read a zero-based frozen data row and verify its embedded index."""
    with path.open("rb") as stream:
        for line_index, line in enumerate(stream):
            if line_index != index:
                continue
            raw = line.rstrip(b"\r\n")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError(f"{path} line {index + 1} is not a JSON object")
            observed_index = int(payload.get("index", index))
            if observed_index != index:
                raise ValueError(
                    f"{path} line {index + 1} has payload index {observed_index}"
                )
            prompt_input = payload.get("input")
            if not isinstance(prompt_input, str) or not prompt_input:
                raise ValueError(f"{path} line {index + 1} has no non-empty input")
            return raw, payload
    raise IndexError(f"{path} has no zero-based line index {index}")


def resolve_official_output_log(suite_root: Path, task: str) -> Path:
    """Resolve the single frozen official sample log for ``task``."""
    outputs = suite_root.expanduser().resolve() / "outputs"
    matches = sorted(outputs.glob(f"samples_official_fixed_{task}_*.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one official output log for {task!r} under "
            f"{outputs}, found {len(matches)}"
        )
    return matches[0]


def _selected_official_output_row(
    path: Path,
    *,
    index: int,
    maximum_length: int,
) -> tuple[bytes, dict[str, Any], int]:
    matches: list[tuple[bytes, dict[str, Any], int]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = line.rstrip(b"\r\n")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                continue
            document = payload.get("doc")
            if not isinstance(document, dict):
                continue
            if int(document.get("max_length", -1)) != maximum_length:
                continue
            if int(document.get("index", -1)) != index:
                continue
            matches.append((raw, payload, line_number))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {maximum_length}-token official output row "
            f"for index {index} in {path}, found {len(matches)}"
        )
    return matches[0]


def load_official_prompt(
    suite_root: Path,
    task: str,
    index: int,
    *,
    maximum_length: int = SEQUENCE_LENGTH,
    official_output_path: Path | None = None,
) -> OfficialPrompt:
    """Load the exact logged model input, cross-checked against frozen data.

    The official prompt is ``arguments.gen_args_0.arg_0``.  Reconstructing a
    chat prompt from ``doc.input`` is intentionally not accepted here because
    that would no longer replay the audited model input byte for byte.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    suite = suite_root.expanduser().resolve()
    source_path = suite / "data" / "128k" / f"{task}.jsonl"
    output_path = (
        resolve_official_output_log(suite, task)
        if official_output_path is None
        else official_output_path.expanduser().resolve()
    )
    source_raw, source_payload = _selected_jsonl_row(source_path, index)
    output_raw, output_payload, output_line = _selected_official_output_row(
        output_path,
        index=index,
        maximum_length=maximum_length,
    )
    document = output_payload["doc"]
    if document.get("input") != source_payload["input"]:
        raise RuntimeError(f"official data/output prompt mismatch for {task}[{index}]")
    try:
        prompt = output_payload["arguments"]["gen_args_0"]["arg_0"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"official output has no exact model input for {task}[{index}]"
        ) from error
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError(f"empty exact model input for {task}[{index}]")
    declared_length = document.get("length")
    return OfficialPrompt(
        task=task,
        index=index,
        prompt=prompt,
        source_path=str(source_path),
        source_line_1_based=index + 1,
        source_row_sha256=_sha256(source_raw),
        official_output_path=str(output_path),
        official_output_line_1_based=output_line,
        official_output_row_sha256=_sha256(output_raw),
        prompt_sha256=_sha256(prompt.encode("utf-8")),
        prompt_characters=len(prompt),
        source_declared_length=(
            int(declared_length) if declared_length is not None else None
        ),
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    native = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(native.dtype).encode("ascii"))
    digest.update(json.dumps(list(native.shape), separators=(",", ":")).encode("ascii"))
    byte_view = native.view(torch.uint8).reshape(-1)
    chunk_bytes = 8 * 1024 * 1024
    for first in range(0, byte_view.numel(), chunk_bytes):
        digest.update(memoryview(byte_view[first : first + chunk_bytes].numpy()))
    return digest.hexdigest()


def fit_exact_token_ids(
    encoded: torch.Tensor,
    maximum_length: int = SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the parent's exact truncate-or-repeat-whole-row policy."""
    if encoded.device.type != "cpu" or encoded.ndim != 2 or encoded.shape[0] != 1:
        raise ValueError("encoded token IDs must have CPU shape [1, tokens]")
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    original_count = int(encoded.shape[1])
    if original_count <= 0:
        raise ValueError("the official prompt tokenized to an empty sequence")
    if original_count >= maximum_length:
        exact = encoded[:, :maximum_length].contiguous()
        policy = "truncate"
    else:
        # This is deliberately the same expression used by the parent runner.
        repeats = (maximum_length // original_count) + 1
        exact = encoded.repeat(1, repeats)[:, :maximum_length].contiguous()
        policy = "repeat_prompt_token_ids_then_truncate"
    metadata = {
        "token_count_before_exact_length": original_count,
        "tokens_appended_from_prompt_prefix": max(0, maximum_length - original_count),
        "tokens_truncated": max(0, original_count - maximum_length),
        "benchmark_token_count": maximum_length,
        "length_policy": policy,
        "exact_token_ids_sha256": _tensor_sha256(exact),
    }
    return exact, metadata


def tokenize_official_prompt(
    tokenizer: Any,
    prompt: OfficialPrompt | str,
    *,
    maximum_length: int = SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Tokenize exactly as the parent benchmark and build one exact-length row."""
    prompt_text = prompt.prompt if isinstance(prompt, OfficialPrompt) else prompt
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt must be non-empty text")
    encoded = tokenizer(prompt_text, return_tensors="pt").input_ids
    if not isinstance(encoded, torch.Tensor):
        raise TypeError("tokenizer did not return tensor input_ids")
    encoded = encoded.detach().to(device="cpu", dtype=torch.long)
    exact, metadata = fit_exact_token_ids(encoded, maximum_length)
    metadata["prompt_sha256"] = _sha256(prompt_text.encode("utf-8"))
    return exact, metadata


def load_dense_model_and_tokenizer(
    *,
    model_id: str = MODEL_ID,
    device: str | torch.device = "cuda",
    revision: str | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
) -> tuple[Any, torch.nn.Module]:
    """Load an unpatched BF16 base model for dense-upstream capture."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    common: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        common["revision"] = revision
    if cache_dir is not None:
        common["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
        **common,
    )
    model.eval()
    validate_dense_qwen3_model(model)
    return tokenizer, model


def _model_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("loaded model does not expose model.layers")
    return layers


def validate_dense_qwen3_model(
    model: torch.nn.Module,
    *,
    expected_layers: int = EXPECTED_LAYERS,
) -> None:
    """Reject a patched/non-Qwen3 model before the dense capture begins."""
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise TypeError("density replay requires the base Qwen3 model family")
    expected_architecture = {
        "hidden_size": 4096,
        "num_hidden_layers": expected_layers,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    mismatches = {
        name: getattr(config, name, None)
        for name, expected in expected_architecture.items()
        if getattr(config, name, None) != expected
    }
    if mismatches:
        raise ValueError(
            "density replay requires the Qwen3-8B architecture; mismatched "
            f"configuration values: {mismatches}"
        )
    layers = _model_layers(model)
    if len(layers) != expected_layers:
        raise ValueError(
            f"density replay requires {expected_layers} layers, found {len(layers)}"
        )
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise TypeError(f"model layer {index} has no self_attn module")
        if getattr(attention, "layer_idx", index) != index:
            raise RuntimeError(f"attention layer {index} has an unexpected layer_idx")
        forward = type(attention).forward
        module_name = str(getattr(forward, "__module__", ""))
        function_name = str(getattr(forward, "__name__", ""))
        if module_name.startswith(("bts.", "baselines.")) or function_name in {
            "bts_attention_forward",
            "prism_attention_forward",
        }:
            raise RuntimeError(
                "the model attention class was patched before dense capture"
            )


def _clone_capture(value: Any, capture_device: torch.device | None) -> Any:
    if isinstance(value, torch.Tensor):
        if capture_device is None:
            return value.detach().clone()
        return value.detach().to(device=capture_device, copy=True)
    if isinstance(value, tuple):
        return tuple(_clone_capture(item, capture_device) for item in value)
    if isinstance(value, list):
        return [_clone_capture(item, capture_device) for item in value]
    if isinstance(value, dict):
        return {
            key: _clone_capture(item, capture_device) for key, item in value.items()
        }
    return value


def _hidden_states(
    positional: Sequence[Any], keywords: Mapping[str, Any]
) -> torch.Tensor:
    hidden = keywords.get("hidden_states")
    if hidden is None and positional:
        hidden = positional[0]
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError("captured attention call has no tensor hidden_states")
    return hidden


@torch.inference_mode()
def capture_dense_attention_calls(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    device: str | torch.device,
    expected_layers: int = EXPECTED_LAYERS,
    capture_device: str | torch.device | None = None,
) -> tuple[CapturedAttentionCall, ...]:
    """Capture all dense-upstream attention inputs from one model prefill.

    Hooks only observe and clone inputs.  Dense attention executes normally for
    every layer except the last; the private exception at the final pre-hook is
    the same early-stop boundary as the parent experiment.
    """
    if input_ids.device.type != "cpu" or input_ids.ndim != 2:
        raise ValueError("input_ids must be a CPU rank-2 tensor")
    if tuple(input_ids.shape) != (1, SEQUENCE_LENGTH):
        raise ValueError(
            "formal density capture requires input_ids shape "
            f"[1, {SEQUENCE_LENGTH}], got {list(input_ids.shape)}"
        )
    layers = _model_layers(model)
    if len(layers) != expected_layers:
        raise ValueError(
            f"expected {expected_layers} attention layers, found {len(layers)}"
        )
    # A real Hugging Face model must pass the complete base-model check.  Tiny
    # structural fakes used by unit tests intentionally have no HF config.
    if getattr(getattr(model, "config", None), "model_type", None) is not None:
        validate_dense_qwen3_model(model, expected_layers=expected_layers)
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise TypeError(f"model layer {index} has no self_attn module")
        attention_forward = type(attention).forward
        if str(getattr(attention_forward, "__module__", "")).startswith(
            ("bts.", "baselines.")
        ) or str(getattr(attention_forward, "__name__", "")) in {
            "bts_attention_forward",
            "prism_attention_forward",
        }:
            raise RuntimeError(
                "the model attention class was patched before dense capture"
            )
    destination = None if capture_device is None else torch.device(capture_device)
    captured: dict[int, CapturedAttentionCall] = {}
    handles: list[Any] = []

    def make_hook(layer_index: int) -> Callable[..., None]:
        def hook(
            module: torch.nn.Module,
            positional: tuple[Any, ...],
            keywords: dict[str, Any],
        ) -> None:
            if layer_index in captured:
                raise RuntimeError(
                    f"dense attention layer {layer_index} was invoked more than once"
                )
            cloned_positional = tuple(_clone_capture(positional, destination))
            cloned_keywords = dict(_clone_capture(keywords, destination))
            hidden = _hidden_states(cloned_positional, cloned_keywords)
            if tuple(hidden.shape[:2]) != tuple(input_ids.shape):
                raise RuntimeError(
                    f"layer {layer_index} hidden batch/length {list(hidden.shape[:2])} "
                    f"does not match input IDs {list(input_ids.shape)}"
                )
            captured[layer_index] = CapturedAttentionCall(
                layer_index=layer_index,
                module=module,
                positional=cloned_positional,
                keywords=cloned_keywords,
                hidden_shape=tuple(int(value) for value in hidden.shape),
                hidden_dtype=str(hidden.dtype),
                capture_device=str(hidden.device),
            )
            if layer_index == expected_layers - 1:
                raise _DenseCaptureComplete()

        return hook

    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise TypeError(f"model layer {index} has no self_attn module")
        handles.append(
            attention.register_forward_pre_hook(make_hook(index), with_kwargs=True)
        )
    try:
        try:
            model(input_ids.to(device=device), use_cache=False)
        except _DenseCaptureComplete:
            pass
    finally:
        for handle in handles:
            handle.remove()
    expected = list(range(expected_layers))
    if sorted(captured) != expected:
        raise RuntimeError(
            "dense forward did not capture every attention layer: "
            f"observed={sorted(captured)}, expected={expected}"
        )
    return tuple(captured[index] for index in expected)


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


@torch.inference_mode()
def replay_bts_attention_density(
    calls: Sequence[CapturedAttentionCall],
    *,
    device: str | torch.device,
    expected_layers: int = EXPECTED_LAYERS,
    forward: Callable[..., Any] | None = None,
    collector: Any | None = None,
) -> dict[str, Any]:
    """Replay BTS once per captured layer and return the 36-layer mean."""
    if len(calls) != expected_layers:
        raise ValueError(f"expected {expected_layers} captured calls, got {len(calls)}")
    if [call.layer_index for call in calls] != list(range(expected_layers)):
        raise ValueError("captured calls are not in contiguous layer order")
    if forward is None or collector is None:
        from bts.bts import STAT_COLLECTOR, bts_attention_forward

        forward = bts_attention_forward if forward is None else forward
        collector = STAT_COLLECTOR if collector is None else collector
    if not callable(forward):
        raise TypeError("BTS forward must be callable")
    required_collector = (
        "collect_density",
        "collect_select_time",
        "collect_stage_time",
        "reset",
        "summary",
    )
    missing = [name for name in required_collector if not hasattr(collector, name)]
    if missing:
        raise TypeError("BTS collector is missing: " + ", ".join(missing))

    previous_flags = {
        "collect_density": collector.collect_density,
        "collect_select_time": collector.collect_select_time,
        "collect_stage_time": collector.collect_stage_time,
    }
    target = torch.device(device)
    collector.reset()
    collector.collect_density = True
    collector.collect_select_time = False
    collector.collect_stage_time = False
    summary: dict[str, Any] | None = None
    densities: tuple[float, ...] = ()
    try:
        for call in calls:
            positional = _move_to_device(call.positional, target)
            keywords = _move_to_device(call.keywords, target)
            output = forward(call.module, *positional, **keywords)
            del output, positional, keywords
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        summary = collector.summary()
        raw_densities = getattr(collector, "_densities", None)
        if not isinstance(raw_densities, list):
            raise TypeError("BTS collector does not expose its per-layer densities")
        densities = tuple(float(value) for value in raw_densities)
    finally:
        collector.reset()
        for name, value in previous_flags.items():
            setattr(collector, name, value)
    if summary is None:
        raise RuntimeError("BTS replay emitted no density summary")
    if int(summary.get("count", -1)) != expected_layers:
        raise RuntimeError(
            "BTS replay did not emit exactly one density per layer: "
            f"{summary.get('count')} != {expected_layers}"
        )
    if len(densities) != expected_layers:
        raise RuntimeError(
            f"BTS collector retained {len(densities)} per-layer densities"
        )
    if any(not 0.0 < value <= 1.0 for value in densities):
        raise RuntimeError("BTS replay emitted an invalid logical density")
    mean_density = float(summary["total"])
    expected_mean = sum(densities) / len(densities)
    if mean_density != expected_mean:
        raise RuntimeError("BTS collector mean does not match per-layer densities")
    return {
        "layer_densities": list(densities),
        "mean_density": mean_density,
        "pruning_fraction": 1.0 - mean_density,
        "collector_method": summary.get("method"),
        "collector_count": int(summary["count"]),
        "generation_performed": False,
        "execution_scope": (
            "dense full-model upstream capture followed by BTS attention-only "
            "replay across all 36 layers"
        ),
    }


def evaluate_official_density_sample(
    model: torch.nn.Module,
    tokenizer: Any,
    suite_root: Path,
    task: str,
    index: int,
    *,
    device: str | torch.device,
    capture_device: str | torch.device | None = None,
    official_output_path: Path | None = None,
) -> dict[str, Any]:
    """Run one complete, generation-free official Density sample."""
    prompt = load_official_prompt(
        suite_root,
        task,
        index,
        official_output_path=official_output_path,
    )
    input_ids, input_metadata = tokenize_official_prompt(tokenizer, prompt)
    calls = capture_dense_attention_calls(
        model,
        input_ids,
        device=device,
        capture_device=capture_device,
    )
    capture_metadata = [
        {
            "layer_index": call.layer_index,
            "hidden_shape": list(call.hidden_shape),
            "hidden_dtype": call.hidden_dtype,
            "capture_device": call.capture_device,
        }
        for call in calls
    ]
    result = replay_bts_attention_density(calls, device=device)
    return {
        **prompt.metadata(),
        **input_metadata,
        **result,
        "dense_capture": capture_metadata,
    }


__all__ = (
    "EXPECTED_LAYERS",
    "MODEL_ID",
    "SEQUENCE_LENGTH",
    "CapturedAttentionCall",
    "OfficialPrompt",
    "capture_dense_attention_calls",
    "evaluate_official_density_sample",
    "fit_exact_token_ids",
    "load_dense_model_and_tokenizer",
    "load_official_prompt",
    "replay_bts_attention_density",
    "resolve_official_output_log",
    "tokenize_official_prompt",
    "validate_dense_qwen3_model",
)
