import importlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from transformers import AutoConfig


@dataclass(frozen=True)
class AttentionAdapter:
    """One verified Transformers attention-class integration."""

    module_name: str
    class_name: str


ATTENTION_ADAPTERS = {
    "qwen3": AttentionAdapter(
        module_name="transformers.models.qwen3.modeling_qwen3",
        class_name="Qwen3Attention",
    ),
    "llama": AttentionAdapter(
        module_name="transformers.models.llama.modeling_llama",
        class_name="LlamaAttention",
    ),
}


def _normalize_model_type(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("model config has no non-empty string model_type")
    return value.lower().replace("-", "_")


def _local_model_type(model_id: str) -> str | None:
    """Return a local checkpoint's declared model type, if ``model_id`` is local."""
    path = Path(model_id).expanduser()
    if not path.is_dir():
        return None
    config_path = path / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"local model directory '{model_id}' has no config.json; "
            "cannot verify the text-model contract"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not read local model config '{config_path}': {error}"
        ) from error
    try:
        return _normalize_model_type(config.get("model_type"))
    except RuntimeError as error:
        raise RuntimeError(f"local model config '{config_path}' {error}") from error


def resolve_model_type(model_id: str, *, revision: str | None = None) -> str:
    """Resolve an exact config model type without trusting repository code."""
    model_id = str(model_id or "")
    if not model_id:
        raise RuntimeError(
            "model_id must be a local checkpoint or Hugging Face model ID"
        )
    local_model_type = _local_model_type(model_id)
    if local_model_type is not None:
        return local_model_type

    load_kwargs: dict[str, object] = {"trust_remote_code": False}
    if revision is not None:
        load_kwargs["revision"] = revision
    try:
        config = AutoConfig.from_pretrained(model_id, **load_kwargs)
    except Exception as error:
        raise RuntimeError(
            f"could not load the config for remote model '{model_id}': {error}"
        ) from error
    return _normalize_model_type(getattr(config, "model_type", None))


def get_attention_classes(
    model_id: str,
    *,
    revision: str | None = None,
    allowed_model_types: Iterable[str] | None = None,
):
    """Resolve the verified attention class for a supported text model."""
    model_type = resolve_model_type(model_id, revision=revision)
    allowed = (
        None
        if allowed_model_types is None
        else {_normalize_model_type(value) for value in allowed_model_types}
    )
    if allowed is not None and model_type not in allowed:
        raise RuntimeError(
            f"model_type={model_type!r} is not supported by this attention method; "
            f"allowed model types: {', '.join(sorted(allowed))}"
        )
    adapter = ATTENTION_ADAPTERS.get(model_type)
    if adapter is None:
        supported = ", ".join(sorted(ATTENTION_ADAPTERS))
        raise RuntimeError(
            f"unsupported model_type={model_type!r} for '{model_id}'; "
            f"supported text model types: {supported}"
        )
    module = importlib.import_module(adapter.module_name)
    attention = getattr(module, adapter.class_name, None)
    if attention is None:
        raise RuntimeError(
            f"could not find {adapter.class_name} in the installed Transformers package"
        )
    return [attention]


@lru_cache(maxsize=None)
def get_rotary_fn(module_name: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, "apply_rotary_pos_emb", None)


def apply_patch(
    forward_fn: Callable,
    model_id: str | None,
    *,
    revision: str | None = None,
    allowed_model_types: Iterable[str] | None = None,
) -> int:
    attn_classes = get_attention_classes(
        model_id=model_id,
        revision=revision,
        allowed_model_types=allowed_model_types,
    )
    for cls in attn_classes:
        cls.forward = forward_fn
    return len(attn_classes)
