"""Hugging Face integration for the frozen BTS submission pipeline."""

import math
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from .attention_utils import gqa_decode_attention, num_kv_groups
from .token_selection import STAT_COLLECTOR, bts_prefill_attention
from .utils.patch import apply_patch as _patch_attention_classes
from .utils.patch import get_rotary_fn


def _validate_attention_mask(attention_mask: torch.Tensor | None) -> None:
    """Accept standard causal masks with no padding; reject padding masks."""
    if attention_mask is None:
        return
    if attention_mask.ndim == 2:
        visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask > 0
        if bool(visible.all().item()):
            return
    if (
        attention_mask.ndim == 4
        and attention_mask.shape[1] == 1
        and attention_mask.shape[-2] > 0
        and attention_mask.shape[-1] > 0
        and attention_mask.shape[-2] <= attention_mask.shape[-1]
    ):
        q_len, kv_len = attention_mask.shape[-2:]
        query_positions = torch.arange(q_len, device=attention_mask.device) + kv_len - q_len
        key_positions = torch.arange(kv_len, device=attention_mask.device)
        expected_visible = key_positions[None, :] <= query_positions[:, None]
        if attention_mask.dtype == torch.bool:
            actual_visible = attention_mask
            valid_values = torch.ones((), device=attention_mask.device, dtype=torch.bool)
        else:
            actual_visible = attention_mask == 0
            valid_values = ((attention_mask == 0) | (attention_mask < 0)).all()
        standard_causal = (
            (actual_visible == expected_visible.view(1, 1, q_len, kv_len)).all()
            & valid_values
        )
        if bool(standard_causal.item()):
            return
    raise ValueError(
        "BTS submission supports causal attention with no padding mask; "
        "batch and pad prompts before calling the model"
    )


def bts_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Replacement attention forward with one fixed sparse-prefill method."""
    legacy_cache = kwargs.pop("past_key_value", None)
    if legacy_cache is not None:
        if past_key_values is not None and legacy_cache is not past_key_values:
            raise TypeError(
                "received different caches through past_key_value and past_key_values"
            )
        past_key_values = legacy_cache
    del kwargs
    if self.training:
        raise RuntimeError("the frozen BTS submission is inference-only")
    if getattr(self, "sliding_window", None) is not None:
        raise RuntimeError("sliding-window attention layers are not supported")
    if self.head_dim != 128:
        raise RuntimeError("the frozen BTS submission requires head_dim=128")
    scaling = getattr(self, "scaling", self.head_dim**-0.5)
    if not math.isclose(
        float(scaling),
        self.head_dim**-0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("the frozen BTS submission requires standard attention scaling")
    if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
        raise TypeError("the frozen BTS submission supports DynamicCache only")
    _validate_attention_mask(attention_mask)
    input_shape = hidden_states.shape[:-1]
    projected_shape = (*input_shape, -1, self.head_dim)

    with STAT_COLLECTOR.stage("forward.total", hidden_states):
        with STAT_COLLECTOR.stage("forward.qkv_proj", hidden_states):
            query = self.q_proj(hidden_states).view(projected_shape)
            key = self.k_proj(hidden_states).view(projected_shape)
            value = self.v_proj(hidden_states).view(projected_shape)

        if hasattr(self, "q_norm") and self.q_norm is not None:
            with STAT_COLLECTOR.stage("forward.qk_norm", query):
                query = self.q_norm(query)
                if hasattr(self, "k_norm") and self.k_norm is not None:
                    key = self.k_norm(key)
        elif hasattr(self, "k_norm") and self.k_norm is not None:
            with STAT_COLLECTOR.stage("forward.qk_norm", key):
                key = self.k_norm(key)

        with STAT_COLLECTOR.stage("forward.transpose", query):
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        rotary = get_rotary_fn(self.__class__.__module__)
        if rotary is None:
            raise RuntimeError(
                f"could not find apply_rotary_pos_emb in {self.__class__.__module__}"
            )
        cosine, sine = position_embeddings
        with STAT_COLLECTOR.stage("forward.rotary", query):
            query, key = rotary(query, key, cosine, sine)

        if past_key_values is not None:
            with STAT_COLLECTOR.stage("forward.cache_update", key):
                cache_kwargs = {
                    "sin": sine,
                    "cos": cosine,
                    "cache_position": cache_position,
                }
                key, value = past_key_values.update(
                    key,
                    value,
                    self.layer_idx,
                    cache_kwargs=cache_kwargs,
                )

        q_len, kv_len = query.shape[2], key.shape[2]
        num_kv_groups(query.shape[1], key.shape[1])
        if any(
            tensor.dtype != torch.bfloat16 for tensor in (query, key, value)
        ):
            raise TypeError("the frozen BTS model path requires BF16 Q/K/V")
        if q_len == 1 and past_key_values is not None:
            with STAT_COLLECTOR.stage("forward.decode_gqa", query):
                output = gqa_decode_attention(query, key, value)
        else:
            if q_len != kv_len:
                raise ValueError("sparse prefill requires q_len == kv_len")
            output = bts_prefill_attention(query, key, value)

        with STAT_COLLECTOR.stage("forward.output_reshape", output):
            output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        with STAT_COLLECTOR.stage("forward.o_proj", output):
            output = self.o_proj(output)
    return output, None


def patch_model(model_id: str, *, revision: str | None = None) -> int:
    """Patch supported Transformers attention classes for ``model_id``."""
    return _patch_attention_classes(
        bts_attention_forward,
        model_id,
        revision=revision,
    )
