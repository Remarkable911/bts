"""Engine-facing Q/K/V entry point for the frozen BTS public API.

Inputs have already undergone Q/K normalization and RoPE. FastDeploy retains
cache ownership. Algorithm selection, padding, coarse/fine selection,
Tail256, and packed attention all remain behind ``bts_prefill_attention``.
"""

import torch

from .config import validate_algorithm


@torch.inference_mode()
def prefill_attention(query, key, value, *, config):
    """Validate the engine boundary and invoke the one submitted algorithm."""

    validate_algorithm(config)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q/K/V must have shape [batch, heads, tokens, head_dim]")
    if key.shape != value.shape or query.shape[0] != key.shape[0]:
        raise ValueError("Only full, square prefill with matching Q/K/V is supported")
    if query.shape[2:] != key.shape[2:]:
        raise ValueError("Only full, square prefill with matching Q/K/V is supported")
    if query.shape[1] % key.shape[1] or min(query.shape) <= 0:
        raise ValueError("Invalid GQA heads or empty prefill")
    if not (query.device == key.device == value.device):
        raise ValueError("Q/K/V must share a device")
    if not (query.dtype == key.dtype == value.dtype):
        raise TypeError("Q/K/V must share a dtype")
    if query.dtype != torch.bfloat16:
        raise TypeError("The frozen BTS FastDeploy path requires BF16 Q/K/V")
    if query.shape[-1] != 128:
        raise ValueError("The frozen BTS FastDeploy path requires head_dim=128")

    # Import through the public package namespace. Do not couple deployment to
    # internal selectors, masks, padding helpers, or kernels.
    from bts import bts_prefill_attention

    return bts_prefill_attention(query, key, value)
