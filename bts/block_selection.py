"""Dual-band BTS coarse block selection.

The probability-mass selector in this module is block-level Top-P and is part
of BTS itself. It is distinct from the removed legacy token-level Top-P
method.
"""

from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F

from .attention_utils import (
    expand_kv_head_tensor_to_q_heads,
    gqa_query_key_scores,
    mean_pool,
    num_kv_groups,
)
from .config import (
    BLOCK_SIZE,
    FORCE_RECENT,
    FORCE_SINK,
    HIGH_FREQ_DIM,
    HIGH_FREQ_THRESHOLD,
    LOW_FREQ_DIM,
    LOW_FREQ_THRESHOLD,
)
from .kernels.bts_estimation import (
    dual_band_softmax_triton,
    select_blocks_by_mass_triton,
)


def low_frequency_components(states: torch.Tensor) -> torch.Tensor:
    head_dim = states.shape[-1]
    half = head_dim // 2
    width = LOW_FREQ_DIM // 2
    return torch.cat(
        (states[..., half - width : half], states[..., head_dim - width :]), dim=-1
    )


def high_frequency_components(states: torch.Tensor) -> torch.Tensor:
    half = states.shape[-1] // 2
    width = HIGH_FREQ_DIM // 2
    return torch.cat((states[..., :width], states[..., half : half + width]), dim=-1)


def select_blocks_by_mass(probabilities: torch.Tensor, threshold: float) -> torch.Tensor:
    """Select the stable descending prefix whose block mass reaches threshold."""
    scores = torch.nan_to_num(probabilities, nan=0.0)
    if scores.shape[-2] == 0 or scores.shape[-1] == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_scores, dim=-1)
    selected_sorted = cumulative - sorted_scores < threshold
    selected = torch.empty_like(scores, dtype=torch.bool)
    selected.scatter_(-1, sorted_indices, selected_sorted)
    return selected


def _stage(collector, name: str, tensor: torch.Tensor):
    return nullcontext() if collector is None else collector.stage(name, tensor)


def _torch_block_union(probabilities: torch.Tensor, query_blocks: int) -> torch.Tensor:
    high, low = select_blocks_by_mass(
        probabilities, HIGH_FREQ_THRESHOLD
    ).split(query_blocks, dim=2)
    return high | low


def _bts_block_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    query_blocks: torch.Tensor,
    *,
    collector=None,
) -> torch.Tensor:
    """Return the causal union of low/high-frequency block-mass selections."""
    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("query and key states must be four-dimensional")
    if query_states.shape[2] % BLOCK_SIZE or key_states.shape[2] % BLOCK_SIZE:
        raise ValueError("coarse selection requires lengths padded to 128 tokens")
    batch, heads, q_len, head_dim = query_states.shape
    if LOW_FREQ_DIM > head_dim or HIGH_FREQ_DIM > head_dim:
        raise ValueError("frequency dimensions exceed the attention head dimension")
    num_kv_groups(heads, key_states.shape[1])

    timer_start = timer_end = None
    if collector is not None:
        timer_start, timer_end = collector.maybe_start_select_timer(query_states)
    with _stage(collector, "block_estimate.total", query_states):
        with _stage(collector, "block_estimate.pool", query_states):
            expected = (batch, heads, q_len // BLOCK_SIZE, head_dim)
            if query_blocks.shape != expected:
                raise ValueError(f"cached query means must have shape {expected}")
            if query_blocks.device != query_states.device:
                raise ValueError("cached query means must be on the query device")
            if query_blocks.dtype != query_states.dtype:
                raise TypeError("cached query means must use the query dtype")
            q_blocks = query_blocks
            k_blocks = mean_pool(key_states, BLOCK_SIZE)
            q_high = high_frequency_components(q_blocks)
            q_low = low_frequency_components(q_blocks)
            k_high = high_frequency_components(k_blocks)
            k_low = low_frequency_components(k_blocks)
            num_q_blocks = q_blocks.shape[2]
            num_k_blocks = k_blocks.shape[2]

        with _stage(collector, "block_estimate.calibrate", query_states):
            epsilon = 1e-6
            rq = torch.sqrt(torch.mean(q_blocks**2, dim=(-2, -1), keepdim=True))
            rk = torch.sqrt(torch.mean(k_blocks**2, dim=(-2, -1), keepdim=True))
            rq_high = torch.sqrt(torch.mean(q_high**2, dim=(-2, -1), keepdim=True))
            rk_high = torch.sqrt(torch.mean(k_high**2, dim=(-2, -1), keepdim=True))
            rq_low = torch.sqrt(torch.mean(q_low**2, dim=(-2, -1), keepdim=True))
            rk_low = torch.sqrt(torch.mean(k_low**2, dim=(-2, -1), keepdim=True))
            high_scale = math.sqrt(HIGH_FREQ_DIM / head_dim) * (
                rq_high / rq.clamp_min(epsilon)
            ) * expand_kv_head_tensor_to_q_heads(
                rk_high / rk.clamp_min(epsilon), heads
            )
            low_scale = math.sqrt(LOW_FREQ_DIM / head_dim) * (
                rq_low / rq.clamp_min(epsilon)
            ) * expand_kv_head_tensor_to_q_heads(
                rk_low / rk.clamp_min(epsilon), heads
            )
            high_scale = math.sqrt(HIGH_FREQ_DIM) * high_scale
            low_scale = math.sqrt(LOW_FREQ_DIM) * low_scale

        if query_states.is_cuda:
            with _stage(collector, "block_estimate.logits", query_states):
                probabilities = dual_band_softmax_triton(
                    q_high, q_low, k_high, k_low, high_scale, low_scale
                )
            with _stage(collector, "block_estimate.select", query_states):
                block_mask = select_blocks_by_mass_triton(
                    probabilities,
                    HIGH_FREQ_THRESHOLD,
                    LOW_FREQ_THRESHOLD,
                    force_sink=FORCE_SINK,
                    force_recent=FORCE_RECENT,
                )
            forcing_fused = True
        else:
            with _stage(collector, "block_estimate.logits", query_states):
                q_indices = torch.arange(num_q_blocks, device=query_states.device)[:, None]
                k_indices = torch.arange(num_k_blocks, device=query_states.device)[None, :]
                causal = k_indices <= q_indices + num_k_blocks - num_q_blocks
                causal = causal[None, None]
                logits = torch.empty(
                    (batch, heads, 2 * num_q_blocks, num_k_blocks),
                    device=query_states.device,
                )
                logits[:, :, :num_q_blocks] = (
                    gqa_query_key_scores(q_high, k_high) / high_scale
                ).masked_fill(~causal, -torch.inf)
                logits[:, :, num_q_blocks:] = (
                    gqa_query_key_scores(q_low, k_low) / low_scale
                ).masked_fill(~causal, -torch.inf)
                probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
            with _stage(collector, "block_estimate.select", query_states):
                block_mask = _torch_block_union(probabilities, num_q_blocks)
            forcing_fused = False

        if not forcing_fused:
            if FORCE_SINK and num_k_blocks:
                block_mask[..., 0] = True
            if FORCE_RECENT and num_k_blocks:
                query_index = torch.arange(num_q_blocks, device=block_mask.device)
                key_index = query_index + num_k_blocks - num_q_blocks
                block_mask[:, :, query_index, key_index] = True

    if collector is not None:
        collector.finish_select_timer(timer_start, timer_end)
        if collector.collect_density or collector.collect_stage_time:
            collector.record_block_mask(
                block_mask, record_density=collector.collect_density
            )
    return block_mask
