"""Frozen mean128/block-count-0.4 token selection and packed attention."""

import math

import torch
import torch.nn.functional as F

from .attention_utils import mean_pool
from .block_selection import _bts_block_estimate
from .config import BLOCK_SIZE
from .kernels.compact_candidates import count_candidate_tokens
from .kernels.packed_attention import (
    _fixed_prefix_packed_token_attention,
    direct_packed_token_attention,
)
from .token_budget import block_count_budget
from .token_packing import PackedTokenLayout, finalize_layout, required_sink_recent
from .token_scoring import _query_block_means
from .token_selectors import select_tokens
from .utils.stat_collector import StatCollector


STAT_COLLECTOR = StatCollector.from_env()
MAX_VALIDATED_SEQUENCE_LENGTH = 131072


def _can_use_fixed_prefix_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: PackedTokenLayout,
) -> bool:
    """Accept only the verified 1x32Q/8KV/D128 fixed-prefix layout."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    q_len = query.shape[2]
    if (
        q_len <= 0
        or q_len % BLOCK_SIZE
        or not (query.is_cuda and key.is_cuda and value.is_cuda)
        or not layout.packed_k_indices.is_cuda
        or not layout.packed_token_counts.is_cuda
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or query.shape != (1, 32, q_len, 128)
        or key.shape != (1, 8, q_len, 128)
        or value.shape != key.shape
        or layout.packed_k_indices.dtype != torch.int32
        or layout.packed_token_counts.dtype != torch.int32
        or not layout.packed_k_indices.is_contiguous()
        or not layout.packed_token_counts.is_contiguous()
    ):
        return False
    if not (
        query.device == key.device == value.device
        and layout.packed_k_indices.device == query.device
        and layout.packed_token_counts.device == query.device
    ):
        return False
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        return False
    query_blocks = q_len // BLOCK_SIZE
    if (
        layout.packed_k_indices.ndim != 3
        or layout.packed_k_indices.shape[:2] != query.shape[:2]
        or layout.packed_k_indices.shape[-1] % query_blocks
        or layout.packed_token_counts.shape
        != (query.shape[0], query.shape[1], query_blocks)
    ):
        return False
    tokens_per_row = layout.packed_k_indices.shape[-1] // query_blocks
    return tokens_per_row > 0 and tokens_per_row % BLOCK_SIZE == 0


def _validate_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, tokens, dim]")
    if not all(tensor.device == query.device for tensor in (key, value)):
        raise ValueError("query, key, and value must share one device")
    if key.shape != value.shape:
        raise ValueError("key and value must have matching shapes")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise TypeError("query, key, and value must have matching dtypes")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("query, key, and value must use float16, bfloat16, or float32")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key batch/head dimensions are incompatible")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if query.shape[-1] != 128:
        raise ValueError("the frozen submission supports head_dim=128 only")
    if query.shape[2] != key.shape[2]:
        raise ValueError("the frozen method supports square causal prefill only")
    if query.shape[2] <= 0:
        raise ValueError("prefill tensors must contain at least one token")
    if query.shape[2] > MAX_VALIDATED_SEQUENCE_LENGTH:
        raise ValueError("the submitted direct-attention kernel is validated up to 128K tokens")


def _validate_padded_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: torch.Tensor,
) -> None:
    _validate_qkv(query, key, value)
    if block_mask.device != query.device:
        raise ValueError("block_mask must share the Q/K/V device")
    if query.shape[2] % BLOCK_SIZE:
        raise ValueError("internal prefill tensors must be padded to a multiple of 128")
    blocks = query.shape[2] // BLOCK_SIZE
    expected = (query.shape[0], query.shape[1], blocks, blocks)
    if block_mask.shape != expected or block_mask.dtype != torch.bool:
        raise ValueError(f"block_mask must be boolean with shape {expected}")


def ordinary_candidate_mask(
    block_mask: torch.Tensor,
    q_last_positions: torch.Tensor,
) -> torch.Tensor:
    """Remove required sink/recent blocks before applying the 0.4 budget."""
    rows, key_blocks = block_mask.shape[-2:]
    ordinary = block_mask.clone()
    if key_blocks == 0:
        return ordinary
    ordinary[..., 0] = False
    recent = torch.div(q_last_positions, BLOCK_SIZE, rounding_mode="floor")
    valid = (recent >= 0) & (recent < key_blocks)
    safe = recent.clamp(0, key_blocks - 1)
    index = safe.view(1, 1, rows, 1).expand(*ordinary.shape[:-1], 1)
    ordinary.scatter_(-1, index, ordinary.gather(-1, index) & ~valid.view(1, 1, rows, 1))
    return ordinary


def _cpu_layout(
    query: torch.Tensor,
    key: torch.Tensor,
    block_mask: torch.Tensor,
    means: torch.Tensor,
    q_last: torch.Tensor,
) -> PackedTokenLayout:
    batch, heads = query.shape[:2]
    groups = heads // key.shape[1]
    rows = block_mask.shape[2]
    ordinary = ordinary_candidate_mask(block_mask, q_last)
    selected_rows: list[tuple[int, int, int, torch.Tensor]] = []
    maximum = 0
    offsets = torch.arange(BLOCK_SIZE, device=query.device)
    for batch_index in range(batch):
        for head in range(heads):
            kv_head = head // groups
            for row in range(rows):
                blocks = torch.nonzero(ordinary[batch_index, head, row], as_tuple=False).flatten()
                positions = (blocks[:, None] * BLOCK_SIZE + offsets[None, :]).flatten()
                positions = positions[(positions < key.shape[2]) & (positions <= q_last[row])]
                budget = min(
                    positions.numel(),
                    max(1, math.ceil(blocks.numel() * BLOCK_SIZE * 0.4))
                    if blocks.numel()
                    else 0,
                )
                if budget:
                    scores = means[batch_index, head, row] @ key[batch_index, kv_head, positions].float().T
                    indices = torch.topk(scores, k=budget, largest=True, sorted=True).indices
                    positions = positions.index_select(0, indices)
                else:
                    positions = positions[:0]
                selected_rows.append((batch_index, head, row, positions))
                maximum = max(maximum, positions.numel())

    capacity = max(BLOCK_SIZE, math.ceil((maximum + 2 * BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE)
    selected = torch.full(
        (batch, heads, rows, capacity),
        key.shape[2],
        device=query.device,
        dtype=torch.long,
    )
    counts = torch.zeros((batch, heads, rows), device=query.device, dtype=torch.int32)
    for batch_index, head, row, positions in selected_rows:
        selected[batch_index, head, row, : positions.numel()] = positions
        counts[batch_index, head, row] = positions.numel()
    required = required_sink_recent(block_mask, q_last, kv_len=key.shape[2])
    return finalize_layout(
        selected,
        counts,
        required,
        kv_len=key.shape[2],
    )


def _build_token_layout(
    query: torch.Tensor,
    key: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    query_means: torch.Tensor,
) -> PackedTokenLayout:
    """Build the internal mean128 + block-count-0.4 token layout."""
    dummy_value = key
    _validate_padded_inputs(query, key, dummy_value, block_mask)
    query_blocks = _query_block_means(query, query_means)
    if not query.is_cuda:
        return _cpu_layout(
            query,
            key,
            block_mask,
            query_blocks.means,
            query_blocks.last_positions,
        )

    with STAT_COLLECTOR.stage("token_sparse.layout.candidate_mask", query):
        ordinary = ordinary_candidate_mask(block_mask, query_blocks.last_positions).contiguous()
        block_counts, token_counts = count_candidate_tokens(
            ordinary,
            query_blocks.last_positions,
            kv_len=key.shape[2],
        )
        budgets = block_count_budget(block_counts, token_counts)

    selected = select_tokens(
        query_blocks.means,
        key,
        ordinary,
        block_counts,
        budgets,
        query_blocks.last_positions,
        collector=STAT_COLLECTOR,
    )
    with STAT_COLLECTOR.stage("token_sparse.layout.required_indices", query):
        required = required_sink_recent(
            block_mask, query_blocks.last_positions, kv_len=key.shape[2]
        )
    with STAT_COLLECTOR.stage("token_sparse.layout.pack_indices", query):
        return finalize_layout(
            selected,
            budgets,
            required,
            kv_len=key.shape[2],
        )


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: PackedTokenLayout,
) -> torch.Tensor:
    output = torch.zeros_like(query)
    groups = query.shape[1] // key.shape[1]
    tokens_per_row = layout.tokens_per_row
    scale = query.shape[-1] ** -0.5
    for batch in range(query.shape[0]):
        for head in range(query.shape[1]):
            for row in range(layout.packed_token_counts.shape[-1]):
                q_start = row * BLOCK_SIZE
                q_end = min(q_start + BLOCK_SIZE, query.shape[2])
                count = int(layout.packed_token_counts[batch, head, row])
                start = row * tokens_per_row
                positions = layout.packed_k_indices[batch, head, start : start + count].long()
                if not count:
                    continue
                logits = query[batch, head, q_start:q_end].float() @ key[
                    batch, head // groups, positions
                ].float().T
                q_positions = torch.arange(
                    q_start, q_end, device=query.device
                ) + max(key.shape[2] - query.shape[2], 0)
                logits = (logits * scale).masked_fill(
                    positions[None, :] > q_positions[:, None], -torch.inf
                )
                probabilities = logits.softmax(-1).nan_to_num()
                output[batch, head, q_start:q_end] = (
                    probabilities @ value[batch, head // groups, positions].float()
                ).to(query.dtype)
    return output


def _prefix_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    query_means: torch.Tensor,
    packed_statistics: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run the no-tail prefix core; Tail dispatch deliberately lives above it."""
    _validate_padded_inputs(query, key, value, block_mask)
    with STAT_COLLECTOR.stage("token_sparse.layout", query):
        layout = _build_token_layout(
            query,
            key,
            block_mask,
            query_means=query_means,
        )
    if STAT_COLLECTOR.collect_density or STAT_COLLECTOR.collect_stage_time:
        packed_statistics["packed_blocks"] = torch.div(
            layout.packed_token_counts + BLOCK_SIZE - 1,
            BLOCK_SIZE,
            rounding_mode="floor",
        ).sum()
    if not query.is_cuda:
        return _reference_attention(query, key, value, layout)
    with STAT_COLLECTOR.stage("direct_packed_attn.kernel", query):
        if _can_use_fixed_prefix_kernel(query, key, value, layout):
            return _fixed_prefix_packed_token_attention(
                query,
                key,
                value,
                layout.packed_k_indices,
                layout.packed_token_counts,
                source_kv_len=key.shape[2],
            )
        return direct_packed_token_attention(
            query,
            key,
            value,
            layout.packed_k_indices,
            layout.packed_token_counts,
            source_kv_len=key.shape[2],
            q_offset=0,
        )


def _token_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    valid_q_len: int,
    query_means: torch.Tensor,
) -> torch.Tensor:
    """Run the fixed internal layout and Tail256 stages."""
    _validate_padded_inputs(query, key, value, block_mask)
    from .tail_attention import tail_preserving_attention

    return tail_preserving_attention(
        query,
        key,
        value,
        block_mask,
        valid_q_len=valid_q_len,
        query_means=query_means,
        prefix_attention=_prefix_sparse_attention,
        collector=STAT_COLLECTOR,
    )


def bts_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run the submission's one end-to-end sparse-prefill method."""
    _validate_qkv(query, key, value)
    if query.dtype != torch.bfloat16:
        raise TypeError("the frozen sparse-prefill path requires BF16 Q/K/V")
    valid_q_len = query.shape[2]
    padding = (-valid_q_len) % BLOCK_SIZE
    with STAT_COLLECTOR.stage("forward.prefill_pad", query):
        if padding:
            query = F.pad(query, (0, 0, 0, padding), value=0.0)
            key = F.pad(key, (0, 0, 0, padding), value=0.0)
            value = F.pad(value, (0, 0, 0, padding), value=0.0)

    with STAT_COLLECTOR.stage("forward.prefill_q_block_mean_cache", query):
        query_means = mean_pool(query, BLOCK_SIZE)
    block_mask = _bts_block_estimate(
        query,
        key,
        query_means,
        collector=STAT_COLLECTOR,
    )
    with STAT_COLLECTOR.stage("token_sparse.total", query):
        output = _token_sparse_attention(
            query,
            key,
            value,
            block_mask,
            valid_q_len=valid_q_len,
            query_means=query_means,
        )
    return output[:, :, :valid_q_len] if padding else output
