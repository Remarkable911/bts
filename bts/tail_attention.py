"""Tail256: preserve every coarse-selected edge for the final query suffix."""

import torch

from .config import BLOCK_SIZE, DENSE_TAIL_TOKENS
from .kernels.packed_attention import direct_packed_token_attention


def tail_query_start(q_len: int, valid_q_len: int | None) -> int:
    """Round the requested 256-token suffix outward to 128-token blocks."""
    valid = q_len if valid_q_len is None else valid_q_len
    if (
        isinstance(valid, bool)
        or not isinstance(valid, int)
        or not 0 < valid <= q_len
        or q_len - valid >= BLOCK_SIZE
    ):
        raise ValueError("valid_q_len may describe padding in the final block only")
    return max(0, valid - DENSE_TAIL_TOKENS) // BLOCK_SIZE * BLOCK_SIZE


def pack_bts_tail(
    block_mask: torch.Tensor,
    *,
    query_start: int,
    q_len: int,
    valid_kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand all selected coarse blocks for each tail row, without scoring."""
    batch, heads, rows, key_blocks = block_mask.shape
    device = block_mask.device
    last_query = query_start + (torch.arange(rows, device=device) + 1) * BLOCK_SIZE - 1
    last_query = last_query.clamp_max(query_start + q_len - 1)
    block_ids = torch.arange(key_blocks, device=device, dtype=torch.int32)
    live = block_mask & (
        block_ids.view(1, 1, 1, -1) * BLOCK_SIZE
        <= last_query.view(1, 1, -1, 1)
    )
    live = live & (
        block_ids.view(1, 1, 1, -1) * BLOCK_SIZE < valid_kv_len
    )
    ranks = live.to(torch.long).cumsum(-1) - 1
    destinations = torch.where(live, ranks, key_blocks)
    sources = torch.where(live, block_ids.view(1, 1, 1, -1), key_blocks)
    compact = torch.full(
        (batch, heads, rows, key_blocks + 1),
        key_blocks,
        device=device,
        dtype=torch.int32,
    )
    compact.scatter_(-1, destinations, sources)
    offsets = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int32)
    positions = compact[..., :key_blocks, None] * BLOCK_SIZE + offsets
    positions = positions.reshape(batch, heads, rows, key_blocks * BLOCK_SIZE)
    positions = positions.clamp_max(valid_kv_len)
    counts = (positions < valid_kv_len).sum(-1, dtype=torch.int32)
    return positions.reshape(batch, heads, -1), counts


def _reference_tail(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: torch.Tensor,
    start: int,
    valid_kv_len: int,
) -> torch.Tensor:
    output = torch.zeros_like(query)
    groups = query.shape[1] // key.shape[1]
    keys = torch.arange(valid_kv_len, device=query.device)
    scale = query.shape[-1] ** -0.5
    for batch in range(query.shape[0]):
        for head in range(query.shape[1]):
            for row in range(block_mask.shape[2]):
                lo = row * BLOCK_SIZE
                hi = min(lo + BLOCK_SIZE, query.shape[2])
                allowed = block_mask[batch, head, row].repeat_interleave(BLOCK_SIZE)
                positions = keys[allowed[:valid_kv_len]]
                if not positions.numel():
                    continue
                scores = query[batch, head, lo:hi].float() @ key[
                    batch, head // groups, positions
                ].float().T
                query_positions = start + torch.arange(lo, hi, device=query.device)
                scores = (scores * scale).masked_fill(
                    positions[None, :] > query_positions[:, None], -torch.inf
                )
                probabilities = scores.softmax(-1).nan_to_num()
                output[batch, head, lo:hi] = (
                    probabilities @ value[batch, head // groups, positions].float()
                ).to(query.dtype)
    return output


def tail_preserving_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    valid_q_len: int,
    query_means: torch.Tensor,
    prefix_attention,
    collector,
) -> torch.Tensor:
    """Run block-count/mean128 on the prefix and preserve tail candidates."""
    if query.shape[2] != key.shape[2]:
        raise ValueError("Tail256 requires square causal prefill")
    length = query.shape[2]
    valid = length if valid_q_len is None else valid_q_len
    start = tail_query_start(length, valid_q_len)
    rows = length // BLOCK_SIZE
    if block_mask.shape != (query.shape[0], query.shape[1], rows, rows):
        raise ValueError("block_mask does not match the padded square prefill")

    output = torch.empty_like(query)
    prefix_statistics: dict[str, torch.Tensor] = {}
    if start:
        prefix_means = query_means[..., : start // BLOCK_SIZE, :]
        with collector.stage("token_sparse.prefix", query):
            output[:, :, :start] = prefix_attention(
                query[:, :, :start],
                key[:, :, :start],
                value[:, :, :start],
                block_mask[:, :, : start // BLOCK_SIZE, : start // BLOCK_SIZE],
                query_means=prefix_means,
                packed_statistics=prefix_statistics,
            )

    suffix_query = query[:, :, start:]
    suffix_mask = block_mask[:, :, start // BLOCK_SIZE :, :]
    with collector.stage("token_sparse.tail.layout", query):
        packed, counts = pack_bts_tail(
            suffix_mask,
            query_start=start,
            q_len=length - start,
            valid_kv_len=valid,
        )
    with collector.stage("token_sparse.tail.attention", query):
        if query.is_cuda:
            suffix_output = direct_packed_token_attention(
                suffix_query,
                key,
                value,
                packed,
                counts,
                source_kv_len=valid,
                q_offset=start,
            )
        else:
            suffix_output = _reference_tail(
                suffix_query, key, value, suffix_mask, start, valid
            )
        output[:, :, start:] = suffix_output

    collector.record_stage_value("token_sparse.tail.requested_tokens", DENSE_TAIL_TOKENS)
    collector.record_stage_value("token_sparse.tail.protected_tokens", valid - start)
    collector.record_stage_value("token_sparse.tail.query_blocks", rows - start // BLOCK_SIZE)
    if collector.collect_density or collector.collect_stage_time:
        tail_packed_blocks = torch.div(
            counts + BLOCK_SIZE - 1,
            BLOCK_SIZE,
            rounding_mode="floor",
        ).sum()
        total_packed_blocks = tail_packed_blocks + prefix_statistics.get(
            "packed_blocks", 0
        )
        total_causal_blocks = (
            query.shape[0] * query.shape[1] * rows * (rows + 1) // 2
        )
        final_density = float(total_packed_blocks.item()) / total_causal_blocks
        if collector.collect_density:
            collector.replace_latest_density(final_density)
        collector.record_stage_value(
            "token_sparse.final_block_density", final_density
        )
        collector.record_stage_value(
            "token_sparse.tail.combined_density", final_density
        )
    return output
