"""Append sink/recent positions to prepacked selected-token prefixes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _append_required_rows(
    REQUIRED,
    SELECTED_COUNTS,
    OUTPUT,
    PACKED_COUNTS,
    stride_rb,
    stride_rh,
    stride_rq,
    stride_rt,
    stride_cb,
    stride_ch,
    stride_cq,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_ot,
    required_width,
    kv_len,
    HEADS: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    TILE: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    offsets = tl.arange(0, TILE)
    in_bounds = offsets < required_width
    values = tl.load(
        REQUIRED
        + batch * stride_rb
        + head * stride_rh
        + query_block * stride_rq
        + offsets * stride_rt,
        mask=in_bounds,
        other=kv_len,
    )
    valid = in_bounds & (values >= 0) & (values < kv_len)
    ranks = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    selected_count = tl.load(
        SELECTED_COUNTS
        + batch * stride_cb
        + head * stride_ch
        + query_block * stride_cq
    )
    tl.store(
        PACKED_COUNTS + batch_head * QUERY_BLOCKS + query_block,
        selected_count + tl.sum(valid.to(tl.int32)),
    )
    tl.store(
        OUTPUT
        + batch * stride_ob
        + head * stride_oh
        + query_block * stride_oq
        + (selected_count + ranks) * stride_ot,
        values,
        mask=valid,
    )


def append_required_to_prepacked_rows_cuda(
    selected: torch.Tensor,
    selected_counts: torch.Tensor,
    required: torch.Tensor,
    *,
    kv_len: int,
    selected_count_upper_bound: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not selected.is_cuda or selected.ndim != 4:
        raise ValueError("selected rows must be a four-dimensional CUDA tensor")
    if selected_counts.shape != selected.shape[:-1]:
        raise ValueError("selected counts do not match selected rows")
    if required.shape[:-1] != selected.shape[:-1]:
        raise ValueError("required positions do not match selected rows")
    if selected_count_upper_bound + required.shape[-1] > selected.shape[-1]:
        raise ValueError("selected rows do not have enough required-token slack")
    counts = selected_counts.to(torch.int32)
    packed_counts = torch.empty_like(counts)
    batch, heads, query_blocks, _ = selected.shape
    _append_required_rows[(query_blocks, batch * heads)](
        required,
        counts,
        selected,
        packed_counts,
        *required.stride(),
        *counts.stride(),
        *selected.stride(),
        required.shape[-1],
        kv_len,
        HEADS=heads,
        QUERY_BLOCKS=query_blocks,
        TILE=triton.next_power_of_2(required.shape[-1]),
        num_warps=4,
    )
    return selected, packed_counts
