"""Leak-free vertical/slash index conversion for MInference.

MInference 0.1.6 ships this conversion as a custom CUDA extension.  The
extension used by the LongBench environment retains the four output tensors
from every invocation, which costs roughly 5.6 GiB per 32K request on Qwen3.
This Triton translation follows the extension's merge algorithm exactly while
returning ordinary PyTorch tensors whose storage can be reused between heads.

The upstream MInference operator is MIT licensed; see the bundled notice.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["N_HEADS", "N_ROWS", "NNZ_V", "NNZ_S"]
)
def _convert_vertical_slash_indexes_kernel(
    SEQLENS,
    VERTICAL,
    SLASH,
    BLOCK_COUNT,
    BLOCK_OFFSET,
    COLUMN_COUNT,
    COLUMN_INDEX,
    N_HEADS,
    N_ROWS,
    NNZ_V,
    NNZ_S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_idx_m = row_id % N_ROWS
    batch_head = row_id // N_ROWS
    batch_idx = batch_head // N_HEADS

    seqlen = tl.load(SEQLENS + batch_idx)
    start_m = block_idx_m * BLOCK_M
    if start_m < seqlen:
        end_m = start_m + BLOCK_M
        vertical_base = batch_head * NNZ_V
        slash_base = batch_head * NNZ_S
        row_base = batch_head * N_ROWS + block_idx_m

        v = 1
        s = 1
        v_idx = tl.load(VERTICAL + vertical_base)
        s_idx = tl.load(SLASH + slash_base)

        # Valid MInference slash sets always contain a value below every active
        # query-row endpoint.  Retain the native kernel's ordering while
        # guarding malformed inputs from an out-of-bounds read.
        while (s_idx >= end_m) & (s < NNZ_S):
            s_idx = tl.load(SLASH + slash_base + s)
            s += 1
        s_idx = tl.maximum(end_m - s_idx, BLOCK_M)
        range_start = s_idx - BLOCK_M
        range_end = s_idx

        tmp_col_cnt = 0
        tmp_blk_cnt = 0
        running = 1
        while running != 0:
            if v_idx < range_end:
                if v_idx < range_start:
                    tl.store(
                        COLUMN_INDEX + row_base * NNZ_V + tmp_col_cnt,
                        v_idx,
                    )
                    tmp_col_cnt += 1
                if v < NNZ_V:
                    v_idx = tl.load(VERTICAL + vertical_base + v)
                    v += 1
                else:
                    v_idx = end_m + BLOCK_M
            else:
                if s < NNZ_S:
                    s_idx = tl.maximum(
                        end_m - tl.load(SLASH + slash_base + s), BLOCK_M
                    )
                    s += 1
                    if s_idx > range_end + BLOCK_M:
                        idx = range_start
                        while idx < range_end:
                            tl.store(
                                BLOCK_OFFSET + row_base * NNZ_S + tmp_blk_cnt,
                                idx,
                            )
                            tmp_blk_cnt += 1
                            idx += BLOCK_N
                        range_start = s_idx - BLOCK_M
                        range_end = s_idx
                    elif s_idx > range_end:
                        range_end += BLOCK_M
                else:
                    idx = range_start
                    while idx < range_end:
                        tl.store(
                            BLOCK_OFFSET + row_base * NNZ_S + tmp_blk_cnt,
                            idx,
                        )
                        tmp_blk_cnt += 1
                        idx += BLOCK_N
                    running = 0

        tl.store(BLOCK_COUNT + row_base, tmp_blk_cnt)
        tl.store(COLUMN_COUNT + row_base, tmp_col_cnt)


def convert_vertical_slash_indexes(
    seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    context_size: int,
    block_size_m: int,
    block_size_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match ``minference.cuda.convert_vertical_slash_indexes`` without its leak."""

    if block_size_m != 64 or block_size_n != 64:
        raise ValueError("MInference vertical/slash conversion requires 64x64 blocks")
    if not isinstance(context_size, int) or context_size <= 0:
        raise ValueError("context_size must be a positive integer")
    if seqlens.ndim != 1 or vertical_indexes.ndim != 3 or slash_indexes.ndim != 3:
        raise ValueError("Expected seqlens [B] and vertical/slash indexes [B,H,K]")
    batch_size, num_heads, nnz_vertical = vertical_indexes.shape
    slash_batch, slash_heads, nnz_slash = slash_indexes.shape
    if (slash_batch, slash_heads) != (batch_size, num_heads):
        raise ValueError("Vertical and slash batch/head dimensions must match")
    if seqlens.shape[0] != batch_size:
        raise ValueError("seqlens batch dimension must match the index tensors")
    if nnz_vertical <= 0 or nnz_slash <= 0:
        raise ValueError("Vertical and slash index sets must be non-empty")
    tensors = (seqlens, vertical_indexes, slash_indexes)
    if not all(t.is_cuda and t.device == seqlens.device for t in tensors):
        raise ValueError("MInference index tensors must share one CUDA device")
    if not all(t.dtype == torch.int32 for t in tensors):
        raise TypeError("MInference index tensors must use torch.int32")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("MInference index tensors must be contiguous")

    num_rows = triton.cdiv(context_size, block_size_m)
    options = {"dtype": seqlens.dtype, "device": seqlens.device}
    block_count = torch.zeros((batch_size, num_heads, num_rows), **options)
    block_offset = torch.zeros(
        (batch_size, num_heads, num_rows, nnz_slash), **options
    )
    column_count = torch.zeros((batch_size, num_heads, num_rows), **options)
    column_index = torch.zeros(
        (batch_size, num_heads, num_rows, nnz_vertical), **options
    )
    _convert_vertical_slash_indexes_kernel[(batch_size * num_heads * num_rows,)](
        seqlens,
        vertical_indexes,
        slash_indexes,
        block_count,
        block_offset,
        column_count,
        column_index,
        N_HEADS=num_heads,
        N_ROWS=num_rows,
        NNZ_V=nnz_vertical,
        NNZ_S=nnz_slash,
        BLOCK_M=block_size_m,
        BLOCK_N=block_size_n,
        num_warps=1,
        num_stages=1,
    )
    return block_count, block_offset, column_count, column_index
