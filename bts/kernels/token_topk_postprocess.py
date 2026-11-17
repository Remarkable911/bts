"""Map sorted or unordered Top-K results to exact per-row token budgets."""

import torch
import triton
import triton.language as tl


# Hybrid Top-K post-processing for compact token selection.
#
# PyTorch's CUDA Top-K remains responsible for selecting the unordered set.  For
# small power-of-two K values, this kernel sorts composite score/index keys,
# applies the row-specific budget, maps bucket-local indices back to source-token
# positions, and writes the final int32 packed-row prefix in one launch.
#
# Keeping the actual selection in ``torch.topk`` avoids the register/local-memory
# cost of maintaining a large global Top-K inside one Triton program. The sorted
# large-K fallback uses a tiled, fused index mapping. Eligible unordered large-K
# rows are handled by token_radix_select instead.


@triton.jit
def _compact_sorted_topk_kernel(
    VALUES, INDICES, BLOCKS, ROWS, COUNTS, OUTPUT,
    stride_vr, stride_vk, stride_ir, stride_ik, stride_br, stride_bb,
    stride_or, stride_ot, stride_rows, stride_counts, kv_len,
    TOPK: tl.constexpr, BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr,
):
    bucket_row = tl.program_id(0)
    rank = tl.program_id(1) * TILE + tl.arange(0, TILE)
    output_row = tl.load(ROWS + bucket_row * stride_rows).to(tl.int64)
    desired = tl.load(COUNTS + output_row * stride_counts)
    keep = (rank < TOPK) & (rank < desired)
    values = tl.load(VALUES + bucket_row * stride_vr + rank * stride_vk,
                     mask=keep, other=-float("inf")).to(tl.float32)
    keep = keep & (tl.abs(values) < float("inf"))
    indices = tl.load(INDICES + bucket_row * stride_ir + rank * stride_ik,
                      mask=keep, other=0).to(tl.int64)
    block = tl.load(BLOCKS + bucket_row * stride_br + (indices // BLOCK_SIZE) * stride_bb,
                    mask=keep, other=0).to(tl.int64)
    position = block * BLOCK_SIZE + indices % BLOCK_SIZE
    tl.store(OUTPUT + output_row * stride_or + rank * stride_ot, position.to(tl.int32),
             mask=keep & (position >= 0) & (position < kv_len))


def compact_sorted_topk_to_token_positions(
    topk_values: torch.Tensor,
    local_indices: torch.Tensor,
    bucket_blocks: torch.Tensor,
    bucket_rows: torch.Tensor,
    desired_counts: torch.Tensor,
    output: torch.Tensor,
    *,
    block_size: int,
    kv_len: int,
) -> None:
    """Map sorted Top-K directly to a sentinel-initialized output, for any K.

    Tiling ranks avoids the large per-row register footprint of sorting a long
    vector in Triton. The selected set and order remain those of torch.topk.
    """
    tensors = (topk_values, local_indices, bucket_blocks, bucket_rows, desired_counts, output)
    if not all(t.is_cuda and t.device == output.device for t in tensors):
        raise ValueError("All Top-K mapping tensors must be on the same CUDA device.")
    if topk_values.ndim != 2 or local_indices.shape != topk_values.shape:
        raise ValueError("topk_values and local_indices must have the same 2D shape.")
    if bucket_blocks.ndim != 2 or bucket_blocks.shape[0] != topk_values.shape[0]:
        raise ValueError("bucket_blocks must have one row per Top-K row.")
    if bucket_rows.ndim != 1 or bucket_rows.shape[0] != topk_values.shape[0]:
        raise ValueError("bucket_rows must have one entry per Top-K row.")
    if desired_counts.ndim != 1 or output.ndim != 2 or output.shape[0] != desired_counts.shape[0]:
        raise ValueError("output must have one row per desired count.")
    if output.shape[1] < topk_values.shape[1] or block_size <= 0 or kv_len < 0:
        raise ValueError("Invalid output capacity, block_size, or kv_len.")
    if output.dtype != torch.int32:
        raise TypeError("output must use int32 token metadata.")
    rows, topk = topk_values.shape
    if rows == 0 or topk == 0:
        return
    _compact_sorted_topk_kernel[(rows, triton.cdiv(topk, 256))](
        topk_values, local_indices, bucket_blocks, bucket_rows, desired_counts, output,
        *topk_values.stride(), *local_indices.stride(), *bucket_blocks.stride(),
        *output.stride(), bucket_rows.stride(0), desired_counts.stride(0),
        int(kv_len), TOPK=topk, BLOCK_SIZE=int(block_size), TILE=256,
        num_warps=4,
    )


@triton.jit
def _radix_budget_kernel(VALUES, INDICES, BLOCKS, ROWS, COUNTS, OUT,
                         VR: tl.constexpr, VK: tl.constexpr, IR: tl.constexpr, IK: tl.constexpr,
                         BR: tl.constexpr, BK: tl.constexpr, RR: tl.constexpr, CR: tl.constexpr,
                         OR: tl.constexpr, OK: tl.constexpr,
                         K: tl.constexpr, N: tl.constexpr, BS: tl.constexpr, KV: tl.constexpr,
                         ADAPTIVE: tl.constexpr):
    r = tl.program_id(0)
    x = tl.arange(0, N)
    row = tl.load(ROWS + r * RR).to(tl.int64)
    desired = tl.load(COUNTS + row * CR).to(tl.int32)
    values = tl.load(VALUES + r * VR + x * VK, x < K, -float("inf")).to(tl.float32)
    indices = tl.load(INDICES + r * IR + x * IK, x < K, 0).to(tl.int32)
    valid = (x < K) & (tl.abs(values) < float("inf"))
    desired = tl.minimum(tl.maximum(desired, 0), tl.sum(valid.to(tl.int32)))
    if desired == 0:
        return
    if desired == K:
        keep = valid
    else:
        # Ordered float keys, with -0 and +0 treated as a score tie.
        bits = tl.where(values == 0, 0.0, values).to(tl.uint32, bitcast=True)
        keys = bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
        active = valid
        remaining = desired
        if ADAPTIVE:
            # The highest differing bit skips common score prefixes. A 64-bin
            # histogram is enough to narrow the active range without sorting it.
            bins = tl.arange(0, 64)
            key_min = tl.min(tl.where(active, keys, 0xFFFFFFFF), 0)
            key_max = tl.max(tl.where(active, keys, 0), 0)
            active_count = tl.sum(active.to(tl.int32))
            while (key_min < key_max) & (active_count > remaining):
                leading = tl.inline_asm_elementwise("clz.b32 $0, $1;", constraints="=r,r",
                    args=[key_min ^ key_max], dtype=tl.int32, is_pure=True, pack=1)
                shift = tl.maximum(0, 31 - leading - 5)
                digits = 63 - ((keys >> shift) & 63).to(tl.int32)
                hist = tl.histogram(digits, 64, mask=active)
                prefix = tl.cumsum(hist)
                chosen = tl.min(tl.where(prefix >= remaining, bins, 64), 0)
                before = tl.sum(tl.where(bins < chosen, hist, 0), 0)
                remaining -= before
                active = active & (digits == chosen)
                active_count = tl.sum(tl.where(bins == chosen, hist, 0), 0)
                key_min = tl.min(tl.where(active, keys, 0xFFFFFFFF), 0)
                key_max = tl.max(tl.where(active, keys, 0), 0)
            # If all remaining candidates fit, their minimum is a valid exact
            # threshold. Otherwise all remaining keys tie and the index rule below
            # resolves that tie. Neither case needs another histogram.
            threshold = key_min
        else:
            threshold = tl.full((), 0, tl.uint32)
            bins = tl.arange(0, 256)
            for digit in tl.static_range(4):
                shift = 24 - digit * 8
                # Reverse the digit so an ascending prefix sum represents highest scores first.
                digits = 255 - ((keys >> shift) & 255).to(tl.int32)
                hist = tl.histogram(digits, 256, mask=active)
                prefix = tl.cumsum(hist)
                chosen = tl.min(tl.where(prefix >= remaining, bins, 256), 0)
                before = tl.sum(tl.where(bins < chosen, hist, 0), 0)
                remaining -= before
                threshold = threshold | ((255 - chosen).to(tl.uint32) << shift)
                active = active & (digits == chosen)
        # Exact score threshold found. Resolve ties by candidate index, without
        # imposing any score order on the selected output.
        better = valid & (keys > threshold)
        ties = valid & (keys == threshold)
        needed = desired - tl.sum(better.to(tl.int32))
        if tl.sum(ties.to(tl.int32)) > needed:
            # Candidate indices are unique. Find the needed-th smallest tied index.
            low = tl.min(tl.where(ties, indices, 0x7FFFFFFF), 0)
            high = tl.max(tl.where(ties, indices, 0), 0)
            while low < high:
                middle = low + (high - low) // 2
                num = tl.sum((ties & (indices <= middle)).to(tl.int32))
                high = tl.where(num >= needed, middle, high)
                low = tl.where(num < needed, middle + 1, low)
            ties = ties & (indices <= low)
        keep = better | ties
    rank = tl.cumsum(keep.to(tl.int32)) - 1
    block = tl.load(BLOCKS + r * BR + (indices // BS) * BK, keep, 0).to(tl.int64)
    positions = block * BS + indices % BS
    tl.store(OUT + row * OR + rank * OK, positions.to(tl.int32),
             keep & (positions >= 0) & (positions < KV))


def radix_budget_to_token_positions(values, indices, blocks, rows, counts, output, *, block_size, kv_len, adaptive=True):
    """Select exact budgets; output must be initialized with the KV sentinel.

    ``values`` must be an unordered Top-K superset of every row's budget.
    Output follows the superset's order, so attention accumulation can differ.
    Adaptive refinement preserves the fixed-pass result and output order.
    """
    if values.ndim != 2 or indices.shape != values.shape:
        raise ValueError("Values and indices must have matching 2D shapes.")
    tensors = (values, indices, blocks, rows, counts, output)
    if not all(t.is_cuda and t.device == values.device for t in tensors):
        raise ValueError("All tensors must use the same CUDA device.")
    if blocks.ndim != 2 or blocks.shape[0] != values.shape[0]:
        raise ValueError("Blocks must have one row per superset row.")
    if rows.ndim != 1 or rows.shape[0] != values.shape[0] or counts.ndim != 1:
        raise ValueError("Rows and counts must be row vectors.")
    if output.ndim != 2 or output.shape[0] != counts.shape[0]:
        raise ValueError("Output must have one row per count.")
    if output.dtype != torch.int32 or output.shape[1] < values.shape[1]:
        raise ValueError("Output must be int32 with sufficient row capacity.")
    if values.dtype != torch.float32 or indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Radix selection expects FP32 values and integer candidate indices.")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (blocks, rows, counts)):
        raise TypeError("Blocks, rows, and counts must be integer metadata.")
    if block_size <= 0 or kv_len < 0 or values.shape[1] > 16384:
        raise ValueError("Invalid block size, KV length, or Top-K above 16384.")
    nrows, topk = values.shape
    if not nrows or not topk:
        return
    _radix_budget_kernel[(nrows,)](values, indices, blocks, rows, counts, output,
        *values.stride(), *indices.stride(), *blocks.stride(), rows.stride(0), counts.stride(0),
        *output.stride(), K=topk, N=triton.next_power_of_2(topk), BS=block_size, KV=kv_len,
        ADAPTIVE=bool(adaptive), num_warps=8 if topk >= 2048 else 4)

