"""Exact compact-row Top-K with an unordered, finite packed output prefix.

Scores are finite FP32 values or -inf masks. Candidate block IDs must be unique
within each row; masked candidate slots must have -inf scores. Output is int32,
preinitialized to kv_len. Row IDs are unique valid output rows. No host/device
synchronization or score quantization is used here.
"""

import torch
import triton
import triton.language as tl


# Keep N static: it lets the compiler prune inactive reduction lanes. The
# budget and sentinel vary independently and need not specialize the kernel.
@triton.jit(do_not_specialize=["K", "KV"])
def _direct_radix_kernel(
    SCORES, BLOCKS, ROWS, BUDGETS, OUTPUT,
    sr, sx, br, bx, rr, dr, outr, outx,
    N: tl.constexpr, TILE: tl.constexpr, K,
    VARIABLE_K: tl.constexpr, BS: tl.constexpr, KV,
):
    row = tl.program_id(0)
    output_row = tl.load(ROWS + row * rr).to(tl.int64)
    x = tl.arange(0, TILE)
    value = tl.load(SCORES + row * sr + x * sx, x < N, -float("inf"))
    valid = (x < N) & (tl.abs(value) < float("inf"))
    count = K
    if VARIABLE_K:
        count = tl.minimum(tl.maximum(tl.load(BUDGETS + output_row * dr), 0), K)
    finite_count = tl.sum(valid.to(tl.int32))
    count = tl.minimum(count, finite_count)
    keep = valid & (count > 0)
    if (count > 0) & (count < finite_count):
        # Monotonic FP32 bit keys; +0 and -0 have the same score.
        bits = tl.where(value == 0, 0., value).to(tl.uint32, bitcast=True)
        keys = bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
        active = valid
        remaining = count
        bins = tl.arange(0, 64)
        lo = tl.min(tl.where(active, keys, 0xFFFFFFFF), 0)
        hi = tl.max(tl.where(active, keys, 0), 0)
        active_count = finite_count
        while (lo < hi) & (active_count > remaining):
            leading = tl.inline_asm_elementwise(
                "clz.b32 $0, $1;", constraints="=r,r", args=[lo ^ hi],
                dtype=tl.int32, is_pure=True, pack=1,
            )
            shift = tl.maximum(0, 31 - leading - 5)
            digit = 63 - ((keys >> shift) & 63).to(tl.int32)
            histogram = tl.histogram(digit, 64, mask=active)
            prefix = tl.cumsum(histogram)
            chosen = tl.min(tl.where(prefix >= remaining, bins, 64), 0)
            remaining -= tl.sum(tl.where(bins < chosen, histogram, 0), 0)
            active &= digit == chosen
            active_count = tl.sum(tl.where(bins == chosen, histogram, 0), 0)
            lo = tl.min(tl.where(active, keys, 0xFFFFFFFF), 0)
            hi = tl.max(tl.where(active, keys, 0), 0)
        better = valid & (keys > lo)
        ties = valid & (keys == lo)
        needed = count - tl.sum(better.to(tl.int32))
        # Boundary ties prefer lower candidate slot, with no atomic ordering.
        keep = better | (ties & (tl.cumsum(ties.to(tl.int32)) <= needed))
    rank = tl.cumsum(keep.to(tl.int32)) - 1
    block = tl.load(BLOCKS + row * br + (x // BS) * bx, keep, 0).to(tl.int64)
    position = block * BS + x % BS
    tl.store(OUTPUT + output_row * outr + rank * outx, position.to(tl.int32), keep)


def _validate(values, blocks, rows, output, k, block_size, kv_len):
    if not all(t.is_cuda and t.device == output.device for t in (values, blocks, rows, output)):
        raise ValueError("Top-K tensors must share a CUDA device.")
    if values.ndim != 2 or blocks.ndim != 2 or blocks.shape[0] != values.shape[0]:
        raise ValueError("Scores and candidate blocks must have matching rows.")
    if rows.ndim != 1 or rows.numel() != values.shape[0] or output.ndim != 2:
        raise ValueError("Expected one output row ID per score row and a 2D output.")
    if output.dtype != torch.int32 or rows.dtype not in (torch.int32, torch.int64):
        raise TypeError("Output must be int32 and row IDs must be integers.")
    if blocks.dtype not in (torch.int32, torch.int64):
        raise TypeError("Candidate block IDs must be integers.")
    if not 0 <= k <= min(values.shape[1], output.shape[1]) or block_size <= 0 or kv_len < 0:
        raise ValueError("Invalid K, output capacity, block size, or sequence length.")


def direct_topk_to_token_positions(
    scores, bucket_blocks, bucket_rows, desired_counts, output, *,
    topk, block_size, kv_len,
):
    """Select exact FP32 global Top-K and map in one CTA per compact score row.

    desired_counts=None uses uniform topk; otherwise budgets index global output
    rows and are clipped to [0, topk]. Only selected prefixes are written. The
    caller must initialize output with kv_len, including all unused tail slots.
    This bounded-row implementation deliberately rejects N > 8192.
    """
    _validate(scores, bucket_blocks, bucket_rows, output, topk, block_size, kv_len)
    if scores.dtype != torch.float32:
        raise TypeError("Direct radix selection requires FP32 scores.")
    r, n = scores.shape
    if n > 8192 or bucket_blocks.shape[1] * block_size < n:
        raise ValueError("Direct radix requires N <= 8192 and a block for every score slot.")
    if desired_counts is not None:
        if (desired_counts.device != scores.device or desired_counts.ndim != 1
                or desired_counts.numel() != output.shape[0]
                or desired_counts.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Budgets must be an integer vector covering all output rows.")
    if r == 0 or topk == 0:
        return
    _direct_radix_kernel[(r,)](
        scores, bucket_blocks, bucket_rows,
        bucket_rows if desired_counts is None else desired_counts, output,
        *scores.stride(), *bucket_blocks.stride(), bucket_rows.stride(0),
        0 if desired_counts is None else desired_counts.stride(0), *output.stride(),
        N=n, TILE=triton.next_power_of_2(n), K=topk,
        VARIABLE_K=desired_counts is not None, BS=block_size, KV=kv_len,
        num_warps=8 if n >= 2048 else 4,
    )


