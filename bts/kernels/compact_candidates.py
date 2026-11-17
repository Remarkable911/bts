"""CUDA compaction, counting, and stable buckets for Top-K candidates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_rows(mask, ids, counts, columns: tl.constexpr, width: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, width)
    selected = tl.load(mask + row * columns + offsets, offsets < columns, other=0)
    selected = selected.to(tl.int32)
    ranks = tl.cumsum(selected) - 1
    tl.store(ids + row * columns + ranks, offsets, (offsets < columns) & (selected != 0))
    tl.store(counts + row, tl.sum(selected))


def compact_candidate_rows(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not mask.is_cuda or mask.dtype != torch.bool or not mask.is_contiguous():
        raise ValueError("candidate mask must be contiguous boolean CUDA memory")
    ids = torch.empty(mask.shape, dtype=torch.int32, device=mask.device)
    counts = torch.empty(mask.shape[:-1], dtype=torch.int32, device=mask.device)
    if counts.numel():
        _compact_rows[(counts.numel(),)](
            mask,
            ids,
            counts,
            mask.shape[-1],
            triton.next_power_of_2(max(1, mask.shape[-1])),
        )
    return ids, counts


@triton.jit(do_not_specialize=["columns", "bucket_width"])
def _gather_bucket(ids, counts, rows, output, columns, bucket_width, tile: tl.constexpr):
    bucket_row = tl.program_id(0).to(tl.int64)
    source_row = tl.load(rows + bucket_row).to(tl.int64)
    offsets = tl.arange(0, tile)
    count = tl.load(counts + source_row)
    values = tl.load(
        ids + source_row * columns + offsets,
        (offsets < count) & (offsets < bucket_width),
        other=columns,
    )
    tl.store(output + bucket_row * bucket_width + offsets, values, offsets < bucket_width)


def gather_candidate_bucket(
    ids: torch.Tensor,
    counts: torch.Tensor,
    rows: torch.Tensor,
    width: int,
) -> torch.Tensor:
    result = torch.empty((rows.numel(), width), dtype=torch.int64, device=ids.device)
    if rows.numel():
        _gather_bucket[(rows.numel(),)](
            ids,
            counts,
            rows,
            result,
            ids.shape[-1],
            width,
            triton.next_power_of_2(width),
        )
    return result


@triton.jit
def _fused_short_select(
    query, key, rows, blocks, q_last, budgets, output,
    qs0: tl.constexpr, qs1: tl.constexpr, qs2: tl.constexpr, qs3: tl.constexpr,
    ks0: tl.constexpr, ks1: tl.constexpr, ks2: tl.constexpr, ks3: tl.constexpr,
    heads: tl.constexpr, query_blocks: tl.constexpr, groups: tl.constexpr,
    kv_len: tl.constexpr, block_width: tl.constexpr, capacity,
    head_dim: tl.constexpr, dim_tile: tl.constexpr, token_width: tl.constexpr,
):
    bucket_row = tl.program_id(0).to(tl.int64)
    row = tl.load(rows + bucket_row).to(tl.int64)
    query_block = row % query_blocks
    head = row // query_blocks % heads
    batch = row // (query_blocks * heads)
    tokens = tl.arange(0, token_width)
    dims = tl.arange(0, dim_tile)
    block = tl.load(blocks + bucket_row * block_width + tokens // 128)
    positions = block * 128 + tokens % 128
    valid = (positions >= 0) & (positions < kv_len)
    valid = valid & (positions <= tl.load(q_last + query_block))
    q = tl.load(
        query + batch * qs0 + head * qs1 + query_block * qs2 + dims * qs3,
        dims < head_dim,
        other=0.0,
    ).to(tl.float32)
    keys = tl.load(
        key + batch * ks0 + (head // groups) * ks1
        + positions[:, None].to(tl.int64) * ks2 + dims[None, :] * ks3,
        valid[:, None] & (dims[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(keys * q[None, :], axis=1)
    scores = tl.where(valid, scores, -float("inf"))
    # Descending score, then ascending source position for deterministic ties.
    bits = scores.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    composite = (ordered.to(tl.uint64) << 32) | (
        0xFFFFFFFF - positions.to(tl.uint32)
    ).to(tl.uint64)
    ranked = tl.sort(composite, descending=True)
    selected = (0xFFFFFFFF - (ranked & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    budget = tl.load(budgets + row)
    tl.store(output + row * capacity + tokens, selected, (tokens < budget) & (tokens < capacity))


def fused_short_token_select(
    query: torch.Tensor,
    key: torch.Tensor,
    rows: torch.Tensor,
    blocks: torch.Tensor,
    q_last: torch.Tensor,
    budgets: torch.Tensor,
    output: torch.Tensor,
) -> None:
    width = blocks.shape[1] * 128
    _fused_short_select[(rows.numel(),)](
        query, key, rows, blocks, q_last, budgets, output,
        *query.stride(), *key.stride(), query.shape[1], query.shape[2],
        query.shape[1] // key.shape[1], key.shape[2], blocks.shape[1],
        output.shape[-1], query.shape[-1], triton.next_power_of_2(query.shape[-1]),
        width, num_warps=8 if width >= 512 else 4,
    )


@triton.jit
def _count_candidates(
    mask, q_last, block_counts, token_counts,
    key_blocks: tl.constexpr, query_blocks: tl.constexpr,
    kv_len: tl.constexpr, tile: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, tile)
    selected = tl.load(mask + row * key_blocks + columns, columns < key_blocks, other=0)
    selected = selected.to(tl.int64)
    starts = columns.to(tl.int64) * 128
    ends = tl.minimum(starts + 128, kv_len)
    ends = tl.minimum(ends, tl.load(q_last + row % query_blocks) + 1)
    valid_tokens = tl.minimum(tl.maximum(ends - starts, 0), 128)
    tl.store(block_counts + row, tl.sum(selected, axis=0))
    tl.store(token_counts + row, tl.sum(selected * valid_tokens, axis=0))


def count_candidate_tokens(
    mask: torch.Tensor,
    q_last_positions: torch.Tensor,
    *,
    kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not mask.is_cuda or mask.dtype != torch.bool or not mask.is_contiguous():
        raise ValueError("candidate mask must be contiguous boolean CUDA memory")
    if q_last_positions.shape != (mask.shape[2],) or q_last_positions.dtype != torch.long:
        raise ValueError("q_last_positions must contain one int64 value per query block")
    blocks = torch.empty(mask.shape[:-1], dtype=torch.int64, device=mask.device)
    tokens = torch.empty_like(blocks)
    if blocks.numel():
        _count_candidates[(blocks.numel(),)](
            mask, q_last_positions, blocks, tokens, mask.shape[-1], mask.shape[-2],
            kv_len, triton.next_power_of_2(max(1, mask.shape[-1])),
            num_warps=4, enable_fp_fusion=False,
        )
    return blocks, tokens


def candidate_bucket_widths(num_blocks: int, subdivisions: int) -> list[int]:
    if num_blocks < 1 or subdivisions not in (1, 2, 4):
        raise ValueError("require positive block count and 1, 2, or 4 subdivisions")
    maximum = (num_blocks - 1).bit_length()
    widths = [1 << exponent for exponent in range(min(maximum, 6) + 1)]
    for exponent in range(7, maximum + 1):
        lower = 1 << (exponent - 1)
        widths.extend(
            lower + lower * subdivision // subdivisions
            for subdivision in range(1, subdivisions + 1)
        )
    return widths


@triton.jit
def _summarize_buckets(
    counts, budgets, bucket_ids, ranks, partial,
    rows: tl.constexpr, chunks: tl.constexpr, buckets: tl.constexpr,
    subdivisions: tl.constexpr, tile: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = chunk * tile + tl.arange(0, tile)
    valid = row < rows
    count = tl.load(counts + row, valid, other=0).to(tl.int32)
    budget = tl.load(budgets + row, valid, other=0).to(tl.int32)
    exponent = (
        tl.maximum(count - 1, 1).to(tl.float32).to(tl.int32, bitcast=True) >> 23
    ) - 127
    bucket = tl.where(count > 1, exponent + 1, 0)
    if subdivisions > 1:
        lower = 1 << tl.maximum(bucket - 1, 0)
        sub = tl.maximum(count - lower - 1, 0) // tl.maximum(lower // subdivisions, 1)
        bucket = tl.where(bucket >= 7, 7 + (bucket - 7) * subdivisions + sub, bucket)
    bucket = tl.where(count > 0, bucket, buckets - 1)
    tl.store(bucket_ids + row, bucket, valid)
    for bucket_index in range(buckets):
        selected = valid & (bucket == bucket_index)
        rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
        tl.store(ranks + row, rank, selected)
        address = bucket_index * chunks + chunk
        stride = buckets * chunks
        tl.store(partial + address, tl.sum(selected.to(tl.int64), axis=0))
        tl.store(partial + stride + address, tl.max(tl.where(selected, count, 0), axis=0))
        tl.store(partial + 2 * stride + address, tl.max(tl.where(selected, budget, 0), axis=0))
        tl.store(partial + 3 * stride + address, tl.sum(tl.where(selected, count, 0).to(tl.int64), axis=0))


@triton.jit
def _bucket_offsets(
    partial, offsets, metadata,
    buckets: tl.constexpr, chunks: tl.constexpr,
    bucket_tile: tl.constexpr, chunk_tile: tl.constexpr,
):
    bucket = tl.arange(0, bucket_tile)
    chunk = tl.arange(0, chunk_tile)
    address = bucket[:, None] * chunks + chunk[None, :]
    valid = (bucket[:, None] < buckets) & (chunk[None, :] < chunks)
    stride = buckets * chunks
    counts = tl.load(partial + address, valid, other=0)
    max_counts = tl.load(partial + stride + address, valid, other=0)
    max_budgets = tl.load(partial + 2 * stride + address, valid, other=0)
    blocks = tl.load(partial + 3 * stride + address, valid, other=0)
    totals = tl.sum(counts, axis=1)
    maximum_counts = tl.max(max_counts, axis=1)
    maximum_budgets = tl.max(max_budgets, axis=1)
    base = tl.cumsum(totals, axis=0) - totals
    starts = base[:, None] + tl.cumsum(counts, axis=1) - counts
    tl.store(offsets + address, starts, valid)
    tl.store(metadata, tl.max(maximum_counts, axis=0))
    tl.store(metadata + 1, tl.max(maximum_budgets, axis=0))
    tl.store(metadata + 2, tl.sum(tl.sum(blocks, axis=1), axis=0))
    tl.store(metadata + 3 + bucket, totals, bucket < buckets)
    tl.store(metadata + 3 + buckets + bucket, maximum_counts, bucket < buckets)
    tl.store(metadata + 3 + 2 * buckets + bucket, maximum_budgets, bucket < buckets)


@triton.jit
def _stable_order(
    bucket_ids, ranks, offsets, order,
    rows: tl.constexpr, chunks: tl.constexpr, tile: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = chunk * tile + tl.arange(0, tile)
    valid = row < rows
    bucket = tl.load(bucket_ids + row, valid, other=0)
    rank = tl.load(ranks + row, valid, other=0)
    start = tl.load(offsets + bucket * chunks + chunk, valid, other=0)
    tl.store(order + start + rank, row.to(tl.int64), valid)


def prepare_candidate_buckets(
    counts: torch.Tensor,
    budgets: torch.Tensor,
    *,
    num_blocks: int,
    subdivisions: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    widths = candidate_bucket_widths(num_blocks, subdivisions)
    if (
        not counts.is_cuda or counts.device != budgets.device
        or counts.shape != budgets.shape or not counts.is_contiguous()
        or not budgets.is_contiguous()
    ):
        raise ValueError("counts and budgets must be matching contiguous CUDA tensors")
    rows = counts.numel()
    buckets = len(widths) + 1
    order = torch.empty(rows, dtype=torch.int64, device=counts.device)
    metadata = torch.empty(3 + 3 * buckets, dtype=torch.int64, device=counts.device)
    if not rows:
        metadata.zero_()
        return order, metadata, widths
    tile = 256
    chunks = triton.cdiv(rows, tile)
    bucket_ids = torch.empty(rows, dtype=torch.int32, device=counts.device)
    ranks = torch.empty_like(bucket_ids)
    partial = torch.empty((4, buckets, chunks), dtype=torch.int64, device=counts.device)
    offsets = torch.empty((buckets, chunks), dtype=torch.int64, device=counts.device)
    _summarize_buckets[(chunks,)](
        counts, budgets, bucket_ids, ranks, partial, rows, chunks, buckets,
        subdivisions, tile, num_warps=4,
    )
    _bucket_offsets[(1,)](
        partial, offsets, metadata, buckets, chunks,
        triton.next_power_of_2(buckets), triton.next_power_of_2(chunks),
        num_warps=4,
    )
    _stable_order[(chunks,)](
        bucket_ids, ranks, offsets, order, rows, chunks, tile, num_warps=4
    )
    return order, metadata, widths
