"""Direct causal attention over fixed-width packed token rows."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _direct_packed_attention(
    query,
    key,
    value,
    output,
    packed_indices,
    packed_counts,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    qs3: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    ks3: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    vs3: tl.constexpr,
    os0: tl.constexpr,
    os1: tl.constexpr,
    os2: tl.constexpr,
    os3: tl.constexpr,
    ps0: tl.constexpr,
    ps1: tl.constexpr,
    ps2: tl.constexpr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    q_len,
    source_kv_len,
    packed_kv_len,
    tokens_per_row,
    q_offset,
    softmax_scale,
    HEADS: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(128 % BLOCK_M == 0)
    query_tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)

    queries = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    keys_in_tile = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    query_valid = queries < q_len
    q = tl.load(
        query
        + batch * qs0
        + head * qs1
        + queries[:, None] * qs2
        + dims[None, :] * qs3,
        mask=query_valid[:, None],
        other=0.0,
    )
    query_positions = q_offset + queries
    query_block = (query_tile * BLOCK_M) // 128
    packed_start = query_block * tokens_per_row
    row_tokens = tl.load(
        packed_counts
        + batch * cs0
        + head * cs1
        + query_block * cs2
    ).to(tl.int32)
    row_tokens = tl.maximum(0, tl.minimum(row_tokens, tokens_per_row))

    kv_head = head // GROUPS
    key_base = key + batch * ks0 + kv_head * ks1
    value_base = value + batch * vs0 + kv_head * vs1
    packed_base = packed_indices + batch * ps0 + head * ps1
    dtype = query.type.element_ty
    qk_scale = softmax_scale * 1.4426950408889634

    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for local_start in tl.range(0, row_tokens, BLOCK_N):
        local = local_start + keys_in_tile
        packed_offset = packed_start + local
        packed_valid = (local < row_tokens) & (packed_offset < packed_kv_len)
        key_positions = tl.load(
            packed_base + packed_offset * ps2,
            mask=packed_valid,
            other=source_kv_len,
        )
        key_valid = packed_valid & (key_positions >= 0) & (
            key_positions < source_kv_len
        )
        k = tl.load(
            key_base
            + key_positions[:, None] * ks2
            + dims[None, :] * ks3,
            mask=key_valid[:, None],
            other=0.0,
        )
        logits = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        visible = (
            query_valid[:, None]
            & key_valid[None, :]
            & (query_positions[:, None] >= key_positions[None, :])
        )
        logits = tl.where(visible, logits, -1.0e20)
        new_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        probabilities = tl.math.exp2(logits - new_maximum[:, None])
        probabilities = tl.where(visible, probabilities, 0.0)
        local_denominator = tl.sum(probabilities, axis=1)
        correction = tl.math.exp2(maximum - new_maximum)
        accumulator *= correction[:, None]
        v = tl.load(
            value_base
            + key_positions[:, None] * vs2
            + dims[None, :] * vs3,
            mask=key_valid[:, None],
            other=0.0,
        )
        accumulator += tl.dot(probabilities.to(dtype), v)
        denominator = denominator * correction + local_denominator
        maximum = new_maximum

    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        output
        + batch * os0
        + head * os1
        + queries[:, None] * os2
        + dims[None, :] * os3,
        result.to(dtype),
        mask=query_valid[:, None],
    )


@triton.jit
def _fixed_prefix_attn_fwd(
    Q,
    K,
    V,
    O,
    PACKED_K_INDICES,
    PACKED_TOKEN_COUNTS,
    stride_bz_q,
    stride_h_q,
    stride_seq_q,
    stride_d_q,
    stride_bz_k,
    stride_h_k,
    stride_seq_k,
    stride_d_k,
    stride_bz_v,
    stride_h_v,
    stride_seq_v,
    stride_d_v,
    stride_bz_o,
    stride_h_o,
    stride_seq_o,
    stride_d_o,
    stride_bz_packed_k_indices,
    stride_h_packed_k_indices,
    stride_seq_packed_k_indices,
    stride_bz_packed_token_counts,
    stride_h_packed_token_counts,
    stride_q_block_packed_token_counts,
    source_kv_len,
    packed_tokens_per_q,
    softmax_scale,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Exact specialization over fixed prefixes produced by the internal packer."""
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_bz = tl.program_id(2).to(tl.int64)

    offs_m = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        Q
        + pid_bz * stride_bz_q
        + pid_h * stride_h_q
        + offs_m[:, None] * stride_seq_q
        + offs_d[None, :] * stride_d_q
    )
    q = tl.load(q_ptrs)
    q_indices = offs_m

    packed_q_start = pid_seq * packed_tokens_per_q
    packed_q_start = tl.multiple_of(packed_q_start, BLOCK_M)
    row_packed_tokens = tl.load(
        PACKED_TOKEN_COUNTS
        + pid_bz * stride_bz_packed_token_counts
        + pid_h * stride_h_packed_token_counts
        + pid_seq * stride_q_block_packed_token_counts
    ).to(tl.int32)
    row_packed_tokens = tl.maximum(
        0,
        tl.minimum(row_packed_tokens, packed_tokens_per_q),
    )
    packed_k_indices_base = (
        PACKED_K_INDICES
        + pid_bz * stride_bz_packed_k_indices
        + pid_h * stride_h_packed_k_indices
    )

    kv_h = pid_h // num_kv_groups
    k_base = K + pid_bz * stride_bz_k + kv_h * stride_h_k
    v_base = V + pid_bz * stride_bz_v + kv_h * stride_h_v
    dtype = Q.type.element_ty
    qk_scale = softmax_scale * 1.4426950408889634

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    full_end = (row_packed_tokens // BLOCK_N) * BLOCK_N
    for local_start in tl.range(0, full_end, BLOCK_N):
        local_start = tl.multiple_of(local_start, BLOCK_N)
        packed_offsets = tl.max_contiguous(
            packed_q_start + local_start + offs_n,
            BLOCK_N,
        )
        k_indices = tl.load(
            packed_k_indices_base
            + packed_offsets * stride_seq_packed_k_indices
        )
        key_valid = (k_indices >= 0) & (k_indices < source_kv_len)
        safe_k_indices = tl.where(key_valid, k_indices, 0)

        k_ptrs = (
            k_base
            + safe_k_indices[:, None] * stride_seq_k
            + offs_d[None, :] * stride_d_k
        )
        k = tl.load(k_ptrs, mask=key_valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        valid = key_valid[None, :] & (
            q_indices[:, None] >= safe_k_indices[None, :]
        )

        qk = tl.where(valid, qk, -1.0e20)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        p = tl.where(valid, p, 0.0)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]
        v_ptrs = (
            v_base
            + safe_k_indices[:, None] * stride_seq_v
            + offs_d[None, :] * stride_d_v
        )
        v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    if full_end < row_packed_tokens:
        local_start = tl.multiple_of(full_end, BLOCK_N)
        local_offsets = local_start + offs_n
        packed_offsets = tl.max_contiguous(
            packed_q_start + local_offsets,
            BLOCK_N,
        )
        packed_mask = local_offsets < row_packed_tokens
        k_indices = tl.load(
            packed_k_indices_base
            + packed_offsets * stride_seq_packed_k_indices,
            mask=packed_mask,
            other=source_kv_len,
        )
        key_valid = packed_mask & (k_indices >= 0) & (
            k_indices < source_kv_len
        )
        safe_k_indices = tl.where(key_valid, k_indices, 0)

        k_ptrs = (
            k_base
            + safe_k_indices[:, None] * stride_seq_k
            + offs_d[None, :] * stride_d_k
        )
        k = tl.load(k_ptrs, mask=key_valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        valid = key_valid[None, :] & (
            q_indices[:, None] >= safe_k_indices[None, :]
        )

        qk = tl.where(valid, qk, -1.0e20)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        p = tl.where(valid, p, 0.0)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]
        v_ptrs = (
            v_base
            + safe_k_indices[:, None] * stride_seq_v
            + offs_d[None, :] * stride_d_v
        )
        v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    denom = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / denom[:, None]
    o_ptrs = (
        O
        + pid_bz * stride_bz_o
        + pid_h * stride_h_o
        + offs_m[:, None] * stride_seq_o
        + offs_d[None, :] * stride_d_o
    )
    tl.store(o_ptrs, out.to(dtype))

def direct_packed_token_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed_indices: torch.Tensor,
    packed_counts: torch.Tensor,
    *,
    source_kv_len: int,
    q_offset: int = 0,
) -> torch.Tensor:
    """Run the submitted fixed-row packed attention kernel."""
    tensors = (query, key, value, packed_indices, packed_counts)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("direct packed attention expects CUDA tensors")
    if not all(tensor.device == query.device for tensor in tensors[1:]):
        raise ValueError("all direct-attention tensors must share one CUDA device")
    if key.shape != value.shape or query.shape[0] != key.shape[0]:
        raise ValueError("query/key/value shapes are incompatible")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise TypeError("query, key, and value must have matching dtypes")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("direct packed attention requires a floating Q/K/V dtype")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if query.shape[-1] != 128:
        raise ValueError("the frozen direct-attention kernel supports head_dim=128 only")
    if query.shape[2] <= 0 or query.shape[2] % 128:
        raise ValueError("query length must be a positive multiple of 128")
    query_blocks = query.shape[2] // 128
    expected_counts = (query.shape[0], query.shape[1], query_blocks)
    if packed_counts.shape != expected_counts:
        raise ValueError(f"packed_counts must have shape {expected_counts}")
    if packed_indices.ndim != 3 or packed_indices.shape[:2] != query.shape[:2]:
        raise ValueError("packed_indices must have shape [batch, query_heads, tokens]")
    if packed_indices.shape[-1] % max(query_blocks, 1):
        raise ValueError("packed storage must divide evenly across query blocks")
    if packed_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("packed indices must use int32 or int64")
    if packed_counts.dtype not in (torch.int32, torch.int64):
        raise TypeError("packed counts must use int32 or int64")
    if not 0 < source_kv_len <= key.shape[2]:
        raise ValueError("source_kv_len is outside the key/value sequence")
    if q_offset < 0:
        raise ValueError("q_offset must be nonnegative")

    packed_indices = packed_indices.contiguous()
    packed_counts = packed_counts.contiguous()
    tokens_per_row = packed_indices.shape[-1] // max(query_blocks, 1)
    if not tokens_per_row:
        return torch.zeros_like(query)
    output = torch.empty_like(query, memory_format=torch.preserve_format)
    block_m, block_n = 128, 32
    _direct_packed_attention[
        (triton.cdiv(query.shape[2], block_m), query.shape[1], query.shape[0])
    ](
        query,
        key,
        value,
        output,
        packed_indices,
        packed_counts,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *packed_indices.stride(),
        *packed_counts.stride(),
        query.shape[2],
        source_kv_len,
        packed_indices.shape[-1],
        tokens_per_row,
        q_offset,
        1.0 / math.sqrt(query.shape[-1]),
        HEADS=query.shape[1],
        QUERY_BLOCKS=query_blocks,
        GROUPS=query.shape[1] // key.shape[1],
        HEAD_DIM=query.shape[-1],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return output


def _validate_fixed_prefix_contract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed_indices: torch.Tensor,
    packed_counts: torch.Tensor,
    *,
    source_kv_len: int,
) -> int:
    """Validate the synchronization-free internal prefix contract.

    Count bounds and active token-ID bounds are guarantees of the internal
    packer. Reading them back here would synchronize the GPU on every call.
    """
    tensors = (query, key, value, packed_indices, packed_counts)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fixed-prefix attention requires CUDA tensors")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("fixed-prefix attention tensors must share one CUDA device")
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        raise ValueError("fixed-prefix attention requires an SM90 CUDA device")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("fixed-prefix attention requires rank-four Q/K/V")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("fixed-prefix attention requires BF16 Q/K/V")
    if (
        query.shape[0] != 1
        or query.shape[1] != 32
        or key.shape[0] != 1
        or key.shape[1] != 8
        or value.shape[:2] != key.shape[:2]
        or query.shape[-1] != 128
        or key.shape[-1] != 128
        or value.shape[-1] != 128
    ):
        raise ValueError(
            "fixed-prefix attention requires the 1x32Q/8KV, D128 GQA4 layout"
        )
    q_len = query.shape[2]
    if q_len <= 0 or q_len % 128:
        raise ValueError("fixed-prefix attention requires a positive 128-aligned length")
    if source_kv_len != q_len or key.shape[2] != q_len or value.shape[2] != q_len:
        raise ValueError("fixed-prefix attention requires equal Q/K/V and source lengths")
    if packed_indices.dtype != torch.int32 or packed_counts.dtype != torch.int32:
        raise ValueError("fixed-prefix attention requires int32 packed IDs and counts")
    if not packed_indices.is_contiguous() or not packed_counts.is_contiguous():
        raise ValueError("fixed-prefix attention metadata must be contiguous")
    query_blocks = q_len // 128
    if packed_indices.ndim != 3 or packed_indices.shape[:2] != query.shape[:2]:
        raise ValueError("fixed-prefix packed IDs must match Q batch and heads")
    if packed_indices.shape[-1] % query_blocks:
        raise ValueError("fixed-prefix packed storage must divide evenly across rows")
    tokens_per_row = packed_indices.shape[-1] // query_blocks
    if tokens_per_row <= 0 or tokens_per_row % 128:
        raise ValueError("fixed-prefix row capacity must be positive and 128-aligned")
    if packed_counts.shape != (*query.shape[:2], query_blocks):
        raise ValueError("fixed-prefix count shape must match Q batch, heads, and rows")
    return tokens_per_row


def _fixed_prefix_packed_token_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed_indices: torch.Tensor,
    packed_counts: torch.Tensor,
    *,
    source_kv_len: int,
) -> torch.Tensor:
    """Run the exact default kernel for a layout built by the prefix packer."""
    tokens_per_row = _validate_fixed_prefix_contract(
        query,
        key,
        value,
        packed_indices,
        packed_counts,
        source_kv_len=source_kv_len,
    )
    output = torch.empty_like(query, memory_format=torch.preserve_format)
    grid = (query.shape[2] // 128, query.shape[1], query.shape[0])
    _fixed_prefix_attn_fwd[grid](
        query,
        key,
        value,
        output,
        packed_indices,
        packed_counts,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *packed_indices.stride(),
        *packed_counts.stride(),
        int(source_kv_len),
        int(tokens_per_row),
        1.0 / math.sqrt(query.shape[-1]),
        num_kv_groups=4,
        HEAD_DIM=128,
        BLOCK_M=128,
        BLOCK_N=32,
        num_warps=4,
        num_stages=2,
    )
    return output

