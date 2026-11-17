import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Triton 3.x requires tl.dot tile sizes with M/N/K >= 16.
        triton.Config({"BLOCK_QB": 16, "BLOCK_KB": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QB": 16, "BLOCK_KB": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_QB": 32, "BLOCK_KB": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QB": 32, "BLOCK_KB": 64}, num_warps=8, num_stages=2),
    ],
    key=["K_BLOCKS", "Q_BLOCKS", "HIGH_DIM", "LOW_DIM"],
)
@triton.jit
def _dual_band_softmax_kernel(
    QH_ptr, QL_ptr, KH_ptr, KL_ptr, PROBS_ptr,
    stride_qh_b, stride_qh_h, stride_qh_qb, stride_qh_d,
    stride_ql_b, stride_ql_h, stride_ql_qb, stride_ql_d,
    stride_kh_b, stride_kh_h, stride_kh_kb, stride_kh_d,
    stride_kl_b, stride_kl_h, stride_kl_kb, stride_kl_d,
    stride_p_b, stride_p_h, stride_p_qb, stride_p_kb,
    B, H,
    scale_h_ptr, scale_l_ptr, stride_scale_b, stride_scale_h,
    scale_h, scale_l,
    USE_SCALE_TENSOR: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    BLOCK_QB: tl.constexpr,
    BLOCK_KB: tl.constexpr,
    HIGH_DIM: tl.constexpr,
    LOW_DIM: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    BLOCK_DL: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H
    if b >= B:
        return
    kv_h = h // NUM_KV_GROUPS

    qb_start = pid_qb * BLOCK_QB
    q_idx = qb_start + tl.arange(0, BLOCK_QB)
    mask_q = q_idx < Q_BLOCKS

    if USE_SCALE_TENSOR:
        scale_h_val = tl.load(scale_h_ptr + b * stride_scale_b + h * stride_scale_h).to(tl.float32)
        scale_l_val = tl.load(scale_l_ptr + b * stride_scale_b + h * stride_scale_h).to(tl.float32)
    else:
        scale_h_val = tl.full((), scale_h, tl.float32)
        scale_l_val = tl.full((), scale_l, tl.float32)

    d_h = tl.arange(0, BLOCK_DH)
    d_l = tl.arange(0, BLOCK_DL)

    qh_ptrs = (
        QH_ptr
        + b * stride_qh_b
        + h * stride_qh_h
        + q_idx[:, None] * stride_qh_qb
        + d_h[None, :] * stride_qh_d
    )
    ql_ptrs = (
        QL_ptr
        + b * stride_ql_b
        + h * stride_ql_h
        + q_idx[:, None] * stride_ql_qb
        + d_l[None, :] * stride_ql_d
    )
    qh = tl.load(qh_ptrs, mask=mask_q[:, None] & (d_h[None, :] < HIGH_DIM), other=0.0)
    ql = tl.load(ql_ptrs, mask=mask_q[:, None] & (d_l[None, :] < LOW_DIM), other=0.0)

    offs_k = tl.arange(0, BLOCK_KB)
    shift = K_BLOCKS - Q_BLOCKS
    max_kv = q_idx + shift
    mask_value = tl.full((), -1.0e9, tl.float32)

    m_h = tl.full((BLOCK_QB,), mask_value, tl.float32)
    m_l = tl.full((BLOCK_QB,), mask_value, tl.float32)
    l_h = tl.zeros((BLOCK_QB,), tl.float32)
    l_l = tl.zeros((BLOCK_QB,), tl.float32)

    for k_start in range(0, K_BLOCKS, BLOCK_KB):
        k_idx = k_start + offs_k
        mask_k = k_idx < K_BLOCKS
        mask_kv = mask_q[:, None] & mask_k[None, :] & (k_idx[None, :] <= max_kv[:, None])

        kh_ptrs = (
            KH_ptr
            + b * stride_kh_b
            + kv_h * stride_kh_h
            + k_idx[None, :] * stride_kh_kb
            + d_h[:, None] * stride_kh_d
        )
        kl_ptrs = (
            KL_ptr
            + b * stride_kl_b
            + kv_h * stride_kl_h
            + k_idx[None, :] * stride_kl_kb
            + d_l[:, None] * stride_kl_d
        )
        kh = tl.load(kh_ptrs, mask=(d_h[:, None] < HIGH_DIM) & mask_k[None, :], other=0.0)
        kl = tl.load(kl_ptrs, mask=(d_l[:, None] < LOW_DIM) & mask_k[None, :], other=0.0)

        logits_h = tl.dot(qh, kh).to(tl.float32) / scale_h_val
        logits_l = tl.dot(ql, kl).to(tl.float32) / scale_l_val
        logits_h = tl.where(mask_kv, logits_h, mask_value)
        logits_l = tl.where(mask_kv, logits_l, mask_value)

        max_h = tl.maximum(m_h, tl.max(logits_h, axis=1))
        max_l = tl.maximum(m_l, tl.max(logits_l, axis=1))

        exp_h = tl.exp(logits_h - max_h[:, None])
        exp_l = tl.exp(logits_l - max_l[:, None])
        exp_h = tl.where(mask_kv, exp_h, 0.0)
        exp_l = tl.where(mask_kv, exp_l, 0.0)

        l_h = l_h * tl.exp(m_h - max_h) + tl.sum(exp_h, axis=1)
        l_l = l_l * tl.exp(m_l - max_l) + tl.sum(exp_l, axis=1)
        m_h = max_h
        m_l = max_l

    l_h = tl.maximum(l_h, 1.0e-9)
    l_l = tl.maximum(l_l, 1.0e-9)

    for k_start in range(0, K_BLOCKS, BLOCK_KB):
        k_idx = k_start + offs_k
        mask_k = k_idx < K_BLOCKS
        mask_kv = mask_q[:, None] & mask_k[None, :] & (k_idx[None, :] <= max_kv[:, None])

        kh_ptrs = (
            KH_ptr
            + b * stride_kh_b
            + kv_h * stride_kh_h
            + k_idx[None, :] * stride_kh_kb
            + d_h[:, None] * stride_kh_d
        )
        kl_ptrs = (
            KL_ptr
            + b * stride_kl_b
            + kv_h * stride_kl_h
            + k_idx[None, :] * stride_kl_kb
            + d_l[:, None] * stride_kl_d
        )
        kh = tl.load(kh_ptrs, mask=(d_h[:, None] < HIGH_DIM) & mask_k[None, :], other=0.0)
        kl = tl.load(kl_ptrs, mask=(d_l[:, None] < LOW_DIM) & mask_k[None, :], other=0.0)

        logits_h = tl.dot(qh, kh).to(tl.float32) / scale_h_val
        logits_l = tl.dot(ql, kl).to(tl.float32) / scale_l_val
        logits_h = tl.where(mask_kv, logits_h, mask_value)
        logits_l = tl.where(mask_kv, logits_l, mask_value)

        p_h = tl.exp(logits_h - m_h[:, None]) / l_h[:, None]
        p_l = tl.exp(logits_l - m_l[:, None]) / l_l[:, None]
        p_h = tl.where(mask_kv, p_h, 0.0)
        p_l = tl.where(mask_kv, p_l, 0.0)

        p_h_ptrs = (
            PROBS_ptr
            + b * stride_p_b
            + h * stride_p_h
            + q_idx[:, None] * stride_p_qb
            + k_idx[None, :] * stride_p_kb
        )
        p_l_ptrs = (
            PROBS_ptr
            + b * stride_p_b
            + h * stride_p_h
            + (q_idx + Q_BLOCKS)[:, None] * stride_p_qb
            + k_idx[None, :] * stride_p_kb
        )
        store_mask = mask_q[:, None] & mask_k[None, :]
        tl.store(p_h_ptrs, p_h, mask=store_mask)
        tl.store(p_l_ptrs, p_l, mask=store_mask)


def dual_band_softmax_triton(
    q_high: torch.Tensor,
    q_low: torch.Tensor,
    k_high: torch.Tensor,
    k_low: torch.Tensor,
    scale_h: torch.Tensor | float,
    scale_l: torch.Tensor | float,
) -> torch.Tensor:
    if not q_high.is_cuda:
        raise RuntimeError("dual_band_softmax_triton expects CUDA tensors.")
    q_high = q_high.contiguous()
    q_low = q_low.contiguous()
    k_high = k_high.contiguous()
    k_low = k_low.contiguous()

    bsz, heads, q_blocks, high_dim = q_high.shape
    k_bsz, kv_heads, k_blocks, high_dim_k = k_high.shape
    k_low_bsz, k_low_heads, k_low_blocks, low_dim = k_low.shape
    if k_bsz != bsz or k_low_bsz != bsz:
        raise ValueError("Q and K tensors must have the same batch size.")
    if k_low_heads != kv_heads or k_low_blocks != k_blocks:
        raise ValueError("K high/low tensors must have matching head and block dimensions.")
    if high_dim_k != high_dim:
        raise ValueError("Q/K high-frequency dimensions must match.")
    if heads % kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads for GQA.")
    num_kv_groups = heads // kv_heads
    block_dh = triton.next_power_of_2(high_dim)
    block_dl = triton.next_power_of_2(low_dim)

    probs = torch.empty(
        (bsz, heads, 2 * q_blocks, k_blocks),
        device=q_high.device,
        dtype=torch.float32,
    )

    use_scale_tensor = isinstance(scale_h, torch.Tensor)
    if use_scale_tensor != isinstance(scale_l, torch.Tensor):
        raise RuntimeError("scale_h and scale_l must both be tensors or floats.")
    if use_scale_tensor:
        scale_h_t = scale_h.to(device=q_high.device, dtype=torch.float32, non_blocking=True)
        scale_l_t = scale_l.to(device=q_high.device, dtype=torch.float32, non_blocking=True)
        stride_scale_b, stride_scale_h = scale_h_t.stride()[:2]
        scale_h_ptr = scale_h_t
        scale_l_ptr = scale_l_t
        scale_h_val = 1.0
        scale_l_val = 1.0
    else:
        scale_h_ptr = q_high
        scale_l_ptr = q_low
        stride_scale_b = 0
        stride_scale_h = 0
        scale_h_val = float(scale_h)
        scale_l_val = float(scale_l)

    grid = lambda META: (triton.cdiv(q_blocks, META["BLOCK_QB"]), bsz * heads)
    _dual_band_softmax_kernel[grid](
        q_high,
        q_low,
        k_high,
        k_low,
        probs,
        q_high.stride(0),
        q_high.stride(1),
        q_high.stride(2),
        q_high.stride(3),
        q_low.stride(0),
        q_low.stride(1),
        q_low.stride(2),
        q_low.stride(3),
        k_high.stride(0),
        k_high.stride(1),
        k_high.stride(2),
        k_high.stride(3),
        k_low.stride(0),
        k_low.stride(1),
        k_low.stride(2),
        k_low.stride(3),
        probs.stride(0),
        probs.stride(1),
        probs.stride(2),
        probs.stride(3),
        bsz,
        heads,
        scale_h_ptr,
        scale_l_ptr,
        stride_scale_b,
        stride_scale_h,
        scale_h_val,
        scale_l_val,
        USE_SCALE_TENSOR=use_scale_tensor,
        NUM_KV_GROUPS=num_kv_groups,
        Q_BLOCKS=q_blocks,
        K_BLOCKS=k_blocks,
        HIGH_DIM=high_dim,
        LOW_DIM=low_dim,
        BLOCK_DH=block_dh,
        BLOCK_DL=block_dl,
    )
    return probs


@triton.jit
def _block_mass_selection_kernel(
    probabilities,
    output,
    stride_pb,
    stride_ph,
    stride_pq,
    stride_pk,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_ok,
    batch,
    heads,
    query_blocks,
    key_blocks,
    high_threshold,
    low_threshold,
    BLOCK_K: tl.constexpr,
    FORCE_SINK: tl.constexpr,
    FORCE_RECENT: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    if query_block >= query_blocks:
        return
    batch_index = batch_head // heads
    head = batch_head - batch_index * heads
    keys = tl.arange(0, BLOCK_K)
    valid = keys < key_blocks

    high = tl.load(
        probabilities
        + batch_index * stride_pb
        + head * stride_ph
        + query_block * stride_pq
        + keys * stride_pk,
        mask=valid,
        other=-1.0,
    )
    high_with_tie_break = high - keys.to(tl.float32) * 1e-9
    sorted_high = tl.sort(high_with_tie_break, descending=True)
    cumulative_high = tl.cumsum(sorted_high)
    keep_high = cumulative_high - sorted_high < high_threshold
    high_cutoff = tl.min(
        tl.where(keep_high & (sorted_high >= -0.5), sorted_high, 2.0)
    )

    low = tl.load(
        probabilities
        + batch_index * stride_pb
        + head * stride_ph
        + (query_block + query_blocks) * stride_pq
        + keys * stride_pk,
        mask=valid,
        other=-1.0,
    )
    low_with_tie_break = low - keys.to(tl.float32) * 1e-9
    sorted_low = tl.sort(low_with_tie_break, descending=True)
    cumulative_low = tl.cumsum(sorted_low)
    keep_low = cumulative_low - sorted_low < low_threshold
    low_cutoff = tl.min(
        tl.where(keep_low & (sorted_low >= -0.5), sorted_low, 2.0)
    )

    selected = (high_with_tie_break >= high_cutoff) | (
        low_with_tie_break >= low_cutoff
    )
    if FORCE_SINK:
        selected = selected | (keys == 0)
    if FORCE_RECENT:
        selected = selected | (
            keys == query_block + key_blocks - query_blocks
        )
    tl.store(
        output
        + batch_index * stride_ob
        + head * stride_oh
        + query_block * stride_oq
        + keys * stride_ok,
        selected,
        mask=valid,
    )


def select_blocks_by_mass_triton(
    probabilities: torch.Tensor,
    high_threshold: float,
    low_threshold: float,
    *,
    force_sink: bool = True,
    force_recent: bool = True,
) -> torch.Tensor:
    """CUDA block-level probability-mass selection for both frequency bands."""
    if not probabilities.is_cuda or probabilities.ndim != 4:
        raise ValueError("probabilities must be a four-dimensional CUDA tensor")
    batch, heads, combined_query_blocks, key_blocks = probabilities.shape
    if combined_query_blocks % 2:
        raise ValueError("probabilities must contain equal high/low query bands")
    query_blocks = combined_query_blocks // 2
    if query_blocks == 0 or key_blocks == 0:
        return torch.zeros(
            (batch, heads, query_blocks, key_blocks),
            device=probabilities.device,
            dtype=torch.bool,
        )
    output = torch.empty(
        (batch, heads, query_blocks, key_blocks),
        device=probabilities.device,
        dtype=torch.bool,
    )
    _block_mass_selection_kernel[(query_blocks, batch * heads)](
        probabilities,
        output,
        *probabilities.stride(),
        *output.stride(),
        batch,
        heads,
        query_blocks,
        key_blocks,
        high_threshold,
        low_threshold,
        BLOCK_K=max(2, triton.next_power_of_2(key_blocks)),
        FORCE_SINK=force_sink,
        FORCE_RECENT=force_recent,
    )
    return output

