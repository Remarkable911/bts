"""Exact FP32 scoring of tokens inside compact candidate blocks."""

import torch
import triton
import triton.language as tl

_SUPPORTED_QK_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_SUPPORTED_SCORE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

@triton.jit
def _candidate_block_token_scores_exact_kernel(
    Q_REPR,
    KEY_STATES,
    ROW_INDICES,
    CANDIDATE_BLOCK_IDS,
    Q_LAST_POSITIONS,
    OUTPUT_SCORES,
    stride_b_q,
    stride_h_q,
    stride_qb_q,
    stride_d_q,
    stride_b_k,
    stride_h_k,
    stride_seq_k,
    stride_d_k,
    stride_row_block,
    stride_block_slot,
    stride_out_row,
    stride_out_token,
    kv_len,
    num_k_blocks,
    NUM_Q_HEADS: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Score one candidate KV block for one compacted query row."""
    block_slot = tl.program_id(0).to(tl.int64)
    compact_row = tl.program_id(1).to(tl.int64)
    row_idx = tl.load(ROW_INDICES + compact_row).to(tl.int64)

    q_block_idx = row_idx % NUM_Q_BLOCKS
    head_row = row_idx // NUM_Q_BLOCKS
    q_head_idx = head_row % NUM_Q_HEADS
    batch_idx = head_row // NUM_Q_HEADS
    kv_head_idx = q_head_idx // NUM_KV_GROUPS

    candidate_block = tl.load(
        CANDIDATE_BLOCK_IDS
        + compact_row * stride_row_block
        + block_slot * stride_block_slot
    ).to(tl.int64)
    token_offsets = tl.arange(0, TOKEN_BLOCK_SIZE)
    token_positions = candidate_block * TOKEN_BLOCK_SIZE + token_offsets
    candidate_is_valid = (
        (candidate_block >= 0)
        & (candidate_block < num_k_blocks)
        & (token_positions >= 0)
        & (token_positions < kv_len)
    )
    if CAUSAL:
        q_last_position = tl.load(Q_LAST_POSITIONS + q_block_idx).to(tl.int64)
        candidate_is_valid = candidate_is_valid & (token_positions <= q_last_position)
    safe_positions = tl.where(candidate_is_valid, token_positions, 0)

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    query_ptrs = (
        Q_REPR
        + batch_idx * stride_b_q
        + q_head_idx * stride_h_q
        + q_block_idx * stride_qb_q
        + dim_offsets * stride_d_q
    )
    query = tl.load(query_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
    key_ptrs = (
        KEY_STATES
        + batch_idx * stride_b_k
        + kv_head_idx * stride_h_k
        + safe_positions[:, None] * stride_seq_k
        + dim_offsets[None, :] * stride_d_k
    )
    keys = tl.load(
        key_ptrs,
        mask=candidate_is_valid[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(keys * query[None, :], axis=1)
    scores = tl.where(candidate_is_valid, scores, -float("inf"))

    output_offsets = block_slot * TOKEN_BLOCK_SIZE + token_offsets
    tl.store(
        OUTPUT_SCORES
        + compact_row * stride_out_row
        + output_offsets * stride_out_token,
        scores,
    )


@triton.jit
def _candidate_block_token_scores_pruned_kernel(
    Q_REPR,
    KEY_STATES,
    ROW_INDICES,
    CANDIDATE_BLOCK_IDS,
    Q_LAST_POSITIONS,
    OUTPUT_SCORES,
    stride_b_q,
    stride_h_q,
    stride_qb_q,
    stride_d_q,
    stride_b_k,
    stride_h_k,
    stride_seq_k,
    stride_d_k,
    stride_row_block,
    stride_block_slot,
    stride_out_row,
    stride_out_token,
    kv_len,
    num_k_blocks,
    NUM_Q_HEADS: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BW: tl.constexpr,
    LOOP: tl.constexpr,
):
    """Reuse Q and skip wholly masked blocks; retain the exact FP32 reduction."""
    first_slot = tl.program_id(0).to(tl.int64) * LOOP
    compact_row = tl.program_id(1).to(tl.int64)
    row_idx = tl.load(ROW_INDICES + compact_row).to(tl.int64)

    q_block_idx = row_idx % NUM_Q_BLOCKS
    head_row = row_idx // NUM_Q_BLOCKS
    q_head_idx = head_row % NUM_Q_HEADS
    batch_idx = head_row // NUM_Q_HEADS
    kv_head_idx = q_head_idx // NUM_KV_GROUPS

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    query_ptrs = (
        Q_REPR
        + batch_idx * stride_b_q
        + q_head_idx * stride_h_q
        + q_block_idx * stride_qb_q
        + dim_offsets * stride_d_q
    )
    query = tl.load(query_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
    if CAUSAL:
        q_last_position = tl.load(Q_LAST_POSITIONS + q_block_idx).to(tl.int64)

    for block_slot in range(first_slot, tl.minimum(first_slot + LOOP, BW)):
        candidate_block = tl.load(
            CANDIDATE_BLOCK_IDS
            + compact_row * stride_row_block
            + block_slot * stride_block_slot
        ).to(tl.int64)
        active_block = (candidate_block >= 0) & (candidate_block < num_k_blocks)
        if CAUSAL:
            active_block = active_block & (candidate_block * TOKEN_BLOCK_SIZE <= q_last_position)
        if active_block:
            token_offsets = tl.arange(0, TOKEN_BLOCK_SIZE)
            token_positions = candidate_block * TOKEN_BLOCK_SIZE + token_offsets
            candidate_is_valid = (
                (token_positions >= 0)
                & (token_positions < kv_len)
            )
            if CAUSAL:
                candidate_is_valid = candidate_is_valid & (token_positions <= q_last_position)
            safe_positions = tl.where(candidate_is_valid, token_positions, 0)

            key_ptrs = (
                KEY_STATES
                + batch_idx * stride_b_k
                + kv_head_idx * stride_h_k
                + safe_positions[:, None] * stride_seq_k
                + dim_offsets[None, :] * stride_d_k
            )
            keys = tl.load(
                key_ptrs,
                mask=candidate_is_valid[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1)
            scores = tl.where(candidate_is_valid, scores, -float("inf"))
        else:
            scores = tl.full((TOKEN_BLOCK_SIZE,), -float("inf"), tl.float32)
        token_offsets = tl.arange(0, TOKEN_BLOCK_SIZE)

        output_offsets = block_slot * TOKEN_BLOCK_SIZE + token_offsets
        tl.store(
            OUTPUT_SCORES
            + compact_row * stride_out_row
            + output_offsets * stride_out_token,
            scores,
        )


@triton.jit
def _candidate_block_token_scores_aligned_full_kernel(
    Q_REPR,
    KEY_STATES,
    ROW_INDICES,
    CANDIDATE_BLOCK_IDS,
    OUTPUT_SCORES,
    stride_b_q,
    stride_h_q,
    stride_qb_q,
    stride_d_q,
    stride_b_k,
    stride_h_k,
    stride_seq_k,
    stride_d_k,
    stride_row_block,
    stride_block_slot,
    stride_out_row,
    stride_out_token,
    num_k_blocks,
    NUM_Q_HEADS: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    BW: tl.constexpr,
    LOOP: tl.constexpr,
):
    """Score aligned causal blocks known to precede their query block.

    The ordinary Tail256/mean128 selector removes the diagonal recent block
    before scoring.  Every remaining valid candidate is therefore a complete
    128-token block, so the per-token causal/padding predicates and q-last load
    are redundant.  The FP32 multiply and reduction are unchanged.
    """
    first_slot = tl.program_id(0).to(tl.int64) * LOOP
    compact_row = tl.program_id(1).to(tl.int64)
    row_idx = tl.load(ROW_INDICES + compact_row).to(tl.int64)

    q_block_idx = row_idx % NUM_Q_BLOCKS
    head_row = row_idx // NUM_Q_BLOCKS
    q_head_idx = head_row % NUM_Q_HEADS
    batch_idx = head_row // NUM_Q_HEADS
    kv_head_idx = q_head_idx // NUM_KV_GROUPS

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    query = tl.load(
        Q_REPR
        + batch_idx * stride_b_q
        + q_head_idx * stride_h_q
        + q_block_idx * stride_qb_q
        + dim_offsets * stride_d_q,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    token_offsets = tl.arange(0, TOKEN_BLOCK_SIZE)

    for block_slot in range(first_slot, tl.minimum(first_slot + LOOP, BW)):
        candidate_block = tl.load(
            CANDIDATE_BLOCK_IDS
            + compact_row * stride_row_block
            + block_slot * stride_block_slot
        ).to(tl.int64)
        active_block = (
            (candidate_block >= 0)
            & (candidate_block < num_k_blocks)
            & (candidate_block < q_block_idx)
        )
        if active_block:
            token_positions = candidate_block * TOKEN_BLOCK_SIZE + token_offsets
            keys = tl.load(
                KEY_STATES
                + batch_idx * stride_b_k
                + kv_head_idx * stride_h_k
                + token_positions[:, None] * stride_seq_k
                + dim_offsets[None, :] * stride_d_k,
                mask=dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1)
        else:
            scores = tl.full((TOKEN_BLOCK_SIZE,), -float("inf"), tl.float32)
        tl.store(
            OUTPUT_SCORES
            + compact_row * stride_out_row
            + (block_slot * TOKEN_BLOCK_SIZE + token_offsets) * stride_out_token,
            scores,
        )


def candidate_block_token_scores_exact(
    q_repr: torch.Tensor,
    key_states: torch.Tensor,
    row_indices: torch.Tensor,
    candidate_block_ids: torch.Tensor,
    q_last_positions: torch.Tensor,
    *,
    block_size: int,
    num_key_value_groups: int,
    causal: bool,
    output_dtype: torch.dtype = torch.float32,
    num_warps: int | None = None,
    blocks_per_program: int = 0,
    aligned_causal_full_blocks: bool = False,
) -> torch.Tensor:
    """Score compact candidate blocks without materializing token positions.

    ``row_indices`` maps each compact row back to flattened ``[B, H, QB]``.
    Padding block ids must be negative or at least ``ceil(kv_len / block_size)``;
    their output scores are ``-inf``. ``blocks_per_program=0`` chooses serial
    tasks for large native-precision buckets; 1 retains one block per task.
    """
    tensors = (q_repr, key_states, row_indices, candidate_block_ids, q_last_positions)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("candidate_block_token_scores_exact expects CUDA tensors.")
    if not all(tensor.device == q_repr.device for tensor in tensors[1:]):
        raise ValueError("All candidate score tensors must be on the same CUDA device.")
    if q_repr.ndim != 4 or key_states.ndim != 4:
        raise ValueError("q_repr and key_states must be four-dimensional tensors.")
    if row_indices.ndim != 1 or candidate_block_ids.ndim != 2:
        raise ValueError("row_indices must be 1D and candidate_block_ids must be 2D.")
    if candidate_block_ids.shape[0] != row_indices.shape[0]:
        raise ValueError("candidate_block_ids rows must match row_indices.")
    if q_last_positions.ndim != 1 or q_last_positions.shape[0] != q_repr.shape[2]:
        raise ValueError("q_last_positions must contain one position per query block.")
    if q_repr.dtype not in _SUPPORTED_QK_DTYPES:
        raise TypeError(f"Unsupported q_repr dtype: {q_repr.dtype}.")
    if key_states.dtype not in _SUPPORTED_QK_DTYPES:
        raise TypeError(f"Unsupported key_states dtype: {key_states.dtype}.")
    if output_dtype not in _SUPPORTED_SCORE_DTYPES:
        raise TypeError(f"Unsupported score output dtype: {output_dtype}.")
    if row_indices.dtype != torch.long or candidate_block_ids.dtype != torch.long:
        raise TypeError("row_indices and candidate_block_ids must have dtype torch.int64.")
    if q_last_positions.dtype != torch.long:
        raise TypeError("q_last_positions must have dtype torch.int64.")
    if block_size not in (16, 32, 64, 128, 256):
        raise ValueError("block_size must be one of 16, 32, 64, 128, or 256.")
    if num_key_value_groups <= 0:
        raise ValueError("num_key_value_groups must be positive.")

    if blocks_per_program not in (0, 1, 4, 8, 16, 32):
        raise ValueError("blocks_per_program must be 0 (auto), 1, 4, 8, 16, or 32.")

    if num_warps is not None and num_warps not in (1, 2, 4, 8, 16):
        raise ValueError("num_warps must be a supported power of two.")

    batch_size, num_q_heads, num_q_blocks, head_dim = q_repr.shape
    key_batch_size, num_kv_heads, kv_len, key_head_dim = key_states.shape
    schedule_len = kv_len
    if key_batch_size != batch_size or key_head_dim != head_dim:
        raise ValueError("q_repr and key_states batch size and head dimension must match.")
    if num_q_heads != num_kv_heads * num_key_value_groups:
        raise ValueError("Q heads must equal KV heads * num_key_value_groups.")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in the range [1, 256].")

    num_rows, block_width = candidate_block_ids.shape
    output_scores = torch.empty(
        (num_rows, block_width * block_size),
        device=q_repr.device,
        dtype=output_dtype,
    )
    if num_rows == 0 or block_width == 0:
        return output_scores

    block_d = triton.next_power_of_2(head_dim)
    num_k_blocks = triton.cdiv(kv_len, block_size)
    # Four warps reduced score time on the measured H100 / D=128 buckets.
    # Keep the previous configuration for other tile shapes; callers can A/B
    # the scheduling choice without changing the FP32 reduction or precision.
    if num_warps is None:
        num_warps = 4 if head_dim == 128 and block_size == 128 else (8 if head_dim >= 128 else 4)
    if blocks_per_program == 0:
        # Measured on H100: serial blocks amortize task/Q-load overhead. Small
        # grids retain one block per task to keep sufficient parallelism.
        blocks_per_program = 16 if (
            head_dim == 128 and block_size == 128 and schedule_len >= 65536
            and block_width >= 64 and num_rows >= 128
            and q_repr.dtype == torch.float32 and key_states.dtype == torch.bfloat16
            and output_dtype == torch.float32
        ) else 1
    kernel = _candidate_block_token_scores_exact_kernel
    scheduling = {}
    use_aligned_full_blocks = (
        aligned_causal_full_blocks
        and causal
        and kv_len % block_size == 0
        and num_q_blocks == num_k_blocks
        and q_repr.dtype == torch.float32
        and key_states.dtype == torch.bfloat16
        and output_dtype == torch.float32
        and head_dim == 128
        and block_size == 128
        and torch.cuda.get_device_capability(q_repr.device) == (9, 0)
    )
    if use_aligned_full_blocks:
        kernel = _candidate_block_token_scores_aligned_full_kernel
        scheduling = dict(BW=block_width, LOOP=blocks_per_program)
    elif blocks_per_program > 1:
        kernel = _candidate_block_token_scores_pruned_kernel
        scheduling = dict(BW=block_width, LOOP=blocks_per_program)
    grid = (triton.cdiv(block_width, blocks_per_program), num_rows)
    common_args = (
        q_repr,
        key_states,
        row_indices,
        candidate_block_ids,
    )
    if use_aligned_full_blocks:
        kernel_args = (*common_args, output_scores)
    else:
        kernel_args = (*common_args, q_last_positions, output_scores)
    kernel[grid](
        *kernel_args,
        q_repr.stride(0),
        q_repr.stride(1),
        q_repr.stride(2),
        q_repr.stride(3),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        candidate_block_ids.stride(0),
        candidate_block_ids.stride(1),
        output_scores.stride(0),
        output_scores.stride(1),
        *(() if use_aligned_full_blocks else (int(kv_len),)),
        int(num_k_blocks),
        NUM_Q_HEADS=num_q_heads,
        NUM_Q_BLOCKS=num_q_blocks,
        NUM_KV_GROUPS=int(num_key_value_groups),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        TOKEN_BLOCK_SIZE=int(block_size),
        **({} if use_aligned_full_blocks else {"CAUSAL": bool(causal)}),
        num_warps=num_warps,
        **scheduling,
    )
    return output_scores

