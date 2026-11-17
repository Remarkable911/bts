"""Shared tensor helpers for BTS attention stages.

The coarse block selector, fine token selector, and model forward path all use
these helpers. Keeping them here prevents either selection stage from depending
on the implementation details of the other stage.
"""

import math

import torch


def mean_pool(states: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool a sequence into fixed-size blocks."""
    batch_size, num_heads, length, head_dim = states.shape
    if length % block_size != 0:
        raise ValueError(
            f"mean_pool expects length divisible by block_size, got {length} and {block_size}"
        )
    num_blocks = length // block_size
    return states.view(
        batch_size,
        num_heads,
        num_blocks,
        block_size,
        head_dim,
    ).mean(dim=-2)


def num_kv_groups(num_q_heads: int, num_kv_heads: int) -> int:
    """Return the number of query heads sharing each KV head."""
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads for GQA.")
    return num_q_heads // num_kv_heads


def expand_kv_head_tensor_to_q_heads(
    tensor: torch.Tensor,
    num_q_heads: int,
) -> torch.Tensor:
    """Expand small per-KV-head metadata to query-head layout."""
    num_kv_heads = tensor.shape[1]
    groups = num_kv_groups(num_q_heads, num_kv_heads)
    if groups == 1:
        return tensor
    expanded = tensor.unsqueeze(2).expand(
        tensor.shape[0],
        num_kv_heads,
        groups,
        *tensor.shape[2:],
    )
    return expanded.reshape(tensor.shape[0], num_q_heads, *tensor.shape[2:])


def gqa_query_key_scores(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    """Compute query-key scores without materializing repeated KV heads."""
    batch_size, num_q_heads, q_len, head_dim = query.shape
    key_batch_size, num_kv_heads, kv_len, key_dim = key.shape
    if key_batch_size != batch_size or key_dim != head_dim:
        raise ValueError("Q/K batch size and head dimension must match.")

    groups = num_kv_groups(num_q_heads, num_kv_heads)
    query_fp32 = query.to(torch.float32)
    key_fp32 = key.to(torch.float32)
    if groups == 1:
        return torch.einsum("bhqd,bhkd->bhqk", query_fp32, key_fp32)

    grouped_query = query_fp32.reshape(
        batch_size,
        num_kv_heads,
        groups,
        q_len,
        head_dim,
    )
    scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, key_fp32)
    return scores.reshape(batch_size, num_q_heads, q_len, kv_len)


def gqa_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Run exact decode attention without materializing repeated K/V heads."""
    batch_size, num_q_heads, q_len, head_dim = query_states.shape
    value_batch_size, num_kv_heads, kv_len, value_dim = value_states.shape
    if key_states.shape != (value_batch_size, num_kv_heads, kv_len, value_dim):
        raise ValueError("K and V must have the same shape.")
    if value_batch_size != batch_size or value_dim != head_dim:
        raise ValueError("Q and K/V batch size and head dimension must match.")

    groups = num_kv_groups(num_q_heads, num_kv_heads)
    scores = gqa_query_key_scores(query_states, key_states) / math.sqrt(head_dim)
    attention = torch.softmax(scores, dim=-1, dtype=torch.float32)
    if groups == 1:
        return torch.matmul(attention.to(value_states.dtype), value_states)

    grouped_attention = attention.reshape(
        batch_size,
        num_kv_heads,
        groups,
        q_len,
        kv_len,
    )
    grouped_output = torch.einsum(
        "bhgqk,bhkd->bhgqd",
        grouped_attention.to(value_states.dtype),
        value_states,
    )
    return grouped_output.reshape(batch_size, num_q_heads, q_len, head_dim)

