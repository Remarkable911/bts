"""Mean-128 query representation used by token selection."""

from dataclasses import dataclass

import torch

from .config import BLOCK_SIZE


@dataclass(frozen=True)
class QueryBlocks:
    means: torch.Tensor
    last_positions: torch.Tensor


def _query_block_means(
    query: torch.Tensor,
    cached_means: torch.Tensor,
) -> QueryBlocks:
    """Validate the unique native-dtype mean128 cache and prepare FP32 scores."""
    batch, heads, q_len, dim = query.shape
    if q_len % BLOCK_SIZE:
        raise ValueError("mean128 selection requires a query length divisible by 128")
    rows = q_len // BLOCK_SIZE
    expected = (batch, heads, rows, dim)
    if tuple(cached_means.shape) != expected:
        raise ValueError(
            f"cached query means must have shape {expected}, "
            f"got {tuple(cached_means.shape)}"
        )
    if cached_means.device != query.device:
        raise ValueError("cached query means must be on the query device")
    if cached_means.dtype != query.dtype:
        raise TypeError("cached query means must use the query dtype")
    means = cached_means.to(torch.float32)
    last = (
        (torch.arange(rows, device=query.device, dtype=torch.long) + 1)
        * BLOCK_SIZE
        - 1
    )
    return QueryBlocks(means, last)
