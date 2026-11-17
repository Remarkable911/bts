"""Budget arithmetic for the frozen block-count 0.4 selector."""

import math

import torch

from .config import BLOCK_SIZE, TOKEN_RATIO


def block_count_budget(
    candidate_block_counts: torch.Tensor,
    candidate_token_counts: torch.Tensor,
) -> torch.Tensor:
    """Select ceil(block_count * 128 * 0.4), clipped to valid candidates."""
    budgets = torch.ceil(
        candidate_block_counts.to(torch.float32) * BLOCK_SIZE * TOKEN_RATIO
    ).to(torch.long)
    budgets = torch.minimum(budgets, candidate_token_counts)
    return torch.where(
        candidate_token_counts > 0,
        budgets.clamp_min(1),
        torch.zeros_like(budgets),
    ).to(torch.int32)


def bucket_budget_upper_bound(bucket_blocks: int, row_maximum: int) -> int:
    """Host-known Top-K bound for one compact candidate bucket."""
    bound = math.ceil(bucket_blocks * BLOCK_SIZE * TOKEN_RATIO)
    return max(0, min(bound, row_maximum, bucket_blocks * BLOCK_SIZE))
