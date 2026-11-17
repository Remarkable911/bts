"""Width-specialized exact Top-K implementation of block-count 0.4."""

import math

import torch

from .config import BLOCK_SIZE
from .token_budget import bucket_budget_upper_bound
from .kernels.candidate_token_score import candidate_block_token_scores_exact
from .kernels.compact_candidates import (
    compact_candidate_rows,
    fused_short_token_select,
    gather_candidate_bucket,
    prepare_candidate_buckets,
)
from .kernels.exact_topk import direct_topk_to_token_positions
from .kernels.token_topk_postprocess import (
    compact_sorted_topk_to_token_positions,
    radix_budget_to_token_positions,
)
from .kernels.variable_topk import try_variable_topk_to_token_positions
from .utils.stat_collector import StatCollector


def select_tokens(
    query_means: torch.Tensor,
    key_states: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_block_counts: torch.Tensor,
    budgets: torch.Tensor,
    q_last_positions: torch.Tensor,
    *,
    collector: StatCollector,
) -> torch.Tensor:
    """Select exact row-specific Top-K tokens from compact BTS blocks.

    The width-dependent kernels are implementation fallbacks for this one
    selection rule. They are not user-visible alternative algorithms.
    """
    batch, heads, query_blocks, key_blocks = candidate_mask.shape
    rows = batch * heads * query_blocks
    flat_counts = candidate_block_counts.reshape(rows).contiguous()
    flat_budgets = budgets.reshape(rows).contiguous()

    with collector.stage("token_sparse.layout.candidate_mask", query_means):
        candidate_ids, compact_counts = compact_candidate_rows(candidate_mask.contiguous())
        subdivisions = 4 if key_states.shape[2] >= 131072 else 1
        row_order, metadata_tensor, bucket_widths = prepare_candidate_buckets(
            flat_counts,
            flat_budgets,
            num_blocks=key_blocks,
            subdivisions=subdivisions,
        )
        metadata = metadata_tensor.cpu().tolist()

    bucket_count = len(bucket_widths)
    maximum_blocks, maximum_budget, total_blocks = metadata[:3]
    bucket_row_counts = metadata[3 : 3 + bucket_count]
    count_start = 3 + bucket_count + 1
    bucket_max_counts = metadata[count_start : count_start + bucket_count]
    budget_start = count_start + bucket_count + 1
    bucket_max_budgets = metadata[budget_start : budget_start + bucket_count]

    selected_capacity = max(
        BLOCK_SIZE,
        math.ceil((maximum_budget + 2 * BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE,
    )
    # Keep unwritten selector slots invalid even if a non-finite score is
    # declined by a mapping kernel.
    output = torch.full(
        (rows, selected_capacity),
        key_states.shape[2],
        device=query_means.device,
        dtype=torch.int32,
    )
    if maximum_blocks == 0 or maximum_budget == 0:
        return output.view(batch, heads, query_blocks, selected_capacity)

    total_bucket_slots = 0
    fused_rows = 0
    variable_rows = 0
    row_start = 0
    for bucket_index, bucket_row_count in enumerate(bucket_row_counts):
        width = bucket_widths[bucket_index]
        if width >= 128:
            width = bucket_max_counts[bucket_index]
        if bucket_row_count <= 0:
            continue

        bucket_rows = row_order[row_start : row_start + bucket_row_count]
        row_start += bucket_row_count
        with collector.stage("token_sparse.layout.candidate_mask", query_means):
            bucket_blocks = gather_candidate_bucket(
                candidate_ids, compact_counts, bucket_rows, width
            )
            token_width = width * BLOCK_SIZE
            local_topk = bucket_budget_upper_bound(width, maximum_budget)
            local_topk = min(local_topk, bucket_max_budgets[bucket_index])
            total_bucket_slots += bucket_row_count * width

        if token_width <= 256:
            with collector.stage("token_sparse.layout.fused_score_topk", query_means):
                fused_short_token_select(
                    query_means,
                    key_states,
                    bucket_rows,
                    bucket_blocks,
                    q_last_positions,
                    flat_budgets,
                    output,
                )
            fused_rows += bucket_row_count
            continue

        with collector.stage("token_sparse.layout.score", query_means):
            scores = candidate_block_token_scores_exact(
                query_means,
                key_states,
                bucket_rows,
                bucket_blocks,
                q_last_positions,
                block_size=BLOCK_SIZE,
                num_key_value_groups=heads // key_states.shape[1],
                causal=True,
                output_dtype=torch.float32,
                aligned_causal_full_blocks=True,
            )

        with collector.stage("token_sparse.layout.topk", query_means):
            if token_width > 8192 and try_variable_topk_to_token_positions(
                scores,
                bucket_blocks,
                bucket_rows,
                flat_budgets,
                output,
                topk=local_topk,
                block_size=BLOCK_SIZE,
                kv_len=key_states.shape[2],
            ):
                variable_rows += bucket_row_count
            elif token_width <= 8192 and local_topk <= 2048:
                direct_topk_to_token_positions(
                    scores,
                    bucket_blocks,
                    bucket_rows,
                    flat_budgets,
                    output,
                    topk=local_topk,
                    block_size=BLOCK_SIZE,
                    kv_len=key_states.shape[2],
                )
            elif 4096 < local_topk <= 8192:
                values, indices = torch.topk(
                    scores, k=local_topk, dim=-1, largest=True, sorted=False
                )
                radix_budget_to_token_positions(
                    values,
                    indices,
                    bucket_blocks,
                    bucket_rows,
                    flat_budgets,
                    output,
                    block_size=BLOCK_SIZE,
                    kv_len=key_states.shape[2],
                    adaptive=True,
                )
            else:
                values, indices = torch.topk(
                    scores, k=local_topk, dim=-1, largest=True, sorted=True
                )
                compact_sorted_topk_to_token_positions(
                    values,
                    indices,
                    bucket_blocks,
                    bucket_rows,
                    flat_budgets,
                    output,
                    block_size=BLOCK_SIZE,
                    kv_len=key_states.shape[2],
                )
        del scores

    if collector.collect_stage_time:
        collector.record_stage_value("token_sparse.fused_score_topk_rows", fused_rows)
        collector.record_stage_value("token_sparse.variable_topk_rows", variable_rows)
        collector.record_stage_value("token_sparse.compact_candidate_blocks", total_blocks)
        collector.record_stage_value("token_sparse.bucket_block_slots", total_bucket_slots)
        collector.record_stage_value(
            "token_sparse.bucket_capacity_utilization",
            total_blocks / max(total_bucket_slots, 1),
        )
    return output.view(batch, heads, query_blocks, selected_capacity)
