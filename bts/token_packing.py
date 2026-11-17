"""Fixed-width packed token rows for direct attention."""

from dataclasses import dataclass

import torch

from .config import BLOCK_SIZE
from .kernels.pack_token_rows import append_required_to_prepacked_rows_cuda


@dataclass(frozen=True)
class PackedTokenLayout:
    packed_k_indices: torch.Tensor
    packed_token_counts: torch.Tensor

    @property
    def tokens_per_row(self) -> int:
        return self.packed_k_indices.shape[-1] // self.packed_token_counts.shape[-1]


def required_sink_recent(
    block_mask: torch.Tensor,
    q_last_positions: torch.Tensor,
    *,
    kv_len: int,
) -> torch.Tensor:
    """Materialize selected sink and recent blocks in that stable order."""
    batch, heads, rows, key_blocks = block_mask.shape
    offsets = torch.arange(BLOCK_SIZE, device=block_mask.device, dtype=torch.long)
    sentinel = torch.full(
        (batch, heads, rows, BLOCK_SIZE),
        kv_len,
        device=block_mask.device,
        dtype=torch.long,
    )
    if key_blocks == 0:
        return torch.cat((sentinel, sentinel), dim=-1)

    sink_positions = offsets.view(1, 1, 1, -1).expand_as(sentinel)
    sink_valid = block_mask[..., 0, None]
    sink_valid = sink_valid & (sink_positions <= q_last_positions.view(1, 1, rows, 1))
    sink_valid = sink_valid & (sink_positions < kv_len)
    sink = torch.where(sink_valid, sink_positions, sentinel)

    recent_blocks = torch.div(q_last_positions, BLOCK_SIZE, rounding_mode="floor")
    safe_recent = recent_blocks.clamp(0, key_blocks - 1)
    gather = safe_recent.view(1, 1, rows, 1).expand(batch, heads, rows, 1)
    recent_selected = block_mask.gather(-1, gather).squeeze(-1)
    recent_positions = (
        recent_blocks.view(1, 1, rows, 1) * BLOCK_SIZE
        + offsets.view(1, 1, 1, BLOCK_SIZE)
    ).expand_as(sentinel)
    recent_valid = recent_selected[..., None]
    recent_valid = recent_valid & (recent_blocks.view(1, 1, rows, 1) != 0)
    recent_valid = recent_valid & (recent_positions <= q_last_positions.view(1, 1, rows, 1))
    recent_valid = recent_valid & (recent_positions < kv_len)
    recent = torch.where(recent_valid, recent_positions, sentinel)
    return torch.cat((sink, recent), dim=-1)


def finalize_layout(
    selected: torch.Tensor,
    selected_counts: torch.Tensor,
    required: torch.Tensor,
    *,
    kv_len: int,
) -> PackedTokenLayout:
    """Append required blocks in-place and flatten private query rows."""
    if selected.is_cuda:
        selected, counts = append_required_to_prepacked_rows_cuda(
            selected,
            selected_counts,
            required,
            kv_len=kv_len,
            selected_count_upper_bound=selected.shape[-1] - required.shape[-1],
        )
    else:
        counts = selected_counts.to(torch.int32).clone()
        for row in range(selected.shape[2]):
            for batch in range(selected.shape[0]):
                for head in range(selected.shape[1]):
                    start = int(counts[batch, head, row])
                    values = required[batch, head, row]
                    values = values[(values >= 0) & (values < kv_len)]
                    selected[batch, head, row, start : start + values.numel()] = values
                    counts[batch, head, row] += values.numel()

    return PackedTokenLayout(
        selected.reshape(selected.shape[0], selected.shape[1], -1),
        counts,
    )
