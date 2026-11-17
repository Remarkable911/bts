import os
from contextlib import contextmanager
from typing import Optional

import torch


class StatCollector:
    def __init__(
        self,
        collect_density: bool = False,
        collect_select_time: bool = False,
        collect_stage_time: bool = False,
        method_name: str = "bts",
    ) -> None:
        self.collect_density = collect_density
        self.collect_select_time = collect_select_time
        self.collect_stage_time = collect_stage_time
        self.method_name = method_name
        self._densities = []
        self._events = []
        self._stage_events = {}
        self._stage_values = {}

    @classmethod
    def from_env(cls, method_name: str = "bts") -> "StatCollector":
        collect_density = os.environ.get("COLLECT_DENSITY", "false").lower() in ("1", "true", "yes")
        collect_select_time = os.environ.get("COLLECT_SELECT_TIME", "false").lower() in ("1", "true", "yes")
        collect_stage_time = os.environ.get("COLLECT_BTS_STAGE_TIME", "false").lower() in ("1", "true", "yes")
        return cls(
            collect_density=collect_density,
            collect_select_time=collect_select_time,
            collect_stage_time=collect_stage_time,
            method_name=method_name,
        )

    def reset(self) -> None:
        self._densities = []
        self._events = []
        self._stage_events = {}
        self._stage_values = {}

    def reset_density(self) -> None:
        self._densities = []

    def reset_select_time(self) -> None:
        self._events = []

    def reset_stage_time(self) -> None:
        self._stage_events = {}

    def reset_stage_values(self) -> None:
        self._stage_values = {}

    def maybe_start_select_timer(self, query_states: torch.Tensor):
        if self.collect_select_time and query_states.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end
        return None, None

    def finish_select_timer(self, start, end) -> None:
        if start is None or end is None:
            return
        end.record()
        self._events.append((start, end))

    def drain_select_time_ms(self) -> float:
        if not self._events:
            return 0.0
        total_ms = 0.0
        for start, end in self._events:
            total_ms += start.elapsed_time(end)
        self._events = []
        return total_ms

    def maybe_start_stage_timer(self, name: str, tensor: Optional[torch.Tensor]):
        if self.collect_stage_time and tensor is not None and tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return name, start, end
        return None

    def finish_stage_timer(self, handle) -> None:
        if handle is None:
            return
        name, start, end = handle
        end.record()
        self._stage_events.setdefault(name, []).append((start, end))

    def drain_stage_times_ms(self) -> dict:
        if not self._stage_events:
            return {}
        stage_times = {}
        for name, events in self._stage_events.items():
            total_ms = 0.0
            for start, end in events:
                total_ms += start.elapsed_time(end)
            stage_times[name] = total_ms
        self._stage_events = {}
        return stage_times

    def record_stage_value(self, name: str, value: float | torch.Tensor) -> None:
        if self.collect_stage_time:
            # Scalar GPU diagnostics can be copied when results are drained,
            # outside the measured forward, rather than synchronizing here.
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError("Stage values must be scalar tensors.")
                value = value.detach()
            else:
                value = float(value)
            self._stage_values.setdefault(name, []).append(value)

    def drain_stage_values(self) -> dict:
        if not self._stage_values:
            return {}
        stage_values = {
            name: float(sum(values))
            for name, values in self._stage_values.items()
        }
        self._stage_values = {}
        return stage_values

    @contextmanager
    def stage(self, name: str, tensor: Optional[torch.Tensor]):
        handle = self.maybe_start_stage_timer(name, tensor)
        try:
            yield
        finally:
            self.finish_stage_timer(handle)

    def record_block_mask(
        self,
        block_mask: torch.Tensor,
        *,
        record_density: bool = True,
    ) -> None:
        q_block_num = block_mask.shape[2]
        kv_block_num = block_mask.shape[3]
        num_causal_blocks_per_head = q_block_num * (kv_block_num - q_block_num) + (q_block_num * (q_block_num + 1)) // 2
        total_causal_numel = block_mask.shape[0] * block_mask.shape[1] * num_causal_blocks_per_head
        if total_causal_numel <= 0:
            return
        selected_blocks_per_q = block_mask.sum(dim=-1)
        total_selected_blocks = float(selected_blocks_per_q.sum().item())
        if record_density:
            self._densities.append(total_selected_blocks / total_causal_numel)
        if self.collect_stage_time:
            num_query_rows = block_mask.shape[0] * block_mask.shape[1] * q_block_num
            self.record_stage_value("bts.total_causal_blocks", total_causal_numel)
            self.record_stage_value("bts.total_selected_blocks", total_selected_blocks)
            self.record_stage_value(
                "bts.selected_blocks_per_q_mean",
                total_selected_blocks / num_query_rows,
            )
            self.record_stage_value(
                "bts.selected_blocks_per_q_max",
                selected_blocks_per_q.max().item(),
            )

    def record_density(self, density: float) -> None:
        self._densities.append(float(density))

    def replace_latest_density(self, density: float) -> None:
        density = float(density)
        if self._densities:
            self._densities[-1] = density
        else:
            self._densities.append(density)

    def summary(self) -> Optional[dict]:
        if not self._densities:
            return None
        total = len(self._densities)
        avg = sum(self._densities) / total
        return {
            "method": self.method_name,
            "total": avg,
            "count": total,
        }

    def collect(self, fn):
        """Decorator for estimate functions that return a block_mask."""
        def wrapped(*args, **kwargs):
            query_states = args[0] if args else None
            start, end = self.maybe_start_select_timer(query_states)
            block_mask = fn(*args, **kwargs)
            self.finish_select_timer(start, end)
            if self.collect_density or self.collect_stage_time:
                self.record_block_mask(
                    block_mask,
                    record_density=self.collect_density,
                )
            return block_mask

        return wrapped
