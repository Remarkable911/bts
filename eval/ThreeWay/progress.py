#!/usr/bin/env python3
"""Non-invasive terminal progress for the ThreeWay controller."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Mapping, TextIO


ARM_LABELS = {
    "fd_dense": "FastDeploy Dense",
    "fd_bts": "FastDeploy BTS",
    "vllm_dense": "vLLM Dense",
    "vllm_bts": "vLLM BTS",
}

PHASE_LABELS = {
    "worker_running": "running",
    "validating": "validating",
    "waiting_for_lease_only": "waiting for exclusive GPU",
    "waiting_for_lease_only_after_worker_failure": "recovering worker",
    "contamination_retry_exhausted": "GPU retry limit reached",
    "worker_failure_retry_exhausted": "worker retry limit reached",
    "non_retryable_failure": "controller error",
}


def _load_manifest(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return value


def _elapsed(started_at: Any) -> str | None:
    if not isinstance(started_at, str):
        return None
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    seconds = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def manifest_progress_state(
    manifest: Mapping[str, Any], *, total: int, restart_count: int
) -> tuple[int, str]:
    completed_rows = manifest.get("completed_arms")
    completed = len(completed_rows) if isinstance(completed_rows, list) else 0
    completed = max(0, min(total, completed))
    active = manifest.get("active_arm")
    if not isinstance(active, dict):
        return completed, f"stage={manifest.get('status', 'starting')}"

    restart = active.get("restart")
    attempt = active.get("attempt")
    arm = str(active.get("arm", "unknown"))
    phase = str(active.get("phase", "running"))
    fields = [
        f"restart={int(restart) + 1}/{restart_count}"
        if isinstance(restart, int)
        else f"restart=?/{restart_count}",
        f"arm={ARM_LABELS.get(arm, arm)}",
        f"attempt={int(attempt) + 1}" if isinstance(attempt, int) else "attempt=?",
        f"phase={PHASE_LABELS.get(phase, phase)}",
    ]
    elapsed = _elapsed(active.get("started_at"))
    if elapsed is not None:
        fields.append(f"step={elapsed}")
    return completed, " | ".join(fields)


def _default_bar_factory() -> Callable[..., Any]:
    from tqdm import tqdm

    return tqdm


class ManifestProgress:
    """Refresh a tqdm bar from the controller's atomically written manifest."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        total: int,
        restart_count: int,
        refresh_seconds: float = 5.0,
        stream: TextIO = sys.stderr,
        enabled: bool | None = None,
        bar_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.total = total
        self.restart_count = restart_count
        self.refresh_seconds = refresh_seconds
        self.stream = stream
        self.enabled = stream.isatty() if enabled is None else enabled
        self.bar_factory = bar_factory
        self._bar: Any | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.enabled or self._bar is not None:
            return
        try:
            factory = self.bar_factory or _default_bar_factory()
            manifest = _load_manifest(self.manifest_path)
            completed, postfix = manifest_progress_state(
                manifest,
                total=self.total,
                restart_count=self.restart_count,
            )
            self._bar = factory(
                total=self.total,
                initial=completed,
                desc="BTS 128K workers",
                unit="worker",
                dynamic_ncols=True,
                leave=True,
                file=self.stream,
                mininterval=0.2,
            )
            self._bar.set_postfix_str(postfix, refresh=True)
        except (ImportError, OSError, json.JSONDecodeError, ValueError) as error:
            self.enabled = False
            print(f"progress display unavailable: {error}", file=self.stream)
            return
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name="bts-three-way-progress",
            daemon=True,
        )
        self._thread.start()

    def refresh(self) -> None:
        bar = self._bar
        if bar is None:
            return
        try:
            manifest = _load_manifest(self.manifest_path)
            completed, postfix = manifest_progress_state(
                manifest,
                total=self.total,
                restart_count=self.restart_count,
            )
        except (OSError, json.JSONDecodeError, ValueError):
            return
        with self._lock:
            delta = completed - int(bar.n)
            if delta > 0:
                bar.update(delta)
            elif delta < 0:
                bar.n = completed
            bar.set_postfix_str(postfix, refresh=False)
            bar.refresh()

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self.refresh_seconds):
            try:
                self.refresh()
            except Exception:
                return

    def close(self) -> None:
        bar = self._bar
        if bar is None:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            try:
                # The display must never delay GPU/process cleanup indefinitely.
                thread.join(timeout=1.0)
            except BaseException:
                self._bar = None
                self._thread = None
                raise
            if thread.is_alive():
                # It may be stuck in a terminal write while holding _lock.  Do not
                # touch the same bar again; the daemon will observe _stop when the
                # write eventually returns.
                self._bar = None
                self._thread = None
                return

        try:
            manifest = _load_manifest(self.manifest_path)
            state = manifest_progress_state(
                manifest,
                total=self.total,
                restart_count=self.restart_count,
            )
        except (OSError, json.JSONDecodeError, ValueError):
            state = None

        # A timed acquire makes close best-effort even if an interrupted refresh
        # left another thread in the presentation path.
        acquired = self._lock.acquire(timeout=0.5)
        if not acquired:
            self._bar = None
            self._thread = None
            return
        try:
            if state is not None:
                completed, postfix = state
                delta = completed - int(bar.n)
                if delta > 0:
                    bar.update(delta)
                elif delta < 0:
                    bar.n = completed
                bar.set_postfix_str(postfix, refresh=False)
                bar.refresh()
            bar.close()
        finally:
            self._bar = None
            self._thread = None
            self._lock.release()
