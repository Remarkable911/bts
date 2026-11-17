#!/usr/bin/env python3
"""Display a read-only tqdm view of a running ThreeWay benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


TOTAL_WORKERS = 20
TOTAL_RESTARTS = 5
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


def _postfix(manifest: Mapping[str, Any]) -> str:
    active = manifest.get("active_arm")
    if not isinstance(active, dict):
        return f"stage={manifest.get('status', 'starting')}"
    restart = active.get("restart")
    attempt = active.get("attempt")
    arm = str(active.get("arm", "unknown"))
    phase = str(active.get("phase", "running"))
    fields = [
        f"restart={int(restart) + 1}/{TOTAL_RESTARTS}"
        if isinstance(restart, int)
        else "restart=?/5",
        f"arm={ARM_LABELS.get(arm, arm)}",
        f"attempt={int(attempt) + 1}" if isinstance(attempt, int) else "attempt=?",
        f"phase={PHASE_LABELS.get(phase, phase)}",
    ]
    elapsed = _elapsed(active.get("started_at"))
    if elapsed is not None:
        fields.append(f"step={elapsed}")
    return " | ".join(fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        help="ThreeWay output directory or its manifest.json",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        print("error: --poll-seconds must be positive", file=sys.stderr)
        return 2
    manifest_path = (
        args.output / "manifest.json" if args.output.is_dir() else args.output
    ).resolve()
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        from tqdm import tqdm
    except ImportError:
        print("error: tqdm is not installed in this Python environment", file=sys.stderr)
        return 2

    manifest = _load_manifest(manifest_path)
    initial = min(TOTAL_WORKERS, len(manifest.get("completed_arms", [])))
    bar = tqdm(
        total=TOTAL_WORKERS,
        initial=initial,
        desc="BTS 128K workers",
        unit="worker",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )
    try:
        while True:
            try:
                manifest = _load_manifest(manifest_path)
            except (OSError, json.JSONDecodeError, ValueError):
                time.sleep(args.poll_seconds)
                continue
            completed = min(
                TOTAL_WORKERS, len(manifest.get("completed_arms", []))
            )
            if completed > bar.n:
                bar.update(completed - bar.n)
            elif completed < bar.n:
                bar.n = completed
            bar.set_postfix_str(_postfix(manifest), refresh=False)
            bar.refresh()
            if manifest.get("status") in {"completed", "failed"}:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return 130
    finally:
        bar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
