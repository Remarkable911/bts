#!/usr/bin/env python3
"""Queue one vLLM evaluation until an H100 is stably idle.

This host does not provide a cluster scheduler.  The launcher therefore runs a
single waiter inside a detached tmux session, selects one physical H100 after
several consecutive idle checks, and then starts ``run.py`` exactly once.
GPU locks coordinate only queues created by this launcher; ``run.py`` remains
the authority for exclusivity checks against every external process.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
SUBMISSION_ROOT = HERE.parents[1]
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from eval.VLLM import run  # noqa: E402


RUNNER = HERE / "run.py"


def parse_gpu_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpu-indices must contain integers") from error
    if not indices or len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(
            "--gpu-indices must contain unique non-negative integers"
        )
    return indices


def strip_separator(arguments: Sequence[str]) -> list[str]:
    result = list(arguments)
    if result[:1] == ["--"]:
        result.pop(0)
    if not result:
        raise ValueError("runner arguments are required after --")
    return result


def parse_runner_contract(arguments: Sequence[str]) -> argparse.Namespace:
    runner_args = run.parser().parse_args(strip_separator(arguments))
    if runner_args.dry_run:
        raise ValueError("the queued command must be a real run, not --dry-run")
    if not runner_args.model or runner_args.workload is None or runner_args.output is None:
        raise ValueError("queued runner requires --model, --workload and --output")
    output = runner_args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"runner output must be a new or empty directory: {output}")
    return runner_args


def eligible_gpus(
    inventory: Sequence[Mapping[str, Any]],
    requested_indices: Sequence[int] | None,
) -> list[dict[str, Any]]:
    requested = set(requested_indices) if requested_indices is not None else None
    result = [
        dict(gpu)
        for gpu in inventory
        if "H100" in str(gpu.get("name", ""))
        and (requested is None or int(gpu["physical_index"]) in requested)
    ]
    if requested is not None:
        found = {int(gpu["physical_index"]) for gpu in result}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"requested indices are not physical H100 GPUs: {missing}")
    if not result:
        raise ValueError("no eligible physical H100 GPU was found")
    return sorted(result, key=lambda gpu: int(gpu["physical_index"]))


def queue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-indices",
        type=parse_gpu_indices,
        help="Comma-separated physical H100 indices; default: every local H100",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--session-name")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser


def log_event(queue_dir: Path, message: str) -> None:
    line = f"[{run.utc_now()}] {message}\n"
    with (queue_dir / "queue.log").open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def write_status(queue_dir: Path, state: str, **fields: Any) -> None:
    run.atomic_write_json(
        queue_dir / "status.json",
        {"state": state, "updated_at": run.utc_now(), **fields},
    )


def gpu_lock_paths(gpu: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return every local lease convention used by the project queues."""

    uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(gpu["uuid"]))
    lock_dir = Path("/tmp")
    return (
        lock_dir / f"bts-vllm-queue-{uuid}.lock",
        # The development-tree FastDeploy validator uses this historical name.
        lock_dir / f"prism-fastdeploy-{uuid}.lock",
        # The frozen submission validator uses this name.
        lock_dir / f"bts-fastdeploy-{uuid}.lock",
    )


def output_lock_path(output: Path) -> Path:
    identity = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:24]
    return Path("/tmp") / f"bts-vllm-output-{identity}.lock"


def cuda_visible_device(gpu: Mapping[str, Any]) -> str:
    """Return the physical index required by vLLM 0.17's CUDA platform code."""

    return str(int(gpu["physical_index"]))


def release_gpu_locks(streams: Sequence[Any]) -> None:
    for stream in reversed(streams):
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def try_gpu_locks(gpu: Mapping[str, Any]):
    streams = []
    try:
        for path in gpu_lock_paths(gpu):
            stream = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                stream.close()
                raise
            streams.append(stream)
    except BlockingIOError:
        release_gpu_locks(streams)
        return None
    except BaseException:
        release_gpu_locks(streams)
        raise
    return streams


def select_stably_idle_gpu(
    gpus: Sequence[Mapping[str, Any]],
    streaks: dict[int, int],
    stable_checks: int,
) -> tuple[dict[str, Any] | None, Any | None, dict[str, Any]]:
    snapshot: dict[str, Any] = {}
    for gpu in gpus:
        index = int(gpu["physical_index"])
        processes = run.gpu_processes(gpu)
        snapshot[str(index)] = processes
        streaks[index] = streaks.get(index, 0) + 1 if not processes else 0

    for gpu in gpus:
        index = int(gpu["physical_index"])
        if streaks[index] < stable_checks:
            continue
        locks = try_gpu_locks(gpu)
        if locks is None:
            continue
        processes = run.gpu_processes(gpu)
        snapshot[str(index)] = processes
        if processes:
            streaks[index] = 0
            release_gpu_locks(locks)
            continue
        return dict(gpu), locks, snapshot
    return None, None, snapshot


def read_benchmark_status(output: Path) -> str | None:
    report_path = output / "benchmark_report.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return str(report["comparison"]["overall_status"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def worker(args: argparse.Namespace) -> int:
    queue_dir = args.queue_dir.expanduser().resolve()
    runner_arguments = strip_separator(args.runner_args)
    runner_contract = parse_runner_contract(runner_arguments)
    output = runner_contract.output.expanduser().resolve()
    gpus = eligible_gpus(run.gpu_inventory(), args.gpu_indices)

    queue_lock = (queue_dir / "queue.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"queue already has an active worker: {queue_dir}") from error

    output_lock = output_lock_path(output).open("a+", encoding="utf-8")
    try:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        output_lock.close()
        raise RuntimeError(f"another queue already owns runner output: {output}") from error

    streaks = {int(gpu["physical_index"]): 0 for gpu in gpus}
    candidate_indices = [int(gpu["physical_index"]) for gpu in gpus]
    log_event(
        queue_dir,
        "worker started; candidates="
        f"{candidate_indices}, poll_seconds={args.poll_seconds}, "
        f"stable_checks={args.stable_checks}",
    )
    selected = None
    gpu_locks = None
    while selected is None:
        try:
            selected, gpu_locks, snapshot = select_stably_idle_gpu(
                gpus, streaks, args.stable_checks
            )
            write_status(
                queue_dir,
                "waiting_for_idle_gpu" if selected is None else "gpu_selected",
                worker_pid=os.getpid(),
                candidates=candidate_indices,
                idle_streaks=streaks,
                compute_processes=snapshot,
                selected_gpu=selected,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            streaks = {int(gpu["physical_index"]): 0 for gpu in gpus}
            log_event(queue_dir, f"GPU query failed; retrying: {type(error).__name__}: {error}")
            write_status(
                queue_dir,
                "waiting_for_gpu_query",
                worker_pid=os.getpid(),
                candidates=candidate_indices,
                error_type=type(error).__name__,
                error=str(error),
            )
        if selected is None:
            time.sleep(args.poll_seconds)

    assert gpu_locks is not None
    index = int(selected["physical_index"])
    environment = os.environ.copy()
    # vLLM 0.17 converts CUDA_VISIBLE_DEVICES entries with int().  Keep the
    # UUID for the cooperative lock/audit identity, but expose the physical
    # index to the vLLM process.
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_device(selected)
    environment["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, str(RUNNER), *runner_arguments]
    log_event(queue_dir, f"GPU {index} is stably idle; starting {shlex.join(command)}")
    with (queue_dir / "runner.log").open("w", encoding="utf-8") as runner_log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=runner_log,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
        write_status(
            queue_dir,
            "running",
            worker_pid=os.getpid(),
            runner_pid=process.pid,
            selected_gpu=selected,
            gpu_lease_paths=[str(path) for path in gpu_lock_paths(selected)],
            command=command,
            output=str(output),
        )
        returncode = process.wait()

    benchmark_status = read_benchmark_status(output)
    state = "finished" if returncode in {0, 2, 3} else "failed"
    write_status(
        queue_dir,
        state,
        worker_pid=os.getpid(),
        runner_pid=process.pid,
        selected_gpu=selected,
        returncode=returncode,
        benchmark_status=benchmark_status,
        output=str(output),
    )
    log_event(
        queue_dir,
        f"runner exited with {returncode}; benchmark_status={benchmark_status}",
    )
    release_gpu_locks(gpu_locks)
    output_lock.close()
    queue_lock.close()
    return returncode


def normalized_session_name(queue_dir: Path, requested: str | None) -> str:
    raw = requested or f"bts-vllm-{queue_dir.name}"
    result = re.sub(r"[^A-Za-z0-9_-]", "-", raw).strip("-")
    if not result:
        raise ValueError("--session-name does not contain a usable character")
    return result[:80]


def launch(args: argparse.Namespace) -> int:
    queue_dir = args.queue_dir.expanduser().resolve()
    if queue_dir.exists():
        raise ValueError(f"--queue-dir must not already exist: {queue_dir}")
    runner_arguments = strip_separator(args.runner_args)
    runner_contract = parse_runner_contract(runner_arguments)
    inventory = eligible_gpus(run.gpu_inventory(), args.gpu_indices)
    queue_dir.mkdir(parents=True)
    session_name = normalized_session_name(queue_dir, args.session_name)
    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--queue-dir",
        str(queue_dir),
        "--poll-seconds",
        str(args.poll_seconds),
        "--stable-checks",
        str(args.stable_checks),
    ]
    if args.gpu_indices is not None:
        worker_command.extend(
            ("--gpu-indices", ",".join(str(index) for index in args.gpu_indices))
        )
    worker_command.extend(("--", *runner_arguments))
    tmux_command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session_name,
        f"exec {shlex.join(worker_command)}",
    ]
    launch_record = {
        "created_at": run.utc_now(),
        "queue_dir": str(queue_dir),
        "session_name": session_name,
        "candidate_gpus": inventory,
        "poll_seconds": args.poll_seconds,
        "stable_checks": args.stable_checks,
        "python": sys.executable,
        "worker_command": worker_command,
        "runner_output": str(runner_contract.output.expanduser().resolve()),
    }
    run.atomic_write_json(queue_dir / "launch.json", launch_record)
    write_status(
        queue_dir,
        "queued",
        session_name=session_name,
        candidates=[int(gpu["physical_index"]) for gpu in inventory],
    )
    try:
        subprocess.run(tmux_command, check=True, capture_output=True, text=True)
    except BaseException:
        write_status(queue_dir, "launch_failed", tmux_command=tmux_command)
        raise
    print(json.dumps(launch_record, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = queue_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if args.stable_checks <= 0:
        raise SystemExit("--stable-checks must be positive")
    try:
        return worker(args) if args.worker else launch(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        if args.worker:
            queue_dir = args.queue_dir.expanduser().resolve()
            if queue_dir.is_dir():
                write_status(
                    queue_dir,
                    "failed",
                    worker_pid=os.getpid(),
                    failure_type=type(error).__name__,
                    failure=str(error),
                )
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
