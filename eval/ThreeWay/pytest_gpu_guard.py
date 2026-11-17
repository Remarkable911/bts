"""Shared pytest guard for the formal single-GPU benchmark.

Both the repository-root and FastDeploy-local conftests import this module so
the guard still applies when pytest selects either directory as its rootdir.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import pytest


SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
CUDA_TEST_OPT_IN = "BTS_REPO_TEST_CUDA"
_LOCKS: list[Any] = []
_CONFIGURED_SESSIONS: set[int] = set()


def prepare_environment() -> None:
    """Hide CUDA before pytest imports any test module unless explicitly opted in."""

    if os.environ.get(CUDA_TEST_OPT_IN) != "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _repository_lock_path() -> Path:
    identity = hashlib.sha256(str(SUBMISSION_ROOT).encode("utf-8")).hexdigest()[:16]
    return Path("/tmp") / f"bts-three-way-{identity}.lock"


def _resolve_selected_gpu() -> dict[str, str]:
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selector or "," in selector:
        raise pytest.UsageError(
            f"{CUDA_TEST_OPT_IN}=1 requires CUDA_VISIBLE_DEVICES to name exactly "
            "one physical GPU index or UUID"
        )
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    inventory = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2:
            inventory.append({"index": fields[0], "uuid": fields[1]})
    if selector.isdigit():
        matches = [row for row in inventory if row["index"] == selector]
    else:
        matches = [row for row in inventory if row["uuid"].startswith(selector)]
    if len(matches) != 1:
        raise pytest.UsageError(
            f"cannot resolve CUDA_VISIBLE_DEVICES={selector!r} to one physical GPU"
        )
    return matches[0]


def _gpu_lock_paths(gpu_uuid: str) -> tuple[Path, ...]:
    safe_uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", gpu_uuid)
    lock_dir = Path("/tmp")
    return (
        lock_dir / f"bts-vllm-queue-{safe_uuid}.lock",
        lock_dir / f"prism-fastdeploy-{safe_uuid}.lock",
        lock_dir / f"bts-fastdeploy-{safe_uuid}.lock",
    )


def _acquire(path: Path, operation: int) -> Any:
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), operation | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def _release_all() -> None:
    while _LOCKS:
        stream = _LOCKS.pop()
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def configure(config: Any) -> None:
    """Acquire locks once even when both conftests are loaded in one session."""

    session_id = id(config)
    if session_id in _CONFIGURED_SESSIONS:
        return
    config.addinivalue_line(
        "markers", "cuda: requires BTS_REPO_TEST_CUDA=1 and one allocated GPU"
    )
    if os.environ.get(CUDA_TEST_OPT_IN) == "1":
        num_processes = getattr(config.option, "numprocesses", 0)
        if hasattr(config, "workerinput") or num_processes not in (0, None):
            raise pytest.UsageError(
                "explicit CUDA pytest sessions do not support pytest-xdist; "
                "run one local pytest process so one process owns the GPU locks"
            )
    try:
        # Every pytest session holds a shared repository lock.  The formal
        # controller holds it exclusively from before GPU acquisition until
        # after its CUDA lease is released.
        _LOCKS.append(_acquire(_repository_lock_path(), fcntl.LOCK_SH))
        if os.environ.get(CUDA_TEST_OPT_IN) == "1":
            gpu = _resolve_selected_gpu()
            for path in _gpu_lock_paths(gpu["uuid"]):
                _LOCKS.append(_acquire(path, fcntl.LOCK_EX))
    except BlockingIOError as error:
        _release_all()
        raise pytest.UsageError(
            "formal benchmark or another explicit CUDA test session holds the "
            "required repository/GPU lock"
        ) from error
    except BaseException:
        _release_all()
        raise
    _CONFIGURED_SESSIONS.add(session_id)


def unconfigure(config: Any) -> None:
    session_id = id(config)
    if session_id not in _CONFIGURED_SESSIONS:
        return
    _CONFIGURED_SESSIONS.remove(session_id)
    _release_all()


prepare_environment()
