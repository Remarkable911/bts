"""Low-overhead, SIGTERM-resilient counters for the BTS vLLM backend."""

from __future__ import annotations

import mmap
import os
from pathlib import Path
import struct
import threading
from typing import Mapping


AUDIT_ENV = "BTS_VLLM_AUDIT_DIR"
MAGIC = b"BTSV017\0"
SCHEMA_VERSION = 1
COUNTER_NAMES = (
    "backend_activations",
    "metadata_builders",
    "impl_instances",
    "sparse_calls",
    "sparse_segments",
    "dense_fallback_calls",
    "dense_fallback_segments",
    "mixed_calls",
)
_RECORD = struct.Struct("<8sII" + "Q" * len(COUNTER_NAMES))
_MMAP_BYTES = 4096


def _public_record(
    *, path: str | None, schema: int, pid: int, values: Mapping[str, int]
) -> dict[str, int | str | None]:
    """Expose stable runner-facing names while retaining raw counters."""

    return {
        "path": path,
        "schema_version": schema,
        "pid": pid,
        "backend_mode": "bts",
        "impl_count": values["impl_instances"],
        **values,
    }


class Audit:
    """Process-local counters optionally mirrored to one mmap file per PID."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pid = -1
        self._values = {name: 0 for name in COUNTER_NAMES}
        self._mapping: mmap.mmap | None = None
        self._file = None
        self._path: Path | None = None

    def _close_locked(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
        if self._file is not None:
            self._file.close()
        self._mapping = None
        self._file = None
        self._path = None

    def _ensure_process_locked(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self._close_locked()
            self._pid = pid
            self._values = {name: 0 for name in COUNTER_NAMES}

        requested_dir = os.environ.get(AUDIT_ENV)
        if self._mapping is not None or not requested_dir:
            return
        directory = Path(requested_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"worker-{pid}.mmap"
        file_object = path.open("w+b")
        file_object.truncate(_MMAP_BYTES)
        mapping = mmap.mmap(file_object.fileno(), _MMAP_BYTES, access=mmap.ACCESS_WRITE)
        self._file = file_object
        self._mapping = mapping
        self._path = path
        self._write_locked()

    def _write_locked(self) -> None:
        if self._mapping is None:
            return
        _RECORD.pack_into(
            self._mapping,
            0,
            MAGIC,
            SCHEMA_VERSION,
            self._pid,
            *(self._values[name] for name in COUNTER_NAMES),
        )

    def record(self, **increments: int) -> None:
        unknown = set(increments).difference(COUNTER_NAMES)
        if unknown:
            raise KeyError(f"unknown BTS vLLM audit counters: {sorted(unknown)}")
        if any(
            not isinstance(value, int) or value < 0
            for value in increments.values()
        ):
            raise ValueError("audit increments must be non-negative integers")
        with self._lock:
            self._ensure_process_locked()
            for name, amount in increments.items():
                self._values[name] += amount
            self._write_locked()

    def snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            self._ensure_process_locked()
            return _public_record(
                path=None if self._path is None else str(self._path),
                schema=SCHEMA_VERSION,
                pid=self._pid,
                values=self._values,
            )


def read_audit(path: str | os.PathLike[str]) -> dict[str, int | str]:
    """Read a live or completed worker mmap without importing vLLM."""

    audit_path = Path(path)
    raw = audit_path.read_bytes()[: _RECORD.size]
    if len(raw) != _RECORD.size:
        raise ValueError(f"truncated BTS vLLM audit record: {audit_path}")
    magic, schema, pid, *values = _RECORD.unpack(raw)
    if magic != MAGIC or schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported BTS vLLM audit record: {audit_path}")
    counters = dict(zip(COUNTER_NAMES, values, strict=True))
    return _public_record(
        path=str(audit_path), schema=schema, pid=pid, values=counters
    )


def read_audits(
    directory: str | os.PathLike[str],
) -> list[dict[str, int | str | None]]:
    """Read all per-process records from ``directory`` in stable order."""

    return [read_audit(path) for path in sorted(Path(directory).glob("worker-*.mmap"))]


def snapshot() -> Mapping[str, int | str | None]:
    return AUDIT.snapshot()


AUDIT = Audit()


__all__ = [
    "AUDIT",
    "AUDIT_ENV",
    "COUNTER_NAMES",
    "read_audit",
    "read_audits",
    "snapshot",
]
