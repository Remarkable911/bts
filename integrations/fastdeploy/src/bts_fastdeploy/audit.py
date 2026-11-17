"""Optional per-worker audit records for verifying real BTS execution."""

import atexit
import json
import mmap
import os
from pathlib import Path
import struct

from . import METHOD_DISPLAY_NAME


_COUNTER_MAGIC = b"BTSAUD1\0"
_COUNTER_VERSION = 1
_COUNTER_HEADER = struct.Struct("<8sII")
_UINT64 = struct.Struct("<Q")
_DEPENDENCY_EVIDENCE = {}


def _dependency_evidence():
    require_runtime_imports = (
        os.environ.get("BTS_FASTDEPLOY_REQUIRE_DEPENDENCY_CONTRACT") == "1"
    )
    if require_runtime_imports not in _DEPENDENCY_EVIDENCE:
        from .upstream_contract import (
            dependency_source_contract,
            validate_runtime_dependency_imports,
        )

        contract = dependency_source_contract()
        _DEPENDENCY_EVIDENCE[require_runtime_imports] = {
            "contract": contract,
            "runtime_imports": (
                validate_runtime_dependency_imports(contract)
                if require_runtime_imports
                else {}
            ),
        }
    return _DEPENDENCY_EVIDENCE[require_runtime_imports]


class Audit:
    def __init__(
        self,
        config_sha256,
        directory=None,
        *,
        method_display_name=METHOD_DISPLAY_NAME,
        backend_mode="bts",
    ):
        from .environment import package_provenance

        dependency_evidence = _dependency_evidence()

        self.directory = Path(directory) if directory else None
        self.flush_mode = os.environ.get("BTS_FASTDEPLOY_AUDIT_FLUSH", "immediate")
        if self.flush_mode not in ("immediate", "atexit", "mmap"):
            raise ValueError(
                "BTS_FASTDEPLOY_AUDIT_FLUSH must be 'immediate', 'atexit', "
                "or 'mmap'"
            )
        self.num_layers = None
        self._counter_map = None
        if self.flush_mode == "mmap":
            raw_num_layers = os.environ.get("BTS_FASTDEPLOY_AUDIT_NUM_LAYERS")
            if self.directory is None or raw_num_layers is None:
                raise ValueError(
                    "mmap audit mode requires an audit directory and "
                    "BTS_FASTDEPLOY_AUDIT_NUM_LAYERS"
                )
            self.num_layers = int(raw_num_layers)
            if self.num_layers <= 0:
                raise ValueError("BTS_FASTDEPLOY_AUDIT_NUM_LAYERS must be positive")
        self.data = dict(pid=os.getpid(), method_display_name=method_display_name,
                         backend_mode=backend_mode,
                         config_sha256=config_sha256,
                         dependency_contract_sha256=dependency_evidence["contract"]["sha256"],
                         dependency_contract=dependency_evidence["contract"],
                         runtime_dependency_imports=dependency_evidence["runtime_imports"],
                         package_provenance=package_provenance(), prefill_calls=0,
                         native_decode_calls=0, layers={}, last_lengths=[],
                         rope_guard_calls=0, rope_guard_layers={})
        if self.flush_mode == "mmap":
            self.data["counter_transport"] = "mmap_v1"
            self.data["num_hidden_layers"] = self.num_layers
            self._initialize_counter_map()
            # Metadata is written before model loading/timing. Dynamic counters
            # remain in the shared mapping and survive SIGTERM without hooks.
            self.flush()
        if self.directory and self.flush_mode == "atexit":
            # General callers can buffer until a normal interpreter exit.
            atexit.register(self.flush)

    def _initialize_counter_map(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"worker-{os.getpid()}.counts"
        counter_count = 3 + 2 * self.num_layers
        size = _COUNTER_HEADER.size + counter_count * _UINT64.size
        created = False
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError:
            descriptor = os.open(path, os.O_RDWR)
        try:
            if created:
                os.ftruncate(descriptor, size)
            elif os.fstat(descriptor).st_size != size:
                raise RuntimeError("existing BTS audit counter mapping has wrong size")
            counter_map = mmap.mmap(descriptor, size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(descriptor)
        if created:
            _COUNTER_HEADER.pack_into(
                counter_map,
                0,
                _COUNTER_MAGIC,
                _COUNTER_VERSION,
                self.num_layers,
            )
        else:
            magic, version, num_layers = _COUNTER_HEADER.unpack_from(counter_map)
            if (magic, version, num_layers) != (
                _COUNTER_MAGIC,
                _COUNTER_VERSION,
                self.num_layers,
            ):
                counter_map.close()
                raise RuntimeError("existing BTS audit counter mapping is incompatible")
        self._counter_map = counter_map

    def _increment_counter(self, index, amount=1):
        offset = _COUNTER_HEADER.size + int(index) * _UINT64.size
        value = _UINT64.unpack_from(self._counter_map, offset)[0] + amount
        _UINT64.pack_into(self._counter_map, offset, value)
        return value

    def flush(self):
        if not self.directory:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"worker-{os.getpid()}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        temporary.replace(target)

    def record(self, layer, lengths=None):
        if lengths is None:
            if self._counter_map is None:
                self.data["native_decode_calls"] += 1
            else:
                self.data["native_decode_calls"] = self._increment_counter(1)
        else:
            layer = int(layer)
            sequence_count = sum(int(length) > 0 for length in lengths)
            if sequence_count < 1:
                raise ValueError("BTS prefill audit requires at least one active sequence")
            if self._counter_map is None:
                self.data["prefill_calls"] += sequence_count
            else:
                self.data["prefill_calls"] = self._increment_counter(
                    0, sequence_count
                )
            layers = self.data["layers"]
            if self._counter_map is None:
                layers[str(layer)] = layers.get(str(layer), 0) + sequence_count
            else:
                layers[str(layer)] = self._increment_counter(
                    3 + layer, sequence_count
                )
            self.data["last_lengths"] = list(lengths)
        if self.directory and self.flush_mode == "immediate":
            self.flush()

    def record_rope_guard(self, layer):
        """Record the semantic-only PaddleFormers identity-RoPE guard."""
        layer = int(layer)
        if self._counter_map is None:
            self.data["rope_guard_calls"] += 1
        else:
            self.data["rope_guard_calls"] = self._increment_counter(2)
        layers = self.data["rope_guard_layers"]
        if self._counter_map is None:
            layers[str(layer)] = layers.get(str(layer), 0) + 1
        else:
            layers[str(layer)] = self._increment_counter(
                3 + self.num_layers + layer
            )
        if self.directory and self.flush_mode == "immediate":
            self.flush()


def _merge_mmap_counters(data, path):
    raw = path.read_bytes()
    if len(raw) < _COUNTER_HEADER.size:
        raise RuntimeError(f"truncated BTS audit counter mapping: {path}")
    magic, version, num_layers = _COUNTER_HEADER.unpack_from(raw)
    if magic != _COUNTER_MAGIC or version != _COUNTER_VERSION:
        raise RuntimeError(f"invalid BTS audit counter mapping: {path}")
    expected_size = _COUNTER_HEADER.size + (3 + 2 * num_layers) * _UINT64.size
    if len(raw) != expected_size:
        raise RuntimeError(f"wrong-sized BTS audit counter mapping: {path}")
    values = struct.unpack_from(
        f"<{3 + 2 * num_layers}Q", raw, _COUNTER_HEADER.size
    )
    data["prefill_calls"] = values[0]
    data["native_decode_calls"] = values[1]
    data["rope_guard_calls"] = values[2]
    data["layers"] = {
        str(layer): values[3 + layer]
        for layer in range(num_layers)
        if values[3 + layer]
    }
    data["rope_guard_layers"] = {
        str(layer): values[3 + num_layers + layer]
        for layer in range(num_layers)
        if values[3 + num_layers + layer]
    }
    return data


def read_audits(directory):
    audits = []
    for path in sorted(Path(directory).glob("worker-*.json")):
        data = json.loads(path.read_text())
        counter_path = path.with_suffix(".counts")
        if data.get("counter_transport") == "mmap_v1":
            if not counter_path.is_file():
                raise RuntimeError(f"missing BTS audit counter mapping: {counter_path}")
            data = _merge_mmap_counters(data, counter_path)
        audits.append(data)
    return audits
