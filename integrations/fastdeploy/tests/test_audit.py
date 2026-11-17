import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bts_fastdeploy.audit import Audit
from bts_fastdeploy import environment
from bts_fastdeploy.environment import (
    package_provenance,
    validate_fastdeploy_plugin_entry_point,
    validate_package_provenance,
)


def _entry_point(value="bts_fastdeploy.plugin:register"):
    return SimpleNamespace(
        name="bts",
        value=value,
        dist=SimpleNamespace(name="bts", version="0.3.0"),
    )


def test_fastdeploy_plugin_entry_point_is_reported(monkeypatch):
    monkeypatch.setattr(
        environment,
        "entry_points",
        lambda *, group: [_entry_point()] if group == "fastdeploy.model_register_plugins" else [],
    )
    assert validate_fastdeploy_plugin_entry_point() == {
        "group": "fastdeploy.model_register_plugins",
        "name": "bts",
        "value": "bts_fastdeploy.plugin:register",
        "distribution": "bts",
        "distribution_version": "0.3.0",
    }


def test_missing_fastdeploy_plugin_entry_point_fails_before_gpu(monkeypatch):
    monkeypatch.setattr(environment, "entry_points", lambda *, group: [])
    with pytest.raises(RuntimeError, match="PYTHONPATH is not sufficient"):
        validate_fastdeploy_plugin_entry_point()


def test_wrong_fastdeploy_plugin_entry_point_is_rejected(monkeypatch):
    monkeypatch.setattr(
        environment,
        "entry_points",
        lambda *, group: [_entry_point("other.module:register")],
    )
    with pytest.raises(RuntimeError, match="wrong callable"):
        validate_fastdeploy_plugin_entry_point()


def test_package_provenance_accepts_active_source_root(monkeypatch):
    provenance = package_provenance()
    root = Path(provenance["bts_file"]).parents[1]
    monkeypatch.setenv("BTS_SOURCE_ROOT", str(root))
    pinned = package_provenance()
    assert pinned["required_source_root"] == str(root)
    assert Path(pinned["bts_file"]).is_relative_to(root / "bts")
    assert Path(pinned["bts_fastdeploy_file"]).is_relative_to(root)
    assert Path(pinned["bts_fastdeploy_file"]).parent.name == "bts_fastdeploy"


def test_package_provenance_rejects_wrong_source_root(monkeypatch, tmp_path):
    monkeypatch.setenv("BTS_SOURCE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="BTS_SOURCE_ROOT"):
        package_provenance()


def test_audit_records_worker_package_provenance(monkeypatch):
    monkeypatch.delenv("BTS_SOURCE_ROOT", raising=False)
    audit = Audit("digest")
    assert audit.data["method_display_name"] == "BTS"
    assert audit.data["backend_mode"] == "bts"
    assert audit.data["rope_guard_calls"] == 0
    assert audit.data["rope_guard_layers"] == {}
    recorded = validate_package_provenance(audit.data["package_provenance"])
    assert recorded["bts_file"].endswith("/bts/__init__.py")
    assert recorded["bts_fastdeploy_file"].endswith(
        "/bts_fastdeploy/__init__.py"
    )


def test_audit_fails_closed_for_wrong_worker_source_root(monkeypatch, tmp_path):
    monkeypatch.setenv("BTS_SOURCE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="BTS_SOURCE_ROOT"):
        Audit("digest")


def test_atexit_mode_buffers_until_explicit_flush(monkeypatch, tmp_path):
    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_FLUSH", "atexit")
    audit = Audit("digest", tmp_path)
    audit.record(0, (32766,))
    audit.record_rope_guard(0)
    assert not list(tmp_path.glob("worker-*.json"))
    audit.flush()
    saved = json.loads(next(tmp_path.glob("worker-*.json")).read_text())
    assert saved["prefill_calls"] == 1
    assert saved["layers"] == {"0": 1}
    assert saved["rope_guard_calls"] == 1
    assert saved["rope_guard_layers"] == {"0": 1}


def test_mmap_mode_survives_without_final_python_flush(monkeypatch, tmp_path):
    from bts_fastdeploy.audit import read_audits

    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_FLUSH", "mmap")
    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_NUM_LAYERS", "8")
    audit = Audit("digest", tmp_path)
    audit.record(7, (32766,))
    audit.record_rope_guard(7)
    saved = read_audits(tmp_path)[0]
    assert saved["prefill_calls"] == 1
    assert saved["layers"] == {"7": 1}
    assert saved["rope_guard_calls"] == 1
    assert saved["rope_guard_layers"] == {"7": 1}
    assert saved["counter_transport"] == "mmap_v1"


def test_mmap_mode_aggregates_multiple_backend_instances(monkeypatch, tmp_path):
    from bts_fastdeploy.audit import read_audits

    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_FLUSH", "mmap")
    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_NUM_LAYERS", "4")
    first = Audit("digest", tmp_path)
    second = Audit("digest", tmp_path)
    first.record(0, (128,))
    second.record(3, (128,))
    saved = read_audits(tmp_path)[0]
    assert saved["prefill_calls"] == 2
    assert saved["layers"] == {"0": 1, "3": 1}


@pytest.mark.parametrize("flush_mode", ["immediate", "mmap"])
def test_audit_counts_active_sequences_in_packed_prefill(
    monkeypatch, tmp_path, flush_mode
):
    from bts_fastdeploy.audit import read_audits

    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_FLUSH", flush_mode)
    if flush_mode == "mmap":
        monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_NUM_LAYERS", "4")
    audit = Audit("digest", tmp_path)
    audit.record(2, (17, 0, 129, 0))
    saved = read_audits(tmp_path)[0]
    assert saved["prefill_calls"] == 2
    assert saved["layers"] == {"2": 2}
    if flush_mode == "immediate":
        assert saved["last_lengths"] == [17, 0, 129, 0]


def test_audit_accepts_explicit_dense_backend_schema(monkeypatch):
    monkeypatch.delenv("BTS_SOURCE_ROOT", raising=False)
    audit = Audit(
        "not-applicable-dense",
        method_display_name="FastDeploy dense FlashAttention",
        backend_mode="dense_rope_compat",
    )
    audit.record_rope_guard(3)
    assert audit.data["method_display_name"] == "FastDeploy dense FlashAttention"
    assert audit.data["backend_mode"] == "dense_rope_compat"
    assert audit.data["rope_guard_layers"] == {"3": 1}


def test_unknown_audit_flush_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_FLUSH", "sometimes")
    with pytest.raises(ValueError, match="AUDIT_FLUSH"):
        Audit("digest")
