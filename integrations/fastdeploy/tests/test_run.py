import ast
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import textwrap

import pytest

from bts_fastdeploy import plugin
from bts_fastdeploy import run
from bts_fastdeploy.environment import package_provenance


def test_launcher_reports_forced_chunked_prefill_guard(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FD_DISABLE_CHUNKED_PREFILL", "0")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

    assert run.main([
        "--model", str(model),
        "--output", str(tmp_path / "output"),
        "--dry-run",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["method_display_name"] == "BTS"
    assert report["engine_options"]["enable_chunked_prefill"] is False
    assert report["fastdeploy_environment"]["FD_DISABLE_CHUNKED_PREFILL"] == "1"
    assert report["launcher_package_provenance"]["bts_file"].endswith(
        "/bts/__init__.py"
    )
    assert report["launcher_package_provenance"]["bts_fastdeploy_file"].endswith(
        "/bts_fastdeploy/__init__.py"
    )
    # Dry-run reports the launch contract without changing the caller process.
    assert os.environ["FD_DISABLE_CHUNKED_PREFILL"] == "0"


def test_launcher_exposes_sixteen_sequence_engine_configuration(
    monkeypatch, tmp_path, capsys
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

    assert run.main(
        [
            "--model",
            str(model),
            "--output",
            str(tmp_path / "output"),
            "--max-num-seqs",
            "16",
            "--num-prompts",
            "16",
            "--dry-run",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["engine_options"]["max_num_seqs"] == 16
    assert report["engine_options"]["max_num_batched_tokens"] == 2048


def test_launcher_applies_guard_before_fastdeploy_import():
    tree = ast.parse(textwrap.dedent(inspect.getsource(run.main)))
    apply_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_apply_fastdeploy_environment"
    ]
    import_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "fastdeploy"
    ]
    assert len(apply_lines) == len(import_lines) == 1
    assert apply_lines[0] < import_lines[0]


def test_launcher_overrides_contradictory_parent_environment(monkeypatch):
    monkeypatch.setenv("FD_DISABLE_CHUNKED_PREFILL", "0")
    values = run._required_fastdeploy_environment()
    run._apply_fastdeploy_environment(values)
    assert os.environ["FD_DISABLE_CHUNKED_PREFILL"] == "1"


def test_generation_layer_delta_requires_each_exact_model_layer():
    assert run._generation_layer_delta(
        {0: 4, 1: 4, 2: 4}, {0: 5, 1: 6, 2: 5}, 3
    ) == {"0": 1, "1": 2, "2": 1}


@pytest.mark.parametrize(
    ("after", "match"),
    [
        ({0: 2, 1: 1}, "missing=\\[2\\]"),
        ({0: 1, 1: 1, 2: 1, 3: 1}, "unexpected=\\[3\\]"),
        ({0: 0, 1: 1, 2: 1}, "decreased"),
    ],
)
def test_generation_layer_delta_rejects_incomplete_or_invalid_coverage(after, match):
    with pytest.raises(RuntimeError, match=match):
        run._generation_layer_delta({0: 1}, after, 3)


def test_prefill_layer_calls_validates_worker_total():
    audits = [{"prefill_calls": 3, "layers": {"0": 1, "1": 2}}]
    assert run._prefill_layer_calls(audits) == {0: 1, 1: 2}
    audits[0]["prefill_calls"] = 4
    with pytest.raises(RuntimeError, match="prefill total"):
        run._prefill_layer_calls(audits)


def test_worker_provenance_must_match_launcher():
    launcher = package_provenance()
    run._validate_worker_provenance(
        [{"package_provenance": launcher}], launcher
    )
    mismatched = dict(launcher)
    mismatched["bts_fastdeploy_file"] = str(
        Path(launcher["bts_fastdeploy_file"]).with_name("run.py")
    )
    with pytest.raises(RuntimeError, match="differs from launcher"):
        run._validate_worker_provenance(
            [{"package_provenance": mismatched}], launcher
        )


def test_worker_provenance_requires_an_audit():
    with pytest.raises(RuntimeError, match="no BTS worker audit"):
        run._validate_worker_provenance([], package_provenance())


def test_serialize_generations_accepts_fastdeploy_25_single_completion():
    completion = SimpleNamespace(text="answer", token_ids=(10, 20))
    outputs = [SimpleNamespace(outputs=completion)]

    assert run._serialize_generations(outputs) == [
        {"text": "answer", "token_ids": [10, 20]}
    ]


def test_serialize_generations_accepts_sequence_and_missing_completion():
    first = SimpleNamespace(text="first", token_ids=[1])
    second = SimpleNamespace(text="second", token_ids=None)
    outputs = [
        SimpleNamespace(outputs=[first, second]),
        SimpleNamespace(outputs=None),
    ]

    assert run._serialize_generations(outputs) == [
        {"text": "first", "token_ids": [1]},
        {"text": "second", "token_ids": []},
    ]


def test_disabled_plugin_does_not_change_chunked_prefill_environment(monkeypatch):
    monkeypatch.delenv("BTS_FASTDEPLOY_ENABLE", raising=False)
    monkeypatch.setenv("FD_DISABLE_CHUNKED_PREFILL", "0")
    plugin.register()
    assert os.environ["FD_DISABLE_CHUNKED_PREFILL"] == "0"


def test_enabled_plugin_sets_missing_chunked_prefill_guard(monkeypatch):
    monkeypatch.setenv("BTS_FASTDEPLOY_ENABLE", "1")
    monkeypatch.delenv("FD_DISABLE_CHUNKED_PREFILL", raising=False)
    monkeypatch.setenv("FD_ATTENTION_BACKEND", "not-flash")
    monkeypatch.setattr(plugin, "version", lambda name: "2.5.0")
    with pytest.raises(ValueError, match="FD_ATTENTION_BACKEND"):
        plugin.register()
    assert os.environ["FD_DISABLE_CHUNKED_PREFILL"] == "1"


def test_enabled_plugin_rejects_contradictory_chunked_prefill_guard(monkeypatch):
    monkeypatch.setenv("BTS_FASTDEPLOY_ENABLE", "1")
    monkeypatch.setenv("FD_DISABLE_CHUNKED_PREFILL", "0")
    with pytest.raises(ValueError, match="FD_DISABLE_CHUNKED_PREFILL=1"):
        plugin.register()


def test_plugin_forces_both_fastdeploy_and_paddle_selectors_to_fa2(monkeypatch):
    flags = {"FLAGS_flash_attn_version": 3}

    class FakePaddle:
        @staticmethod
        def set_flags(values):
            flags.update(values)

        base = SimpleNamespace(
            framework=SimpleNamespace(
                get_flags=lambda names: {name: flags[name] for name in names}
            )
        )

    module = SimpleNamespace(FLASH_ATTN_VERSION=3)
    monkeypatch.setitem(sys.modules, "paddle", FakePaddle)

    contract = plugin._force_flash_attention_2(module)

    assert module.FLASH_ATTN_VERSION == 2
    assert flags["FLAGS_flash_attn_version"] == 2
    assert contract == {
        "fastdeploy_flash_attn_version": 2,
        "paddle_flash_attn_version": 2,
        "dense_prefill_op": "flash_attn_unpadded",
        "decode_op": "append_attention",
    }
