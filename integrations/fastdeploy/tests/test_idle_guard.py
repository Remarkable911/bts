import importlib.util
from pathlib import Path

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import pytest


def load_validator():
    path=Path(__file__).resolve().parents[1]/"scripts/validate_gpu.py"
    spec=importlib.util.spec_from_file_location("idle_validator",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_idle_guard_excludes_resident_processes_even_at_zero_utilization(monkeypatch):
    module=load_validator()
    def output(command,**kwargs):
        if "--query-compute-apps=gpu_uuid" in command:
            return "GPU-resident\n"
        return "0, GPU-resident, 200, 0\n1, GPU-free, 20, 0\n2, GPU-busy, 20, 80\n3, GPU-memory, 10000, 0\n"
    monkeypatch.setattr(module.subprocess,"check_output",output)
    assert module.available_gpus() == [("1","GPU-free")]
    assert module.available_gpus({1, 3}) == [("1", "GPU-free")]
    assert module.available_gpus({0, 2}) == []


@pytest.mark.parametrize("value", ["-1", "1.5", "gpu1"])
def test_gpu_index_rejects_invalid_values(value):
    module = load_validator()
    with pytest.raises(argparse.ArgumentTypeError, match="gpu-index"):
        module.gpu_index_arg(value)


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("1", 1), ("17", 17)])
def test_gpu_index_accepts_non_negative_integers(value, expected):
    module = load_validator()
    assert module.gpu_index_arg(value) == expected


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "nan", "inf", "not-a-number"])
def test_gpu_memory_utilization_rejects_out_of_range_values(value):
    module=load_validator()
    with pytest.raises(argparse.ArgumentTypeError, match="gpu-memory-utilization"):
        module.gpu_memory_utilization_arg(value)


@pytest.mark.parametrize("value", ["0.01", "0.20", "0.35", "1"])
def test_gpu_memory_utilization_accepts_bounded_values(value):
    module=load_validator()
    assert module.gpu_memory_utilization_arg(value) == float(value)


def test_subprocess_library_path_finds_packaged_cuda_libraries_without_imports(tmp_path):
    module=load_validator()
    site = tmp_path / "site-packages"
    torch_lib = site / "torch/lib"
    cudnn_lib = site / "nvidia/cudnn/lib"
    cublas_lib = site / "nvidia/cublas/lib"
    inherited = tmp_path / "system-cuda/lib"
    for directory in (torch_lib, cudnn_lib, cublas_lib, inherited):
        directory.mkdir(parents=True)
    value = module.subprocess_library_path(
        search_paths=[str(site)], inherited=str(inherited)
    )
    assert value.split(module.os.pathsep) == [
        str(torch_lib),
        str(cublas_lib),
        str(cudnn_lib),
        str(inherited),
    ]


def test_subprocess_library_path_deduplicates_and_preserves_inherited_paths(tmp_path):
    module=load_validator()
    torch_lib = tmp_path / "torch/lib"
    torch_lib.mkdir(parents=True)
    value = module.subprocess_library_path(
        search_paths=[str(tmp_path), str(tmp_path)],
        inherited=module.os.pathsep.join([str(torch_lib), str(tmp_path / "missing")]),
    )
    assert value.split(module.os.pathsep) == [str(torch_lib), str(tmp_path / "missing")]


def test_subprocess_pythonpath_uses_complete_wheel_target_without_source_fallback(tmp_path):
    module=load_validator()
    target = tmp_path / "wheel-target"
    adapter = target / "bts_fastdeploy"
    adapter.mkdir(parents=True)
    (adapter / "__init__.py").write_text("")
    value = module.subprocess_pythonpath(target, "/existing/pythonpath")
    assert value.split(module.os.pathsep) == [
        str(target.resolve()),
        "/existing/pythonpath",
    ]


def test_subprocess_pythonpath_adds_adapter_fallback_for_core_source(tmp_path):
    module = load_validator()
    source = tmp_path / "core-source"
    (source / "bts").mkdir(parents=True)
    value = module.subprocess_pythonpath(source, "/existing/pythonpath")
    assert value.split(module.os.pathsep) == [
        str(source.resolve()),
        str(module.ROOT / "integrations/fastdeploy/src"),
        "/existing/pythonpath",
    ]


def test_bridge_conftest_imports_both_packages_from_wheel_target(tmp_path):
    target = tmp_path / "wheel-target"
    for package in ("bts", "bts_fastdeploy"):
        directory = target / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("WHEEL_TARGET = True\n")
    conftest = Path(__file__).with_name("conftest.py")
    code = (
        "import json, runpy; "
        f"runpy.run_path({str(conftest)!r}); "
        "import bts, bts_fastdeploy; "
        "print(json.dumps([bts.__file__, bts_fastdeploy.__file__]))"
    )
    env = dict(os.environ, BTS_SOURCE_ROOT=str(target))
    files = json.loads(
        subprocess.check_output([sys.executable, "-I", "-c", code], env=env, text=True)
    )
    assert [Path(value).resolve().parent for value in files] == [
        (target / "bts").resolve(),
        (target / "bts_fastdeploy").resolve(),
    ]


def wait_for_path(path, timeout=5):
    deadline = time.monotonic() + timeout
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.is_file(), f"timed out waiting for {path}"


def assert_pid_gone(pid, timeout=5):
    process_path = Path(f"/proc/{pid}")
    deadline = time.monotonic() + timeout
    while process_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not process_path.exists(), f"process {pid} survived cleanup"


def test_cooperative_gpu_lease_is_nonblocking_and_held_until_release(
    monkeypatch, tmp_path
):
    module = load_validator()
    internal = tmp_path / "internal-locks"
    leases = tmp_path / "leases"
    internal.mkdir()
    leases.mkdir()
    monkeypatch.setattr(module, "GPU_LOCK_DIR", internal)
    lease_path = leases / "gpu1.lock"
    held = lease_path.open("a")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert module.try_acquire_gpu_locks("1", "GPU-test", leases) is None
        # A failed cooperative lock attempt must release the internal UUID lock.
        with (internal / "bts-fastdeploy-GPU-test.lock").open("a") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(probe, fcntl.LOCK_UN)
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()

    acquired = module.try_acquire_gpu_locks("1", "GPU-test", leases)
    assert acquired is not None
    try:
        with lease_path.open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        module.release_locks(acquired)
    with lease_path.open("a") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender, fcntl.LOCK_UN)


def test_run_step_cleans_child_group_after_nonzero_parent_exit(
    monkeypatch, tmp_path
):
    module = load_validator()
    monkeypatch.setattr(module, "PROCESS_GROUP_TERM_TIMEOUT", 0.5)
    monkeypatch.setattr(module, "PROCESS_GROUP_KILL_TIMEOUT", 0.5)
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_code = f"""
import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, "-c", {child_code!r}])
deadline = time.monotonic() + 5
while not Path({str(child_pid)!r}).is_file() and time.monotonic() < deadline:
    time.sleep(0.01)
sys.exit(7)
"""
    with (tmp_path / "step.log").open("w") as log:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            module.run_step(
                [sys.executable, "-c", parent_code],
                cwd=tmp_path,
                env=os.environ.copy(),
                log=log,
                timeout=5,
            )
    assert caught.value.returncode == 7
    wait_for_path(child_pid)
    assert_pid_gone(int(child_pid.read_text()))


def test_run_step_cleans_process_group_after_timeout(monkeypatch, tmp_path):
    module = load_validator()
    monkeypatch.setattr(module, "PROCESS_GROUP_TERM_TIMEOUT", 0.5)
    monkeypatch.setattr(module, "PROCESS_GROUP_KILL_TIMEOUT", 0.5)
    child_pid = tmp_path / "timeout.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os,time; from pathlib import Path; "
            f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
        ),
    ]
    with (tmp_path / "timeout.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            module.run_step(
                command,
                cwd=tmp_path,
                env=os.environ.copy(),
                log=log,
                timeout=0.2,
            )
    wait_for_path(child_pid)
    assert_pid_gone(int(child_pid.read_text()))


def test_sigterm_handler_cleans_active_step_process_group(tmp_path):
    validator_path = (
        Path(__file__).resolve().parents[1] / "scripts/validate_gpu.py"
    )
    worker_pid = tmp_path / "signal-worker.pid"
    worker_code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(worker_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    helper_code = f"""
import importlib.util, os, signal, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("signal_validator", {str(validator_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.PROCESS_GROUP_TERM_TIMEOUT = 0.5
module.PROCESS_GROUP_KILL_TIMEOUT = 0.5
signal.signal(signal.SIGTERM, module.interruption_handler)
with Path({str(tmp_path / 'signal.log')!r}).open("w") as log:
    try:
        module.run_step(
            [sys.executable, "-c", {worker_code!r}], cwd={str(tmp_path)!r},
            env=os.environ.copy(), log=log, timeout=30
        )
    except module.ValidationInterrupted:
        raise SystemExit(143)
raise SystemExit(99)
"""
    helper = subprocess.Popen(
        [sys.executable, "-c", helper_code], start_new_session=True
    )
    pid = None
    try:
        wait_for_path(worker_pid)
        pid = int(worker_pid.read_text())
        os.kill(helper.pid, signal.SIGTERM)
        assert helper.wait(timeout=5) == 143
        assert_pid_gone(pid)
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=5)
        if pid is not None and Path(f"/proc/{pid}").exists():
            os.kill(pid, signal.SIGKILL)


def main_test_paths(tmp_path):
    source = tmp_path / "source"
    (source / "bts").mkdir(parents=True)
    (source / "bts/config.py").write_text("")
    model = tmp_path / "model"
    model.mkdir()
    write_model(model, head_dim=128)
    return source, model, tmp_path / "output"


def test_main_forwards_repeated_gpu_index_filter(monkeypatch, tmp_path):
    module = load_validator()
    source, model, output = main_test_paths(tmp_path)
    observed = []

    def no_idle_gpu(allowed_indices):
        observed.append(allowed_indices)
        return []

    monkeypatch.setattr(module, "available_gpus", no_idle_gpu)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_gpu.py",
            "--model", str(model),
            "--output", str(output),
            "--bts-source", str(source),
            "--gpu-index", "1",
            "--gpu-index", "3",
        ],
    )
    assert module.main() == 2
    assert observed == [{1, 3}]


def test_main_records_interrupted_status(monkeypatch, tmp_path):
    module = load_validator()
    source, model, output = main_test_paths(tmp_path)

    def interrupt(_allowed_indices):
        raise module.ValidationInterrupted(signal.SIGTERM)

    monkeypatch.setattr(module, "available_gpus", interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_gpu.py",
            "--model", str(model),
            "--output", str(output),
            "--bts-source", str(source),
            "--wait",
        ],
    )
    assert module.main() == 143
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["signal"] == signal.SIGTERM
    assert status["signal_name"] == "SIGTERM"


def write_model(tmp_path, **updates):
    config = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_attention_heads": 32,
    }
    config.update(updates)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_model_contract_accepts_unquantized_qwen3_head_dim_128(tmp_path):
    module=load_validator()
    config = module.read_model_contract(write_model(tmp_path))
    assert config["hidden_size"] // config["num_attention_heads"] == 128


def test_model_contract_prefers_explicit_head_dim_like_qwen3_06b(tmp_path):
    module=load_validator()
    config = module.read_model_contract(
        write_model(tmp_path, head_dim=128, hidden_size=1024, num_attention_heads=16)
    )
    assert config["head_dim"] == 128


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "llama"}, "model_type=qwen3"),
        ({"head_dim": 64}, "head_dim=64"),
        ({"hidden_size": 1024, "num_attention_heads": 16}, "head_dim=64"),
        ({"hidden_size": None}, "cannot derive head_dim"),
        ({"hidden_size": 4097}, "not divisible"),
        ({"quantization_config": {"quant_method": "awq"}}, "unquantized"),
    ],
)
def test_model_contract_rejects_incompatible_models(tmp_path, updates, message):
    module=load_validator()
    with pytest.raises(ValueError, match=message):
        module.read_model_contract(write_model(tmp_path, **updates))
