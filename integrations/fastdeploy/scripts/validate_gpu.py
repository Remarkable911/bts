"""Run bounded GPU acceptance, optionally waiting for an entirely idle GPU.

Uses nvidia-smi only while waiting: no CUDA context, memory reservation or model
load until a device has no compute processes, <1 GiB used and <=5% utilization
continuously for --idle-seconds. Existing processes are never stopped.
"""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
GPU_LOCK_DIR = Path("/tmp")
PROCESS_GROUP_TERM_TIMEOUT = 10
PROCESS_GROUP_KILL_TIMEOUT = 2


class ValidationInterrupted(Exception):
    """Raised by the main-process signal handler after recording the signal."""

    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"validation interrupted by {signal.Signals(signum).name}")


def interruption_handler(signum, _frame):
    """Turn SIGINT/SIGTERM into an exception so run_step can clean its PGID."""
    raise ValidationInterrupted(signum)


def gpu_memory_utilization_arg(value):
    """Parse a bounded FastDeploy GPU-memory fraction."""
    try:
        fraction = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "gpu-memory-utilization must be a number in (0, 1]"
        ) from error
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError(
            "gpu-memory-utilization must be in (0, 1]"
        )
    return fraction


def gpu_index_arg(value):
    """Parse a physical, non-negative nvidia-smi GPU index."""
    try:
        index = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "gpu-index must be a non-negative integer"
        ) from error
    if index < 0:
        raise argparse.ArgumentTypeError("gpu-index must be a non-negative integer")
    return index


def read_model_contract(model):
    """Read and validate the model metadata before waiting for a GPU."""
    config_path = model / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"model must contain {config_path.name}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"model config is not valid JSON: {config_path}") from error
    if not isinstance(config, dict):
        raise ValueError("model config.json must contain a JSON object")
    if config.get("model_type") != "qwen3":
        raise ValueError("GPU acceptance is limited to model_type=qwen3")

    quantized = [
        name
        for name in ("quantization_config", "quantization", "quantization_method")
        if config.get(name) not in (None, False, "", {})
    ]
    if quantized:
        raise ValueError(
            "GPU acceptance requires an unquantized model; found "
            + ", ".join(quantized)
        )

    head_dim = config.get("head_dim")
    if head_dim is None:
        hidden_size = config.get("hidden_size")
        num_heads = config.get("num_attention_heads")
        if (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size <= 0
            or not isinstance(num_heads, int)
            or isinstance(num_heads, bool)
            or num_heads <= 0
        ):
            raise ValueError(
                "cannot derive head_dim: require positive integer hidden_size and "
                "num_attention_heads"
            )
        if hidden_size % num_heads:
            raise ValueError(
                "cannot derive head_dim: hidden_size is not divisible by "
                "num_attention_heads"
            )
        head_dim = hidden_size // num_heads
    elif not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    if head_dim != 128:
        raise ValueError(
            f"head_dim={head_dim} is incompatible with frozen BTS; require head_dim=128"
        )
    return config


def subprocess_library_path(search_paths=None, inherited=None):
    """Find packaged Torch/CUDA libraries without importing CUDA modules."""
    search_paths = sys.path if search_paths is None else search_paths
    inherited = os.environ.get("LD_LIBRARY_PATH", "") if inherited is None else inherited
    candidates = []
    for entry in search_paths:
        base = Path(entry or os.getcwd()).resolve()
        candidates.append(base / "torch/lib")
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            candidates.extend(sorted(nvidia.glob("*/lib")))
    result = []
    seen = set()
    for candidate in candidates:
        value = str(candidate)
        if candidate.is_dir() and value not in seen:
            result.append(value)
            seen.add(value)
    for value in inherited.split(os.pathsep):
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return os.pathsep.join(result)


def subprocess_pythonpath(bts_source, inherited=None):
    """Prefer a complete wheel target; add the source adapter only if absent."""
    inherited = os.environ.get("PYTHONPATH", "") if inherited is None else inherited
    bts_source = bts_source.resolve()
    values = [str(bts_source)]
    if not (bts_source / "bts_fastdeploy/__init__.py").is_file():
        values.append(str(ROOT / "integrations/fastdeploy/src"))
    values.extend(value for value in inherited.split(os.pathsep) if value)
    return os.pathsep.join(dict.fromkeys(values))


def process_group_exists(pgid):
    """Return whether a process group still exists without signalling its members."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group_exit(pgid, timeout):
    """Boundedly wait for every process in a dedicated process group to leave."""
    deadline = time.monotonic() + timeout
    while process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not process_group_exists(pgid)


def terminate_process_group(process):
    """Best-effort TERM/KILL of the dedicated group created for one step."""
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not wait_for_process_group_exit(pgid, PROCESS_GROUP_TERM_TIMEOUT):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wait_for_process_group_exit(pgid, PROCESS_GROUP_KILL_TIMEOUT)
    try:
        process.wait(timeout=PROCESS_GROUP_KILL_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=PROCESS_GROUP_KILL_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass


def run_step(command, *, cwd, env, log, timeout):
    """Clean up the command's dedicated process group on failure or interruption."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        terminate_process_group(process)
        raise


def available_gpus(allowed_indices=None):
    """Return idle physical GPUs, optionally restricted to nvidia-smi indices."""
    allowed_indices = None if allowed_indices is None else set(allowed_indices)
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                         "--format=csv,noheader"], text=True)
    occupied = set(processes.splitlines())
    result = []
    for line in raw.splitlines():
        index, gpu_uuid, memory, utilization = [s.strip() for s in line.split(",")]
        if allowed_indices is not None and int(index) not in allowed_indices:
            continue
        if gpu_uuid not in occupied and int(memory) < 1024 and int(utilization) <= 5:
            result.append((index, gpu_uuid))
    return result


def release_locks(handles):
    """Release and close advisory lock handles in reverse acquisition order."""
    for handle in reversed(handles):
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def try_acquire_gpu_locks(index, gpu_uuid, cooperative_lease_dir=None):
    """Non-blockingly acquire this validator's lock and an optional shared lease."""
    paths = [GPU_LOCK_DIR / f"bts-fastdeploy-{gpu_uuid}.lock"]
    if cooperative_lease_dir is not None:
        paths.append(cooperative_lease_dir / f"gpu{index}.lock")
    handles = []
    try:
        for path in paths:
            handle = path.open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            handles.append(handle)
    except BlockingIOError:
        release_locks(handles)
        return None
    except BaseException:
        release_locks(handles)
        raise
    return handles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bts-source", type=Path, default=ROOT,
                        help="Directory containing frozen bts/ source or a wheel --target install")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=120)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--timeout-hours", type=float, default=12)
    parser.add_argument(
        "--gpu-index",
        action="append",
        type=gpu_index_arg,
        help=(
            "Restrict acceptance to this physical nvidia-smi index; repeat to "
            "allow more than one GPU"
        ),
    )
    parser.add_argument(
        "--cooperative-lease-dir",
        type=Path,
        help=(
            "Optional existing directory containing gpu{index}.lock files shared "
            "with other local GPU queues"
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=gpu_memory_utilization_arg,
        default=0.20,
        help="FastDeploy GPU-memory fraction in (0, 1] (default: 0.20)",
    )
    args = parser.parse_args()
    if not (args.bts_source / "bts/config.py").is_file():
        parser.error("bts-source must contain bts/config.py")
    args.model = args.model.expanduser().resolve()
    allowed_gpu_indices = (
        None if args.gpu_index is None else set(args.gpu_index)
    )
    if args.cooperative_lease_dir is not None:
        args.cooperative_lease_dir = (
            args.cooperative_lease_dir.expanduser().resolve()
        )
        if not args.cooperative_lease_dir.is_dir():
            parser.error("cooperative-lease-dir must be an existing directory")
    try:
        model_info = read_model_contract(args.model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    def status(state, **values):
        data = dict(status=state, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    pid=os.getpid(), **values)
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data,indent=2)+"\n")
        tmp.replace(status_path)
        print(json.dumps(data),flush=True)

    lock = (args.output / "validation.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    handled_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        signum: signal.signal(signum, interruption_handler)
        for signum in handled_signals
    }
    candidate_locks = []
    chosen = None
    try:
        deadline = time.monotonic() + args.timeout_hours*3600
        idle_since = {}
        status(
            "waiting_for_idle_gpu",
            model=str(args.model),
            head_dim=model_info.get("head_dim")
            or model_info["hidden_size"] // model_info["num_attention_heads"],
            gpu_memory_utilization=args.gpu_memory_utilization,
            gpu_indices=(
                sorted(allowed_gpu_indices)
                if allowed_gpu_indices is not None
                else None
            ),
            cooperative_lease_dir=(
                str(args.cooperative_lease_dir)
                if args.cooperative_lease_dir is not None
                else None
            ),
        )
        while True:
            idle = available_gpus(allowed_gpu_indices)
            now = time.monotonic()
            ids = {gpu_uuid for _,gpu_uuid in idle}
            idle_since = {k:v for k,v in idle_since.items() if k in ids}
            for _,gpu_uuid in idle:
                idle_since.setdefault(gpu_uuid,now)
            gpu_became_busy = False
            stable = [
                candidate
                for candidate in idle
                if now-idle_since[candidate[1]] >= args.idle_seconds
            ]
            for candidate in stable:
                index, gpu_uuid = candidate
                acquired = try_acquire_gpu_locks(
                    index, gpu_uuid, args.cooperative_lease_dir
                )
                if acquired is None:
                    continue
                try:
                    still_idle = candidate in available_gpus(allowed_gpu_indices)
                except BaseException:
                    release_locks(acquired)
                    raise
                if not still_idle:
                    release_locks(acquired)
                    idle_since.pop(gpu_uuid, None)
                    gpu_became_busy = True
                    continue
                chosen = candidate
                candidate_locks = acquired
                break
            if chosen is not None:
                break
            if not args.wait or now >= deadline:
                status(
                    "deferred_gpu_became_busy"
                    if gpu_became_busy
                    else "deferred_no_idle_gpu"
                )
                return 2
            time.sleep(args.poll_seconds)

        index,gpu_uuid = chosen
        cooperative_lease = (
            str(args.cooperative_lease_dir / f"gpu{index}.lock")
            if args.cooperative_lease_dir is not None
            else None
        )
        # FastDeploy 2.5 casts device_ids to int for NVML; CUDA UUID masks are not
        # accepted by that release. Retain the UUID for locking and evidence.
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=index,CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   BTS_FASTDEPLOY_TEST_CUDA="1",OMP_NUM_THREADS="1")
        env["TORCH_EXTENSIONS_DIR"] = str((args.output/"cache/torch_extensions").resolve())
        env["TRITON_CACHE_DIR"] = str((args.output/"cache/triton").resolve())
        # Paddle imports CUDA libraries before either bridge test or generation can
        # run. Locate the libraries shipped beside this Python interpreter without
        # importing Torch/Paddle or creating a CUDA context during the idle wait.
        env["LD_LIBRARY_PATH"] = subprocess_library_path()
        env["BTS_SOURCE_ROOT"] = str(args.bts_source.resolve())
        env["PYTHONPATH"] = subprocess_pythonpath(
            args.bts_source, env.get("PYTHONPATH", "")
        )
        prompt = args.output / "prompt.txt"
        prompt.write_text("The archive describes how sparse attention selects useful context tokens. " * 60
                          + "\nSummarize the archive in one sentence.\n")
        commands = [
            [sys.executable,"-m","pytest","-q",str(ROOT/"integrations/fastdeploy/tests/test_bridge.py"),
             f"--junitxml={args.output.resolve() / 'gpu_bridge.xml'}"],
        ]
        for name,config in (("default",None),("topk040_tail256",ROOT/"integrations/fastdeploy/configs/topk040_tail256.json")):
            command = [sys.executable,"-m","bts_fastdeploy.run","--model",str(args.model.resolve()),
                       "--prompt-file",str(prompt.resolve()),"--max-model-len","2048","--max-new-tokens","8",
                       "--gpu-memory-utilization",str(args.gpu_memory_utilization),
                       "--output",str((args.output/name).resolve())]
            if config:
                command += ["--config",str(config)]
            commands.append(command)
        for step,command in enumerate(commands):
            status(
                "running",
                gpu=index,
                gpu_uuid=gpu_uuid,
                cooperative_lease=cooperative_lease,
                step=step,
                command=command,
            )
            with (args.output/f"step-{step}.log").open("w") as log:
                # Keep the checkout itself off sys.path[0], so a --target wheel
                # install in PYTHONPATH wins over same-named worktree packages.
                run_step(command,cwd=args.output.resolve(),env=env,log=log,timeout=1800)
        status(
            "passed",
            gpu=index,
            gpu_uuid=gpu_uuid,
            cooperative_lease=cooperative_lease,
        )
        return 0
    except ValidationInterrupted as error:
        values = {
            "signal": error.signum,
            "signal_name": signal.Signals(error.signum).name,
            "error": str(error),
        }
        if chosen is not None:
            values.update(gpu=chosen[0], gpu_uuid=chosen[1])
        status("interrupted", **values)
        return 128 + error.signum
    except (subprocess.SubprocessError,OSError) as error:
        values = {"error": str(error)}
        if chosen is not None:
            values.update(gpu=chosen[0], gpu_uuid=chosen[1])
        status("failed", **values)
        return 1
    finally:
        release_locks(candidate_locks)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
