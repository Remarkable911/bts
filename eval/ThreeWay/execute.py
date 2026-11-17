#!/usr/bin/env python3
"""Execute the frozen four-arm 128K benchmark in isolated engine environments.

The public mode is a controller.  It creates one fresh subprocess for every
``(restart, arm)`` pair, monitors single-GPU exclusivity, normalizes the worker
evidence into the schema in :mod:`eval.ThreeWay.run`, and invokes that module's
analysis without weakening its contract.  The hidden worker mode is entered by
the controller with either the FastDeploy or vLLM Python executable.

``--dry-run`` is intentionally static: it neither imports an inference engine
nor touches NVIDIA state and it never emits measured arm reports.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
SUBMISSION_ROOT = HERE.parents[1]
FASTDEPLOY_SRC = SUBMISSION_ROOT / "integrations" / "fastdeploy" / "src"
VLLM_SRC = SUBMISSION_ROOT / "integrations" / "vllm" / "src"
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from eval.ThreeWay import run as analysis  # noqa: E402
from eval.ThreeWay import terminal_summary  # noqa: E402


EXECUTION_SCHEMA_VERSION = "bts_four_arm_execution_v5"
WORKER_SPEC_SCHEMA_VERSION = "bts_four_arm_worker_spec_v1"
OBSERVATION_SCHEMA_VERSION = "bts_four_arm_worker_observation_v1"
GPU_LEASE_SCHEMA_VERSION = "bts_three_way_gpu_lease_v1"
EXECUTION_PROTOCOL = "bts_four_arm_128k_executor_v5"
ARM_ORDER = analysis.ARMS
RUN_TOKEN_ENV = "BTS_THREE_WAY_RUN_TOKEN"
FORMAL_WARMUPS = 1
FORMAL_SEED = analysis.FORMAL_BOOTSTRAP_SEED
FORMAL_REQUEST_COUNT = 16
FORMAL_MAX_NUM_SEQS = 1
FORMAL_GPU_MEMORY_UTILIZATION = 0.90
FORMAL_PYTORCH_ALLOC_CONF = "expandable_segments:True"
GPU_WORKER_DISAPPEAR_GRACE_SECONDS = 60.0
SUPPORTED_FASTDEPLOY_VERSION = "2.5.0"
SUPPORTED_VLLM_VERSION = "0.17.0"
SUPPORTED_FLASH_ATTN_VERSION = "2.8.2"
VLLM_KV_CACHE_BLOCK_SIZE = 256
VLLM_FA2_ADAPTER_NAME = "bts_vllm_upstream_flash_attn_2_cuda"
VLLM_FA2_LOG_MARKER_PREFIX = "BTS_VLLM_FA2_ADAPTER_CALL "
VLLM_FA2_MARKER_SCHEMA_VERSION = "bts_vllm_fa2_adapter_call_v2"
DEFAULT_VOCAB_SIZE = 151936
BTS_VARIABLE_TOPK_ENV = "BTS_DETERMINISTIC_VARIABLE_TOPK"
BTS_VARIABLE_TOPK_REQUIRE_ENV = "BTS_REQUIRE_DETERMINISTIC_VARIABLE_TOPK"
BTS_VARIABLE_TOPK_EMIT_CONTRACT = "deterministic_warp_segment_prefix_v1"
BTS_VARIABLE_TOPK_REQUIRED_ERROR_PREFIX = (
    "BTS required deterministic variable Top-K: "
)
BTS_VARIABLE_TOPK_FAILURE_MARKER_ENV = (
    "BTS_REQUIRED_VARIABLE_TOPK_FAILURE_MARKER"
)
BTS_VARIABLE_TOPK_FAILURE_MARKER_SCHEMA = (
    "bts_required_variable_topk_failure_v1"
)
BTS_VARIABLE_TOPK_FAILURE_MARKER_FILENAME = (
    "required-variable-topk-failure.json"
)
EXPECTED_MODEL_SHAPE = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FA_VERSION_PATTERN = re.compile(r"Using FlashAttention version\s+(\d+)")


class ExecutionError(RuntimeError):
    """Raised when runtime evidence cannot support a formal arm report."""


class ExternalGpuProcessDetected(ExecutionError):
    """A recoverable arm-attempt failure caused by an unrelated CUDA process."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any]):
        super().__init__(message)
        self.evidence = dict(evidence)


class WorkerAttemptFailed(ExecutionError):
    """A worker failure with cleanup evidence and private retry handles."""

    def __init__(
        self,
        message: str,
        *,
        evidence: Mapping[str, Any],
        process: subprocess.Popen[Any],
        token: str,
    ):
        super().__init__(message)
        self.evidence = dict(evidence)
        # These are deliberately not serialized: the raw token is only used
        # to finish cleanup before the controller may release its GPU lease.
        self.process = process
        self.token = token


class NonRetryableWorkerExit(ExecutionError):
    """A worker exit code reserved for deterministic CLI/protocol failures."""


class NonRetryableWorkerAttemptFailed(WorkerAttemptFailed):
    """A cleaned worker failure which must not consume a retry budget."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repository_test_lock_path() -> Path:
    """Return the host-local lock shared by formal runs and this repo's tests."""

    root_hash = hashlib.sha256(str(SUBMISSION_ROOT).encode("utf-8")).hexdigest()[:16]
    return Path("/tmp") / f"bts-three-way-{root_hash}.lock"


def formal_run_lock_paths(gpu: Mapping[str, Any]) -> tuple[Path, ...]:
    """Cover this repo plus every cooperative per-GPU lock used by the project."""

    uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(gpu["uuid"]))
    lock_dir = Path("/tmp")
    return (
        repository_test_lock_path(),
        lock_dir / f"bts-vllm-queue-{uuid}.lock",
        lock_dir / f"prism-fastdeploy-{uuid}.lock",
        lock_dir / f"bts-fastdeploy-{uuid}.lock",
    )


def acquire_formal_run_lock(
    *, gpu: Mapping[str, Any], wait_seconds: float, poll_seconds: float
) -> list[Any]:
    """Exclude repository tests and other formal controllers before GPU wait."""

    paths = formal_run_lock_paths(gpu)
    deadline = time.monotonic() + wait_seconds
    announced = False
    while True:
        streams = []
        try:
            for path in paths:
                stream = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    stream.close()
                    raise
                streams.append(stream)
            metadata = streams[0]
            metadata.seek(0)
            metadata.truncate()
            json.dump(
                {
                    "schema_version": "bts_three_way_formal_lock_v1",
                    "pid": os.getpid(),
                    "acquired_at": utc_now(),
                    "submission_root": str(SUBMISSION_ROOT),
                    "gpu_uuid": gpu["uuid"],
                    "lock_paths": [str(path) for path in paths],
                },
                metadata,
                sort_keys=True,
            )
            metadata.write("\n")
            metadata.flush()
            os.fsync(metadata.fileno())
            return streams
        except BlockingIOError:
            release_formal_run_lock(streams)
            if not announced:
                print(
                    "waiting for repository test/formal-run GPU locks "
                    f"{[str(path) for path in paths]}",
                    file=sys.stderr,
                    flush=True,
                )
                announced = True
            remaining = deadline - time.monotonic()
            if wait_seconds <= 0 or remaining <= 0:
                raise ExecutionError(
                    "another repository test session or formal benchmark is active; "
                    f"locks={[str(path) for path in paths]}"
                )
            time.sleep(min(poll_seconds, remaining))
        except BaseException:
            release_formal_run_lock(streams)
            raise


def release_formal_run_lock(streams: Sequence[Any]) -> None:
    for stream in reversed(tuple(streams)):
        if stream.closed:
            continue
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ExecutionError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def absolute_executable_path(path: Path) -> Path:
    """Make an interpreter path absolute without dereferencing venv symlinks.

    CPython virtual environments commonly expose ``bin/python`` as a symlink to
    the base interpreter.  ``Path.resolve()`` would replace that path with the
    base executable and therefore bypass the venv's ``pyvenv.cfg`` when the
    worker is launched.
    """

    return Path(os.path.abspath(os.fspath(path)))


def deterministic_token_ids(
    length: int, vocab_size: int, seed: int = FORMAL_SEED
) -> list[int]:
    """Generate engine-independent raw token IDs with a specified 32-bit LCG."""

    if length <= 0:
        raise ValueError("prompt length must be positive")
    low = 256
    high = int(vocab_size) - 1024
    if high <= low:
        raise ValueError("vocabulary is too small for the deterministic generator")
    span = high - low
    state = int(seed) & 0xFFFFFFFF
    result: list[int] = []
    for _ in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        result.append(low + state % span)
    return result


def ordered_prompts_sha256(prompt: Sequence[int], request_count: int) -> str:
    """Hash canonical JSON for ``[prompt, ...]`` without materializing copies."""

    if request_count <= 0:
        raise ValueError("request_count must be positive")
    digest = hashlib.sha256()
    digest.update(b"[")
    for request in range(request_count):
        if request:
            digest.update(b",")
        digest.update(b"[")
        for position, token in enumerate(prompt):
            if position:
                digest.update(b",")
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("prompt token IDs must be nonnegative integers")
            digest.update(str(token).encode("ascii"))
        digest.update(b"]")
    digest.update(b"]")
    return digest.hexdigest()


def request_ids(request_count: int) -> list[str]:
    return [f"request-{index:06d}" for index in range(request_count)]


def rotated_arm_order(restart: int) -> tuple[str, ...]:
    offset = int(restart) % len(ARM_ORDER)
    return tuple((*ARM_ORDER[offset:], *ARM_ORDER[:offset]))


def engine_for_arm(arm: str) -> str:
    try:
        return analysis.ARM_SPECS[arm][0]
    except KeyError as error:
        raise ExecutionError(f"unknown benchmark arm: {arm!r}") from error


def mode_for_arm(arm: str) -> str:
    try:
        return analysis.ARM_SPECS[arm][1]
    except KeyError as error:
        raise ExecutionError(f"unknown benchmark arm: {arm!r}") from error


def _source_candidates(engine: str) -> list[Path]:
    roots = [SUBMISSION_ROOT / "bts", HERE]
    if engine == "fastdeploy":
        roots.append(SUBMISSION_ROOT / "integrations" / "fastdeploy")
    elif engine == "vllm":
        roots.append(SUBMISSION_ROOT / "integrations" / "vllm")
    else:
        raise ExecutionError(f"unsupported engine for source manifest: {engine}")
    files = {
        path.resolve()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".cpp", ".cu", ".json", ".toml", ".sh"}
        and "__pycache__" not in path.parts
    }
    root_pyproject = SUBMISSION_ROOT / "pyproject.toml"
    if root_pyproject.is_file():
        files.add(root_pyproject.resolve())
    return sorted(files)


def source_manifest(engine: str) -> dict[str, Any]:
    entries = {
        path.relative_to(SUBMISSION_ROOT).as_posix(): file_sha256(path)
        for path in _source_candidates(engine)
    }
    return {
        "schema_version": f"three_way_{engine}_sources_v1",
        "engine": engine,
        "files": entries,
        "sha256": analysis.canonical_sha256(entries),
    }


def canonical_bts_config() -> tuple[dict[str, Any], str]:
    # Load the constant-only module without executing bts/__init__.py, whose
    # public exports intentionally import Torch kernels.  This keeps dry-run
    # independent of both engine stacks and CUDA-capable Python packages.
    path = SUBMISSION_ROOT / "bts" / "config.py"
    module_spec = importlib.util.spec_from_file_location(
        "_three_way_frozen_bts_config", path
    )
    if module_spec is None or module_spec.loader is None:
        raise ExecutionError(f"cannot load frozen BTS config module: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    config = dict(module.describe())
    return config, analysis.canonical_sha256(config)


def validate_bts_config_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    canonical, actual = canonical_bts_config()
    if actual != expected_sha256:
        raise ExecutionError(
            "BTS config does not match the frozen algorithm: "
            f"expected={expected_sha256}, actual={actual}"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"cannot read BTS config {path}: {error}") from error
    if not isinstance(data, dict):
        raise ExecutionError("BTS config must be a JSON object")
    for wrapper in ("runtime_config", "algorithm"):
        if wrapper in data:
            if set(data) != {wrapper} or not isinstance(data[wrapper], dict):
                raise ExecutionError(f"{wrapper} must be the only top-level object")
            data = data[wrapper]
            break
    legacy = {
        "block": {
            "block_size": "block_size",
            "calibrate": "calibrate",
            "force_sink": "force_sink",
            "force_recent": "force_recent",
            "low_freq_dim": "low_freq_dim",
            "high_freq_dim": "high_freq_dim",
            "low_freq_threshold": "low_freq_threshold",
            "high_freq_threshold": "high_freq_threshold",
        },
        "token": {
            "token_select": "token_selection",
            "token_ratio": "token_ratio",
            "score_mode": "token_score",
            "query_block_size": "query_block_size",
            "dense_tail_tokens": "dense_tail_tokens",
        },
    }
    errors: list[str] = []
    for name, value in data.items():
        if name in canonical:
            if value != canonical[name]:
                errors.append(
                    f"{name}={value!r}, frozen value is {canonical[name]!r}"
                )
        elif name in legacy and isinstance(value, dict):
            for nested, nested_value in value.items():
                target = legacy[name].get(nested)
                if target is None:
                    errors.append(f"unsupported field {name}.{nested}")
                elif nested_value != canonical[target]:
                    errors.append(
                        f"{name}.{nested}={nested_value!r}, frozen value is "
                        f"{canonical[target]!r}"
                    )
        elif name == "token_sparse":
            if value is not True:
                errors.append("token_sparse must be true")
        else:
            errors.append(f"unsupported field {name}")
    if errors:
        raise ExecutionError("BTS config is not frozen: " + "; ".join(errors))
    return canonical


def _model_dtype(config: Mapping[str, Any]) -> str:
    return str(config.get("torch_dtype", config.get("dtype", ""))).lower()


def normalized_model_contract(model: Path) -> dict[str, Any]:
    config_path = model / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"cannot read model config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ExecutionError(f"model config must be an object: {config_path}")
    hidden = int(config.get("hidden_size", -1))
    heads = int(config.get("num_attention_heads", -1))
    head_dim = config.get("head_dim")
    if head_dim is None and heads > 0 and hidden % heads == 0:
        head_dim = hidden // heads
    result = {
        "model_type": config.get("model_type"),
        "hidden_size": hidden,
        "num_hidden_layers": int(config.get("num_hidden_layers", -1)),
        "num_attention_heads": heads,
        "num_key_value_heads": int(config.get("num_key_value_heads", -1)),
        "head_dim": int(head_dim) if head_dim is not None else -1,
        "vocab_size": int(config.get("vocab_size", -1)),
        "max_position_embeddings": int(config.get("max_position_embeddings", -1)),
        "dtype": _model_dtype(config),
        "rope_scaling": config.get("rope_scaling"),
        "quantization_config": config.get("quantization_config"),
    }
    shape = {field: result[field] for field in EXPECTED_MODEL_SHAPE}
    if shape != EXPECTED_MODEL_SHAPE:
        raise ExecutionError(
            f"{model} is not the frozen Qwen3-8B shape: {shape!r}"
        )
    if result["dtype"] not in {"bfloat16", "bf16", "paddle.bfloat16"}:
        raise ExecutionError(f"{model} must declare BF16, got {result['dtype']!r}")
    if result["max_position_embeddings"] < analysis.FORMAL_CONTEXT_LENGTH:
        raise ExecutionError(
            f"{model} max_position_embeddings does not cover 128K"
        )
    if result["vocab_size"] <= 1280:
        raise ExecutionError(f"{model} has an invalid vocab_size")
    if result["quantization_config"] not in (None, {}, False):
        raise ExecutionError(f"{model} must be unquantized BF16")
    return result


def validate_cross_engine_models(
    fastdeploy_model: Path, vllm_model: Path
) -> tuple[dict[str, Any], int]:
    fastdeploy = normalized_model_contract(fastdeploy_model)
    vllm = normalized_model_contract(vllm_model)
    comparable_fields = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rope_scaling",
    )
    differences = {
        field: (fastdeploy[field], vllm[field])
        for field in comparable_fields
        if fastdeploy[field] != vllm[field]
    }
    if differences:
        raise ExecutionError(
            "FastDeploy and vLLM model configs are not the same logical model: "
            f"{differences}"
        )
    return {field: fastdeploy[field] for field in comparable_fields}, int(
        fastdeploy["vocab_size"]
    )


_WEIGHT_INDEX = "model.safetensors.index.json"
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def _paired_file_hashes(
    left_root: Path,
    right_root: Path,
    filenames: Sequence[str],
    *,
    label: str,
) -> dict[str, str]:
    """Hash and require identical logical files, avoiding duplicate I/O for links."""

    result: dict[str, str] = {}
    for filename in filenames:
        left = left_root / filename
        right = right_root / filename
        if not left.is_file() or not right.is_file():
            raise ExecutionError(
                f"{label} file must exist in both model directories: {filename}"
            )
        try:
            same_file = os.path.samefile(left, right)
        except OSError:
            same_file = False
        left_hash = file_sha256(left)
        right_hash = left_hash if same_file else file_sha256(right)
        if left_hash != right_hash:
            raise ExecutionError(
                f"FastDeploy/vLLM {label} file differs: {filename}"
            )
        result[filename] = left_hash
    return result


def derive_cross_engine_identity(
    fastdeploy_model: Path,
    vllm_model: Path,
    normalized_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove identical weights/tokenizer and derive shared identity digests."""

    left_shards = sorted(path.name for path in fastdeploy_model.glob("*.safetensors"))
    right_shards = sorted(path.name for path in vllm_model.glob("*.safetensors"))
    if left_shards != right_shards or len(left_shards) != 5:
        raise ExecutionError(
            "the frozen Qwen3-8B model must expose the same five safetensors "
            f"shards in both directories; FastDeploy={left_shards}, vLLM={right_shards}"
        )
    weight_files = [*left_shards, _WEIGHT_INDEX]
    weight_hashes = _paired_file_hashes(
        fastdeploy_model,
        vllm_model,
        weight_files,
        label="weight",
    )
    tokenizer_hashes = _paired_file_hashes(
        fastdeploy_model,
        vllm_model,
        _TOKENIZER_FILES,
        label="tokenizer",
    )
    logical_payload = {
        "schema_version": "logical_qwen3_cross_engine_identity_v1",
        "normalized_model_config": dict(normalized_config),
        "weight_files_sha256": weight_hashes,
    }
    tokenizer_payload = {
        "schema_version": "qwen3_tokenizer_identity_v1",
        "files_sha256": tokenizer_hashes,
    }
    return {
        "logical_model_sha256": analysis.canonical_sha256(logical_payload),
        "tokenizer_sha256": analysis.canonical_sha256(tokenizer_payload),
        "logical_model_payload": logical_payload,
        "tokenizer_payload": tokenizer_payload,
    }


def build_workload(vocab_size: int, request_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = deterministic_token_ids(
        analysis.FORMAL_INPUT_TOKENS, vocab_size, FORMAL_SEED
    )
    ids_hash = analysis.canonical_sha256(request_ids(request_count))
    prompts_hash = ordered_prompts_sha256(prompt, request_count)
    contract = analysis.build_workload_contract(
        request_count=request_count,
        ordered_request_ids_sha256=ids_hash,
        ordered_prompt_tokens_sha256=prompts_hash,
    )
    generator = {
        "name": "lcg32_raw_token_ids_v1",
        "seed": FORMAL_SEED,
        "vocab_size": vocab_size,
        "low_token_id": 256,
        "exclusive_high_token_id": vocab_size - 1024,
        "prompt_tokens": analysis.FORMAL_INPUT_TOKENS,
        "request_count": request_count,
        "all_requests_share_one_prompt": True,
        "ordered_request_ids_sha256": ids_hash,
        "ordered_prompt_tokens_sha256": prompts_hash,
    }
    return contract, generator


def engine_config_contract(
    engine: str,
    *,
    logical_model_sha256: str,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    """Return the route-independent engine config shared by dense and BTS."""

    common = {
        "logical_model_sha256": logical_model_sha256,
        "max_model_len": analysis.FORMAL_CONTEXT_LENGTH,
        "max_num_batched_tokens": analysis.FORMAL_CONTEXT_LENGTH,
        "max_num_seqs": FORMAL_MAX_NUM_SEQS,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": "bfloat16",
        "kv_cache_storage_dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
    }
    if engine == "fastdeploy":
        return {
            "schema_version": "three_way_fastdeploy_engine_config_v1",
            "engine_version": SUPPORTED_FASTDEPLOY_VERSION,
            **common,
            "model_impl": "paddleformers",
            "graph_optimization_config": {
                "use_cudagraph": False,
                "graph_opt_level": 0,
            },
            "fd_attention_backend": "FLASH_ATTN",
            "paddle_flash_attn_version": 2,
            "fastdeploy_flash_attn_version": 2,
            "dense_prefill_op": "flash_attn_unpadded",
            "native_decode_op": "append_attention",
        }
    if engine == "vllm":
        return {
            "schema_version": "three_way_vllm_engine_config_v2",
            "engine_version": SUPPORTED_VLLM_VERSION,
            **common,
            "skip_tokenizer_init": True,
            "kv_cache_dtype_selector": "auto",
            "kv_cache_block_size": VLLM_KV_CACHE_BLOCK_SIZE,
            "attention_config": {"flash_attn_version": 2},
            "fa2_compat_adapter": {
                "name": VLLM_FA2_ADAPTER_NAME,
                "flash_attn_version": SUPPORTED_FLASH_ATTN_VERSION,
                "upstream_extension": "flash_attn_2_cuda",
                "upstream_argument_count": 21,
                "fail_closed": True,
            },
            "generation_config": "vllm",
            "pytorch_alloc_conf": FORMAL_PYTORCH_ALLOC_CONF,
        }
    raise ExecutionError(f"unknown engine config: {engine}")


def _worker_command(
    *,
    python: Path,
    arm: str,
    restart: int,
    model: Path,
    bts_config: Path,
    controller_spec: Path,
    spec_sha256: str,
    observation: Path,
    gpu: str,
) -> list[str]:
    return [
        str(python),
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--arm",
        arm,
        "--restart",
        str(restart),
        "--worker-model",
        str(model),
        "--worker-bts-config",
        str(bts_config),
        "--controller-spec",
        str(controller_spec),
        "--controller-spec-sha256",
        spec_sha256,
        "--worker-output",
        str(observation),
        "--worker-gpu",
        gpu,
    ]


def build_plan(args: argparse.Namespace, *, dry_run: bool) -> dict[str, Any]:
    logical_model_sha256 = (
        require_sha256(args.logical_model_sha256, "--logical-model-sha256")
        if args.logical_model_sha256 is not None
        else None
    )
    tokenizer_sha256 = (
        require_sha256(args.tokenizer_sha256, "--tokenizer-sha256")
        if args.tokenizer_sha256 is not None
        else None
    )
    if args.request_count < FORMAL_REQUEST_COUNT:
        raise ExecutionError(
            f"--request-count must be at least {FORMAL_REQUEST_COUNT}"
        )
    if not math.isclose(
        args.gpu_memory_utilization,
        FORMAL_GPU_MEMORY_UTILIZATION,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ExecutionError(
            "the frozen protocol requires --gpu-memory-utilization=0.90"
        )

    _, bts_config_sha256 = canonical_bts_config()
    vocab_size = int(args.vocab_size)
    model_contract: dict[str, Any] | None = None
    identity_evidence: dict[str, Any] | None = None
    if not dry_run:
        for label, executable in (
            ("--fastdeploy-python", args.fastdeploy_python),
            ("--vllm-python", args.vllm_python),
        ):
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ExecutionError(f"{label} is not an executable file: {executable}")
        for label, model in (
            ("--fastdeploy-model", args.fastdeploy_model),
            ("--vllm-model", args.vllm_model),
        ):
            if not model.is_dir():
                raise ExecutionError(f"{label} is not a directory: {model}")
        if not args.bts_config.is_file():
            raise ExecutionError(f"--bts-config is not a file: {args.bts_config}")
        validate_bts_config_file(args.bts_config, bts_config_sha256)
        model_contract, detected_vocab = validate_cross_engine_models(
            args.fastdeploy_model, args.vllm_model
        )
        if vocab_size != detected_vocab:
            raise ExecutionError(
                f"--vocab-size={vocab_size} differs from model vocab_size={detected_vocab}"
            )
        identity_evidence = derive_cross_engine_identity(
            args.fastdeploy_model, args.vllm_model, model_contract
        )
        detected_logical = identity_evidence["logical_model_sha256"]
        detected_tokenizer = identity_evidence["tokenizer_sha256"]
        if logical_model_sha256 is not None and logical_model_sha256 != detected_logical:
            raise ExecutionError(
                "--logical-model-sha256 differs from the verified shared weights/config"
            )
        if tokenizer_sha256 is not None and tokenizer_sha256 != detected_tokenizer:
            raise ExecutionError(
                "--tokenizer-sha256 differs from the verified shared tokenizer files"
            )
        logical_model_sha256 = detected_logical
        tokenizer_sha256 = detected_tokenizer

    workload, generator = build_workload(vocab_size, args.request_count)
    config_model_identity = (
        logical_model_sha256
        if logical_model_sha256 is not None
        else "<derived-from-matching-five-shard-models-at-execution>"
    )
    engine_configs = {
        engine: engine_config_contract(
            engine,
            logical_model_sha256=config_model_identity,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for engine in ("fastdeploy", "vllm")
    }
    spec_path = args.output.resolve() / "controller-spec.json"
    python_by_engine = {
        "fastdeploy": absolute_executable_path(args.fastdeploy_python),
        "vllm": absolute_executable_path(args.vllm_python),
    }
    model_by_engine = {
        "fastdeploy": args.fastdeploy_model.resolve(),
        "vllm": args.vllm_model.resolve(),
    }
    commands = []
    for restart in range(analysis.FORMAL_RESTARTS):
        for position, arm in enumerate(rotated_arm_order(restart)):
            engine = engine_for_arm(arm)
            arm_dir = (
                args.output.resolve()
                / f"restart-{restart}"
                / arm
                / "attempt-000"
            )
            commands.append(
                {
                    "restart": restart,
                    "position": position,
                    "arm": arm,
                    "engine": engine,
                    "command": _worker_command(
                        python=python_by_engine[engine],
                        arm=arm,
                        restart=restart,
                        model=model_by_engine[engine],
                        bts_config=args.bts_config.resolve(),
                        controller_spec=spec_path,
                        # Filled after the controller spec is materialized.
                        spec_sha256="<controller-spec-sha256>",
                        observation=arm_dir / "observation.json",
                        gpu=args.gpu,
                    ),
                    "log": str(arm_dir / "worker.log"),
                }
            )
    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "protocol": EXECUTION_PROTOCOL,
        "analysis_protocol": analysis.PROTOCOL_NAME,
        "dry_run": dry_run,
        "dry_run_emits_measurements": False,
        "fastdeploy_python": str(args.fastdeploy_python),
        "vllm_python": str(args.vllm_python),
        "fastdeploy_model": str(args.fastdeploy_model),
        "vllm_model": str(args.vllm_model),
        "bts_config": str(args.bts_config),
        "output": str(args.output),
        "gpu": args.gpu,
        "gpu_lease_contract": {
            "enabled": True,
            "mode": "persistent_idle_one_byte_cuda_marker",
            "continuous_across_all_fresh_workers": True,
            "wait_for_gpu_seconds": args.wait_for_gpu_seconds,
            "poll_seconds": args.gpu_poll_seconds,
            "required_continuous_idle_seconds": args.gpu_idle_stability_seconds,
            "contamination_policy": "discard_attempt_keep_lease_wait_and_retry",
            "max_contamination_retries_per_arm": (
                args.max_contamination_retries_per_arm
            ),
            "worker_failure_policy": (
                "discard_attempt_keep_lease_cleanup_wait_and_retry"
            ),
            "max_worker_failure_retries_per_arm": (
                args.max_worker_failure_retries_per_arm
            ),
            "gpu_recovery_timeout_seconds": args.gpu_recovery_timeout_seconds,
        },
        "fixed_contract": {
            "context_length": analysis.FORMAL_CONTEXT_LENGTH,
            "input_tokens_per_request": analysis.FORMAL_INPUT_TOKENS,
            "output_tokens_per_request": analysis.FORMAL_OUTPUT_TOKENS,
            "request_count": args.request_count,
            "restarts": analysis.FORMAL_RESTARTS,
            "warmups_per_fresh_process": FORMAL_WARMUPS,
            "max_model_len": analysis.FORMAL_CONTEXT_LENGTH,
            "max_num_batched_tokens": analysis.FORMAL_CONTEXT_LENGTH,
            "max_num_seqs": FORMAL_MAX_NUM_SEQS,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "eager": True,
            "prefix_caching": False,
            "chunked_prefill": False,
            "disable_custom_all_reduce": True,
            "temperature": 0.0,
            "min_output_tokens": analysis.FORMAL_OUTPUT_TOKENS,
            "effective_ignore_eos": True,
            "seed": FORMAL_SEED,
            "dense_prefill": "flash_attention_2",
            "fastdeploy_decode": "native_append_attention",
            "vllm_decode": "native_vllm_flash_attention_2",
            "bts_variable_topk_emit": BTS_VARIABLE_TOPK_EMIT_CONTRACT,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "identity": {
            "logical_model_sha256": (
                logical_model_sha256
                if logical_model_sha256 is not None
                else "<derived-from-matching-five-shard-models-at-execution>"
            ),
            "tokenizer_sha256": (
                tokenizer_sha256
                if tokenizer_sha256 is not None
                else "<derived-from-matching-tokenizer-files-at-execution>"
            ),
            "bts_config_sha256": bts_config_sha256,
            "model_contract": model_contract,
            "evidence": identity_evidence,
        },
        "workload": workload,
        "generator": generator,
        "engine_configs": engine_configs,
        "engine_config_sha256": {
            engine: analysis.canonical_sha256(config)
            for engine, config in engine_configs.items()
        },
        "arm_order_by_restart": [
            list(rotated_arm_order(restart))
            for restart in range(analysis.FORMAL_RESTARTS)
        ],
        "commands": commands,
    }


def _load_controller_spec(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"cannot load controller spec {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ExecutionError("controller spec must be a JSON object")
    actual = analysis.canonical_sha256(raw)
    if actual != expected_sha256:
        raise ExecutionError(
            f"controller spec hash mismatch: expected={expected_sha256}, actual={actual}"
        )
    if raw.get("schema_version") != WORKER_SPEC_SCHEMA_VERSION:
        raise ExecutionError("unsupported controller spec schema")
    analysis._validate_shared_contract(raw.get("shared_contract"), "shared_contract")
    analysis._validate_workload(raw.get("workload"), "workload")
    if raw.get("shared_contract_sha256") != analysis.canonical_sha256(
        raw["shared_contract"]
    ):
        raise ExecutionError("controller shared contract hash mismatch")
    if raw.get("workload_sha256") != analysis.canonical_sha256(raw["workload"]):
        raise ExecutionError("controller workload hash mismatch")
    return raw


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ExecutionError(f"required Python package is not installed: {name}") from error


def _completion_token_ids_fastdeploy(outputs: Sequence[Any]) -> list[list[int]]:
    result: list[list[int]] = []
    for request_output in outputs:
        completions = request_output.outputs
        if completions is None:
            result.append([])
            continue
        if not isinstance(completions, (list, tuple)):
            completions = (completions,)
        if len(completions) != 1:
            raise ExecutionError("FastDeploy returned multiple completions per request")
        result.append(list(completions[0].token_ids or ()))
    return result


def _completion_token_ids_vllm(outputs: Sequence[Any]) -> list[list[int]]:
    result: list[list[int]] = []
    for request_output in outputs:
        completions = request_output.outputs
        if len(completions) != 1:
            raise ExecutionError("vLLM returned multiple completions per request")
        result.append(list(completions[0].token_ids or ()))
    return result


def _variable_topk_emit_evidence() -> dict[str, Any]:
    """Describe the selector route without importing Torch/Triton in dry-run."""

    return {
        "contract": BTS_VARIABLE_TOPK_EMIT_CONTRACT,
        "environment_variable": BTS_VARIABLE_TOPK_ENV,
        "environment_value": os.environ.get(BTS_VARIABLE_TOPK_ENV),
        "require_environment_variable": BTS_VARIABLE_TOPK_REQUIRE_ENV,
        "require_environment_value": os.environ.get(BTS_VARIABLE_TOPK_REQUIRE_ENV),
        "default_on_when_unset": True,
    }


def _validate_exact_completions(
    completions: Sequence[Sequence[int]], request_count: int
) -> str:
    if len(completions) != request_count:
        raise ExecutionError(
            f"engine completed {len(completions)} requests, expected {request_count}"
        )
    wrong = [
        index
        for index, tokens in enumerate(completions)
        if len(tokens) != analysis.FORMAL_OUTPUT_TOKENS
    ]
    if wrong:
        raise ExecutionError(
            "each measured request must return exactly two tokens; wrong indices="
            f"{wrong[:16]}"
        )
    return analysis.canonical_sha256([list(tokens) for tokens in completions])


def _canonical_dtype(value: Any) -> str:
    """Normalize the dtype spellings exposed by the two pinned runtimes."""

    normalized = str(value).lower().replace("paddle.", "").replace("torch.", "")
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    return aliases.get(normalized, normalized)


def _resolved_fastdeploy_engine_config(llm: Any) -> dict[str, Any]:
    """Read effective FastDeploy 2.5 settings after ``LLM`` construction.

    FastDeploy does not expose every vLLM-style constructor switch.  In
    particular, model/KV dtypes are resolved into ModelConfig/CacheConfig and
    FastDeploy 2.5 has no pipeline-parallel configuration surface.  Keep the
    values and their provenance separate so the report never implies that a
    resolved default was an explicitly passed LLM option.
    """

    cfg = llm.llm_engine.cfg
    graph = cfg.graph_opt_config
    values = {
        "max_model_len": int(cfg.model_config.max_model_len),
        "max_num_batched_tokens": int(
            cfg.scheduler_config.max_num_batched_tokens
        ),
        "max_num_seqs": int(cfg.scheduler_config.max_num_seqs),
        "tensor_parallel_size": int(cfg.parallel_config.tensor_parallel_size),
        # FastDeploy 2.5 has no pipeline_parallel_size EngineArgs/Config
        # member.  With one TP worker and one DP replica this is a single
        # model stage, recorded as an effective capability constraint below.
        "pipeline_parallel_size": 1,
        "data_parallel_size": int(cfg.parallel_config.data_parallel_size),
        "dtype": _canonical_dtype(cfg.model_config.dtype),
        "kv_cache_storage_dtype": _canonical_dtype(cfg.cache_config.cache_dtype),
        "gpu_memory_utilization": float(cfg.cache_config.gpu_memory_utilization),
        "enable_prefix_caching": bool(cfg.cache_config.enable_prefix_caching),
        "enable_chunked_prefill": bool(cfg.cache_config.enable_chunked_prefill),
        "enforce_eager": (
            not bool(graph.use_cudagraph) and int(graph.graph_opt_level) == 0
        ),
        "disable_custom_all_reduce": bool(
            cfg.parallel_config.disable_custom_all_reduce
        ),
        "graph_optimization_config": {
            "use_cudagraph": bool(graph.use_cudagraph),
            "graph_opt_level": int(graph.graph_opt_level),
        },
    }
    return {
        "schema_version": "three_way_fastdeploy_resolved_config_v1",
        "values": values,
        "field_sources": {
            "max_model_len": "llm.llm_engine.cfg.model_config.max_model_len",
            "max_num_batched_tokens": (
                "llm.llm_engine.cfg.scheduler_config.max_num_batched_tokens"
            ),
            "max_num_seqs": "llm.llm_engine.cfg.scheduler_config.max_num_seqs",
            "tensor_parallel_size": (
                "llm.llm_engine.cfg.parallel_config.tensor_parallel_size"
            ),
            "pipeline_parallel_size": (
                "FastDeploy 2.5 exposes no pipeline-parallel EngineArgs/Config "
                "field; TP=DP=1 is one effective model stage"
            ),
            "data_parallel_size": (
                "llm.llm_engine.cfg.parallel_config.data_parallel_size"
            ),
            "dtype": "llm.llm_engine.cfg.model_config.dtype",
            "kv_cache_storage_dtype": "llm.llm_engine.cfg.cache_config.cache_dtype",
            "gpu_memory_utilization": (
                "llm.llm_engine.cfg.cache_config.gpu_memory_utilization"
            ),
            "enable_prefix_caching": (
                "llm.llm_engine.cfg.cache_config.enable_prefix_caching"
            ),
            "enable_chunked_prefill": (
                "llm.llm_engine.cfg.cache_config.enable_chunked_prefill"
            ),
            "enforce_eager": (
                "derived from cfg.graph_opt_config.use_cudagraph=false and "
                "graph_opt_level=0"
            ),
            "disable_custom_all_reduce": (
                "llm.llm_engine.cfg.parallel_config.disable_custom_all_reduce"
            ),
            "graph_optimization_config": "llm.llm_engine.cfg.graph_opt_config",
        },
    }


def _validate_fastdeploy_resolved_config(
    resolved: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    expected_values = {
        field: expected[field]
        for field in (
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "dtype",
            "kv_cache_storage_dtype",
            "gpu_memory_utilization",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "enforce_eager",
            "disable_custom_all_reduce",
            "graph_optimization_config",
        )
    }
    if resolved.get("schema_version") != (
        "three_way_fastdeploy_resolved_config_v1"
    ):
        raise ExecutionError("FastDeploy resolved config has an unsupported schema")
    if resolved.get("values") != expected_values:
        raise ExecutionError(
            "FastDeploy effective engine config differs from the frozen contract: "
            f"observed={resolved.get('values')!r}, expected={expected_values!r}"
        )
    sources = resolved.get("field_sources")
    if not isinstance(sources, dict) or set(sources) != set(expected_values):
        raise ExecutionError("FastDeploy resolved config lacks field provenance")


def _aggregate_fd_audit(
    audits: Sequence[Mapping[str, Any]], field: str
) -> dict[int, int]:
    result: dict[int, int] = {}
    for audit in audits:
        values = audit.get(field)
        if not isinstance(values, dict):
            raise ExecutionError(f"FastDeploy audit lacks {field}")
        for raw_layer, raw_count in values.items():
            layer = int(raw_layer)
            count = int(raw_count)
            if count < 0:
                raise ExecutionError("FastDeploy audit counter is negative")
            result[layer] = result.get(layer, 0) + count
    return result


def _fd_audit_delta(
    baseline: Sequence[Mapping[str, Any]],
    final: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[int, int]:
    before_pids = {int(row["pid"]) for row in baseline}
    after_pids = {int(row["pid"]) for row in final}
    if before_pids != after_pids:
        raise ExecutionError(
            "FastDeploy GPU worker set changed during timing: "
            f"before={sorted(before_pids)}, after={sorted(after_pids)}"
        )
    before = _aggregate_fd_audit(baseline, field)
    after = _aggregate_fd_audit(final, field)
    layers = set(before) | set(after)
    delta = {layer: after.get(layer, 0) - before.get(layer, 0) for layer in layers}
    if any(value < 0 for value in delta.values()):
        raise ExecutionError("FastDeploy audit counters decreased during timing")
    return delta


def _run_fastdeploy_worker(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    prompt: list[int],
) -> dict[str, Any]:
    arm = args.arm
    mode = mode_for_arm(arm)
    version = _package_version("fastdeploy-gpu")
    if version != SUPPORTED_FASTDEPLOY_VERSION:
        raise ExecutionError(
            f"FastDeploy must be {SUPPORTED_FASTDEPLOY_VERSION}, found {version}"
        )
    model_contract = normalized_model_contract(args.worker_model)
    expected_config = spec["model_contract"]
    if expected_config is not None:
        for field, expected in expected_config.items():
            if field in model_contract and model_contract[field] != expected:
                raise ExecutionError(
                    f"FastDeploy model contract changed at {field}: "
                    f"{model_contract[field]!r} != {expected!r}"
                )
    validate_bts_config_file(
        args.worker_bts_config, spec["shared_contract"]["bts_config_sha256"]
    )

    for name in tuple(os.environ):
        if name.startswith("BTS_FASTDEPLOY_") or name.startswith("PRISM_FASTDEPLOY_"):
            os.environ.pop(name, None)
    work_dir = args.worker_output.parent.resolve()
    audit_dir = work_dir / "fastdeploy-audit"
    log_dir = work_dir / "fastdeploy-logs"
    os.environ.update(
        {
            "BTS_FASTDEPLOY_ENABLE": "1",
            "BTS_FASTDEPLOY_MODE": "bts" if mode == "bts" else "dense_rope_compat",
            "BTS_FASTDEPLOY_AUDIT_DIR": str(audit_dir),
            "BTS_FASTDEPLOY_AUDIT_FLUSH": "mmap",
            "BTS_FASTDEPLOY_AUDIT_NUM_LAYERS": str(
                model_contract["num_hidden_layers"]
            ),
            "BTS_FASTDEPLOY_REQUIRE_DEPENDENCY_CONTRACT": "1",
            "BTS_FASTDEPLOY_CONFIG": str(args.worker_bts_config.resolve()),
            "FD_ATTENTION_BACKEND": "FLASH_ATTN",
            "FD_DISABLE_CHUNKED_PREFILL": "1",
            "FD_PLUGINS": "bts",
            "FLAGS_flash_attn_version": "2",
            "FD_LOG_DIR": str(log_dir),
        }
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    if str(FASTDEPLOY_SRC) not in sys.path:
        sys.path.insert(0, str(FASTDEPLOY_SRC))
    from bts_fastdeploy.audit import read_audits
    from bts_fastdeploy.environment import package_provenance, prepare_workers

    launcher_provenance = prepare_workers()
    if package_provenance() != launcher_provenance:
        raise ExecutionError("FastDeploy package provenance changed before launch")
    from fastdeploy import LLM, SamplingParams
    from bts_fastdeploy.plugin import register
    from fastdeploy.model_executor.layers.attention import flash_attn_backend

    register()
    backend_class = flash_attn_backend.FlashAttentionBackend
    expected_mode = "bts" if mode == "bts" else "dense_rope_compat"
    if getattr(backend_class, "_bts_fastdeploy_mode", None) != expected_mode:
        raise ExecutionError("FastDeploy plugin selected the wrong backend mode")
    native_contract = getattr(
        backend_class, "_bts_native_attention_contract", None
    )
    expected_native_contract = {
        "fastdeploy_flash_attn_version": 2,
        "paddle_flash_attn_version": 2,
        "dense_prefill_op": "flash_attn_unpadded",
        "decode_op": "append_attention",
    }
    if native_contract != expected_native_contract:
        raise ExecutionError(
            "FastDeploy did not enforce FA2 prefill/native append decode: "
            f"{native_contract!r}"
        )

    engine_options = {
        "model": str(args.worker_model.resolve()),
        "model_impl": "paddleformers",
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "max_model_len": analysis.FORMAL_CONTEXT_LENGTH,
        "max_num_seqs": FORMAL_MAX_NUM_SEQS,
        "max_num_batched_tokens": analysis.FORMAL_CONTEXT_LENGTH,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "graph_optimization_config": {"use_cudagraph": False, "graph_opt_level": 0},
        "gpu_memory_utilization": float(spec["gpu_memory_utilization"]),
        "disable_custom_all_reduce": True,
    }
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=analysis.FORMAL_OUTPUT_TOKENS,
        min_tokens=analysis.FORMAL_OUTPUT_TOKENS,
        seed=FORMAL_SEED,
    )
    llm = None
    baseline: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    completions: list[list[int]] = []
    elapsed = 0.0
    resolved_engine_config: dict[str, Any] | None = None
    try:
        llm = LLM(**engine_options)
        resolved_engine_config = _resolved_fastdeploy_engine_config(llm)
        _validate_fastdeploy_resolved_config(
            resolved_engine_config, spec["engine_configs"]["fastdeploy"]
        )
        warmup = llm.generate([prompt], sampling, use_tqdm=True)
        _validate_exact_completions(
            _completion_token_ids_fastdeploy(warmup), 1
        )
        baseline = [dict(row) for row in read_audits(audit_dir)]
        if not baseline:
            raise ExecutionError("FastDeploy produced no post-warmup worker audit")
        prompts = [prompt] * int(spec["workload"]["request_count"])
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        elapsed = time.perf_counter() - started
        completions = _completion_token_ids_fastdeploy(outputs)
        final = [dict(row) for row in read_audits(audit_dir)]
        if not final:
            raise ExecutionError("FastDeploy produced no final worker audit")
    finally:
        if llm is not None:
            llm.llm_engine._exit_sub_services()

    request_count = int(spec["workload"]["request_count"])
    if resolved_engine_config is None:
        raise ExecutionError("FastDeploy did not expose its resolved engine config")
    output_sha256 = _validate_exact_completions(completions, request_count)
    expected_layers = set(range(int(model_contract["num_hidden_layers"])))
    for audit in final:
        if audit.get("backend_mode") != expected_mode:
            raise ExecutionError("FastDeploy GPU worker audit has the wrong mode")
        if audit.get("native_attention_contract") != expected_native_contract:
            raise ExecutionError(
                "FastDeploy GPU worker did not observe the native attention contract"
            )
        worker_provenance = audit.get("package_provenance", {})
        for field in ("bts_file", "bts_fastdeploy_file", "required_source_root"):
            if worker_provenance.get(field) != launcher_provenance.get(field):
                raise ExecutionError(
                    f"FastDeploy worker provenance differs at {field}"
                )

    if mode == "bts":
        prefill_delta = _fd_audit_delta(baseline, final, "layers")
        if set(prefill_delta) != expected_layers or any(
            prefill_delta[layer] != request_count for layer in expected_layers
        ):
            raise ExecutionError(
                "FastDeploy BTS did not record exactly one measured prefill "
                "segment per request and model layer"
            )
        dense_prefill_calls = 0
        bts_prefill_calls = request_count
        prefill_evidence = {
            "type": "mmap_per_layer_bts_segment_delta",
            "per_layer": {str(k): prefill_delta[k] for k in sorted(prefill_delta)},
        }
    else:
        rope_delta = _fd_audit_delta(baseline, final, "rope_guard_layers")
        if set(rope_delta) != expected_layers or any(
            rope_delta[layer] <= 0 for layer in expected_layers
        ):
            raise ExecutionError(
                "FastDeploy dense path did not execute its audited RoPE guard "
                "in every model layer"
            )
        if not all(
            row.get("native_flash_attention", {}).get("preserved") is True
            for row in final
        ):
            raise ExecutionError("FastDeploy dense audit did not preserve native FA2")
        dense_prefill_calls = request_count
        bts_prefill_calls = 0
        prefill_evidence = {
            "type": "native_fa2_contract_plus_completed_prefill_requests",
            "per_layer_rope_guard_invocations": {
                str(k): rope_delta[k] for k in sorted(rope_delta)
            },
        }

    # FastDeploy can mix the next 128K prefill with the preceding one-token
    # decode.  Its legacy native_decode_calls counter is therefore not complete
    # for this workload.  The evidence used here is not a static expected count:
    # every measured request actually returned two tokens (hence executed one
    # decode scheduler iteration), while launcher and GPU-worker audits both
    # observed decode_op=append_attention.
    counters = {
        "dense_prefill_calls": dense_prefill_calls,
        "bts_prefill_calls": bts_prefill_calls,
        "dense_decode_calls": request_count,
        "bts_decode_calls": 0,
    }
    return {
        "engine_version": version,
        "measured_seconds": elapsed,
        "completed_requests": request_count,
        "completion_token_ids_sha256": output_sha256,
        "counters": counters,
        "backend_evidence": {
            "requested_dense_attention_backend": "flash_attention_2",
            "bts_variable_topk_emit": _variable_topk_emit_evidence(),
            "observed_native_attention_contract": expected_native_contract,
            "resolved_engine_config": resolved_engine_config,
            "sampling_contract": {
                "temperature": 0.0,
                "min_tokens": analysis.FORMAL_OUTPUT_TOKENS,
                "max_tokens": analysis.FORMAL_OUTPUT_TOKENS,
                "effective_ignore_eos": True,
                "mechanism": "min_tokens_equals_max_tokens_fastdeploy_2_5",
            },
            "launcher_backend_mode": expected_mode,
            "prefill": prefill_evidence,
            "decode": {
                "type": "completed_two_token_generation_plus_observed_native_contract",
                "completed_two_token_requests": request_count,
                "observed_decode_op": "append_attention",
                "legacy_native_decode_counter_not_used": True,
                "reason": "mixed prefill/decode scheduling can undercount it",
            },
            "startup_audits": baseline,
            "final_audits": final,
        },
    }


def _vllm_audit_records(directory: Path) -> list[dict[str, Any]]:
    from bts_vllm.audit import read_audits

    return [dict(row) for row in read_audits(directory)]


def _vllm_fa2_audit_records(directory: Path) -> list[dict[str, Any]]:
    from bts_vllm.fa2_compat import read_call_audits

    try:
        return [dict(row) for row in read_call_audits(directory)]
    except ValueError as error:
        raise ExecutionError(f"invalid vLLM FA2 call audit: {error}") from error


def _vllm_fa2_audit_delta(
    baseline: Sequence[Mapping[str, Any]], final: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    before_by_pid = {int(row["pid"]): row for row in baseline}
    after_by_pid = {int(row["pid"]): row for row in final}
    if set(before_by_pid) != set(after_by_pid):
        raise ExecutionError("vLLM FA2 model-worker PIDs changed during timing")
    before_calls = sum(int(row["calls"]) for row in baseline)
    after_calls = sum(int(row["calls"]) for row in final)
    before_paged = sum(int(row["paged_calls"]) for row in baseline)
    after_paged = sum(int(row["paged_calls"]) for row in final)
    delta = {
        "calls": after_calls - before_calls,
        "paged_calls": after_paged - before_paged,
    }
    if any(value < 0 for value in delta.values()):
        raise ExecutionError("vLLM FA2 call counters decreased during timing")
    return delta


def _validate_vllm_fa2_evidence(
    raw: Any, *, expected_calls: int, expected_calls_per_request: int
) -> None:
    expected_keys = {
        "type",
        "calls",
        "paged_calls",
        "calls_per_request",
        "startup_audits",
        "final_audits",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ExecutionError("vLLM observation has malformed timed FA2 evidence")
    expected_values = {
        "type": "mmap_timed_adapter_call_delta",
        "calls": expected_calls,
        "paged_calls": expected_calls,
        "calls_per_request": expected_calls_per_request,
    }
    if any(raw.get(key) != value for key, value in expected_values.items()):
        raise ExecutionError("vLLM observation has wrong timed FA2 call counts")
    baseline = raw["startup_audits"]
    final = raw["final_audits"]
    if not isinstance(baseline, list) or not isinstance(final, list):
        raise ExecutionError("vLLM timed FA2 audits must be lists")
    if len(baseline) != 1 or len(final) != 1:
        raise ExecutionError("vLLM timed FA2 audit must contain one TP=1 worker")
    if _vllm_fa2_audit_delta(baseline, final) != {
        "calls": expected_calls,
        "paged_calls": expected_calls,
    }:
        raise ExecutionError("vLLM timed FA2 raw audits disagree with their delta")


def validate_vllm_plugin_provenance() -> dict[str, Any]:
    expected = {
        "bts_vllm": "bts_vllm.plugin:register",
        "bts_vllm_fa2_compat": "bts_vllm.fa2_compat:register",
    }
    entry_points = [
        point
        for point in importlib.metadata.entry_points(group="vllm.general_plugins")
        if point.name in expected
    ]
    observed = [(point.name, point.value) for point in entry_points]
    if len(entry_points) != len(expected) or dict(observed) != expected:
        raise ExecutionError(
            "installed vLLM plugin metadata must contain exactly the submitted "
            f"general plugins; expected={expected!r}, observed={observed!r}"
        )
    import bts_vllm
    from bts_vllm import fa2_compat

    package_file = Path(bts_vllm.__file__).resolve()
    adapter_file = Path(fa2_compat.__file__).resolve()
    expected_root = VLLM_SRC.resolve()
    if not package_file.is_relative_to(expected_root) or not adapter_file.is_relative_to(
        expected_root
    ):
        raise ExecutionError(
            "bts_vllm compatibility code imported outside the submitted source "
            f"tree: package={package_file}, adapter={adapter_file}"
        )
    return {
        "entry_points": [
            {"name": name, "value": value} for name, value in sorted(observed)
        ],
        "package_file": str(package_file),
        "adapter_file": str(adapter_file),
        "adapter_name": fa2_compat.ADAPTER_NAME,
        "required_source_root": str(expected_root),
    }


def _vllm_audit_delta(
    baseline: Sequence[Mapping[str, Any]], final: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    from bts_vllm.audit import COUNTER_NAMES

    before_by_pid = {int(row["pid"]): row for row in baseline}
    after_by_pid = {int(row["pid"]): row for row in final}
    if not set(before_by_pid) <= set(after_by_pid):
        raise ExecutionError("a vLLM BTS audit worker disappeared during timing")
    before = {
        name: sum(int(row.get(name, 0)) for row in baseline)
        for name in COUNTER_NAMES
    }
    after = {
        name: sum(int(row.get(name, 0)) for row in final)
        for name in COUNTER_NAMES
    }
    delta = {name: after[name] - before[name] for name in COUNTER_NAMES}
    if any(value < 0 for value in delta.values()):
        raise ExecutionError("vLLM BTS audit counters decreased during timing")
    return delta


def _vllm_plugin_selection(mode: str) -> str:
    if mode == "bts":
        return "bts_vllm_fa2_compat,bts_vllm"
    if mode == "dense":
        return "bts_vllm_fa2_compat"
    raise ExecutionError(f"unsupported vLLM attention mode: {mode}")


def _vllm_engine_options(
    args: argparse.Namespace, spec: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    """Materialize the exact constructor options shared by both vLLM arms."""

    return {
        "model": str(args.worker_model.resolve()),
        "skip_tokenizer_init": True,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "block_size": VLLM_KV_CACHE_BLOCK_SIZE,
        "max_model_len": analysis.FORMAL_CONTEXT_LENGTH,
        "max_num_seqs": FORMAL_MAX_NUM_SEQS,
        "max_num_batched_tokens": analysis.FORMAL_CONTEXT_LENGTH,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
        "attention_backend": "CUSTOM" if mode == "bts" else "FLASH_ATTN",
        "attention_config": {"flash_attn_version": 2},
        "gpu_memory_utilization": float(spec["gpu_memory_utilization"]),
        "disable_custom_all_reduce": True,
        "generation_config": "vllm",
        "seed": FORMAL_SEED,
    }


def _run_vllm_worker(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    prompt: list[int],
) -> dict[str, Any]:
    arm = args.arm
    mode = mode_for_arm(arm)
    version = _package_version("vllm")
    if version != SUPPORTED_VLLM_VERSION:
        raise ExecutionError(f"vLLM must be {SUPPORTED_VLLM_VERSION}, found {version}")
    model_contract = normalized_model_contract(args.worker_model)
    expected_config = spec["model_contract"]
    if expected_config is not None:
        for field, expected in expected_config.items():
            if field in model_contract and model_contract[field] != expected:
                raise ExecutionError(
                    f"vLLM model contract changed at {field}: "
                    f"{model_contract[field]!r} != {expected!r}"
                )
    validate_bts_config_file(
        args.worker_bts_config, spec["shared_contract"]["bts_config_sha256"]
    )
    if str(VLLM_SRC) not in sys.path:
        sys.path.insert(0, str(VLLM_SRC))
    plugin_provenance = validate_vllm_plugin_provenance()
    audit_dir = args.worker_output.parent.resolve() / "vllm-audit"
    fa2_audit_dir = args.worker_output.parent.resolve() / "vllm-fa2-audit"
    from bts_vllm.fa2_compat import FA2_AUDIT_ENV

    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_PLUGINS"] = _vllm_plugin_selection(mode)
    os.environ[FA2_AUDIT_ENV] = str(fa2_audit_dir)
    if mode == "bts":
        os.environ["BTS_VLLM_AUDIT_DIR"] = str(audit_dir)
    else:
        os.environ.pop("BTS_VLLM_AUDIT_DIR", None)
    os.environ["PYTORCH_ALLOC_CONF"] = FORMAL_PYTORCH_ALLOC_CONF
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = FORMAL_PYTORCH_ALLOC_CONF

    from vllm import LLM, SamplingParams, TokensPrompt

    engine_options = _vllm_engine_options(args, spec, mode)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=analysis.FORMAL_OUTPUT_TOKENS,
        min_tokens=analysis.FORMAL_OUTPUT_TOKENS,
        ignore_eos=True,
        # Required by vLLM when skip_tokenizer_init=True.  Completion token IDs
        # remain available and are the only output used by this benchmark.
        detokenize=False,
        seed=FORMAL_SEED,
    )
    llm = None
    baseline: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    fa2_baseline: list[dict[str, Any]] = []
    fa2_final: list[dict[str, Any]] = []
    completions: list[list[int]] = []
    elapsed = 0.0
    resolved_block_size: int | None = None
    try:
        llm = LLM(**engine_options)
        resolved_block_size = int(
            llm.llm_engine.vllm_config.cache_config.block_size
        )
        if resolved_block_size != VLLM_KV_CACHE_BLOCK_SIZE:
            raise ExecutionError(
                "vLLM resolved an incompatible KV-cache block size: "
                f"expected={VLLM_KV_CACHE_BLOCK_SIZE}, "
                f"observed={resolved_block_size}"
            )
        warmup_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt)],
            sampling,
            use_tqdm=False,
        )
        _validate_exact_completions(
            _completion_token_ids_vllm(warmup_outputs), 1
        )
        fa2_baseline = _vllm_fa2_audit_records(fa2_audit_dir)
        if not fa2_baseline:
            raise ExecutionError("vLLM produced no post-warmup FA2 call audit")
        if mode == "bts":
            baseline = _vllm_audit_records(audit_dir)
            if not baseline:
                raise ExecutionError("vLLM BTS produced no post-warmup audit")
        prompts = [
            TokensPrompt(prompt_token_ids=prompt)
            for _ in range(int(spec["workload"]["request_count"]))
        ]
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        completions = _completion_token_ids_vllm(outputs)
        fa2_final = _vllm_fa2_audit_records(fa2_audit_dir)
        if not fa2_final:
            raise ExecutionError("vLLM produced no final FA2 call audit")
        if mode == "bts":
            final = _vllm_audit_records(audit_dir)
            if not final:
                raise ExecutionError("vLLM BTS produced no final audit")
    finally:
        if llm is not None:
            # vLLM 0.17's offline LLM has no public LLM.shutdown(), but its
            # EngineCoreClient does.  Use that version-scoped API so every arm
            # actually releases its EngineCore before the worker exits.
            llm.llm_engine.engine_core.shutdown()
            del llm

    request_count = int(spec["workload"]["request_count"])
    output_sha256 = _validate_exact_completions(completions, request_count)
    num_layers = int(model_contract["num_hidden_layers"])
    fa2_delta = _vllm_fa2_audit_delta(fa2_baseline, fa2_final)
    active_fa2_workers = [
        row for row in fa2_final if int(row.get("calls", 0)) > 0
    ]
    if len(active_fa2_workers) != 1:
        raise ExecutionError("vLLM FA2 audit does not identify one TP=1 model worker")
    fa2_calls_per_request = num_layers * (1 if mode == "bts" else 2)
    expected_fa2_calls = fa2_calls_per_request * request_count
    if fa2_delta != {
        "calls": expected_fa2_calls,
        "paged_calls": expected_fa2_calls,
    }:
        raise ExecutionError(
            "vLLM timed requests did not execute the exact paged FA2 adapter "
            f"call contract: expected={expected_fa2_calls}, observed={fa2_delta}"
        )
    fa2_call_evidence = {
        "type": "mmap_timed_adapter_call_delta",
        "calls": fa2_delta["calls"],
        "paged_calls": fa2_delta["paged_calls"],
        "calls_per_request": fa2_calls_per_request,
        "startup_audits": fa2_baseline,
        "final_audits": fa2_final,
    }
    if mode == "bts":
        delta = _vllm_audit_delta(baseline, final)
        final_model_workers = [
            row for row in final if int(row.get("impl_count", 0)) > 0
        ]
        if (
            len(final_model_workers) != 1
            or sum(int(row.get("impl_count", 0)) for row in final) != num_layers
        ):
            raise ExecutionError(
                "vLLM BTS audit does not contain one complete TP=1 model worker"
            )
        expected_segments = num_layers * request_count
        if delta["sparse_segments"] != expected_segments:
            raise ExecutionError(
                "vLLM BTS did not execute one sparse prefill segment per "
                "request and layer"
            )
        if delta["dense_fallback_segments"] != expected_segments:
            raise ExecutionError(
                "vLLM BTS did not execute one native dense decode segment per "
                "two-token request and layer"
            )
        counters = {
            "dense_prefill_calls": 0,
            "bts_prefill_calls": request_count,
            "dense_decode_calls": request_count,
            "bts_decode_calls": 0,
        }
        backend_evidence = {
            "requested_dense_attention_backend": "flash_attention_2",
            "requested_backend": "CUSTOM",
            "sampling_contract": {
                "temperature": 0.0,
                "min_tokens": analysis.FORMAL_OUTPUT_TOKENS,
                "max_tokens": analysis.FORMAL_OUTPUT_TOKENS,
                "effective_ignore_eos": True,
                "mechanism": "sampling_params_ignore_eos_true",
                "detokenize": False,
            },
            "prefill": {
                "type": "mmap_sparse_segment_delta",
                "segments": delta["sparse_segments"],
                "segments_per_request": num_layers,
            },
            "decode": {
                "type": "mmap_native_dense_fallback_segment_delta",
                "segments": delta["dense_fallback_segments"],
                "segments_per_request": num_layers,
            },
            "startup_audits": baseline,
            "final_audits": final,
            "measured_delta": delta,
            "fa2_adapter_calls": fa2_call_evidence,
            "plugin_provenance": plugin_provenance,
            "fa2_compat_adapter": {
                "name": VLLM_FA2_ADAPTER_NAME,
                "flash_attn_version": SUPPORTED_FLASH_ATTN_VERSION,
                "resolved_kv_cache_block_size": resolved_block_size,
            },
        }
    else:
        counters = {
            "dense_prefill_calls": request_count,
            "bts_prefill_calls": 0,
            "dense_decode_calls": request_count,
            "bts_decode_calls": 0,
        }
        backend_evidence = {
            "requested_dense_attention_backend": "flash_attention_2",
            "requested_backend": "FLASH_ATTN",
            "sampling_contract": {
                "temperature": 0.0,
                "min_tokens": analysis.FORMAL_OUTPUT_TOKENS,
                "max_tokens": analysis.FORMAL_OUTPUT_TOKENS,
                "effective_ignore_eos": True,
                "mechanism": "sampling_params_ignore_eos_true",
                "detokenize": False,
            },
            "prefill": {
                "type": "completed_raw_token_requests_plus_worker_log_backend_evidence",
                "completed_requests": request_count,
            },
            "decode": {
                "type": "completed_two_token_generation_plus_worker_log_backend_evidence",
                "completed_two_token_requests": request_count,
            },
            "fa2_adapter_calls": fa2_call_evidence,
            "plugin_provenance": plugin_provenance,
            "fa2_compat_adapter": {
                "name": VLLM_FA2_ADAPTER_NAME,
                "flash_attn_version": SUPPORTED_FLASH_ATTN_VERSION,
                "resolved_kv_cache_block_size": resolved_block_size,
            },
        }
    backend_evidence["bts_variable_topk_emit"] = _variable_topk_emit_evidence()
    return {
        "engine_version": version,
        "measured_seconds": elapsed,
        "completed_requests": request_count,
        "completion_token_ids_sha256": output_sha256,
        "counters": counters,
        "backend_evidence": backend_evidence,
    }


def worker_main(args: argparse.Namespace) -> int:
    if args.arm not in analysis.ARMS:
        raise ExecutionError(f"worker arm must be one of {analysis.ARMS}")
    if args.restart not in range(analysis.FORMAL_RESTARTS):
        raise ExecutionError("worker restart is outside the frozen 0..4 range")
    expected_engine = engine_for_arm(args.arm)
    if (
        os.environ.get(BTS_VARIABLE_TOPK_ENV) != "1"
        or os.environ.get(BTS_VARIABLE_TOPK_REQUIRE_ENV) != "1"
    ):
        raise ExecutionError(
            "formal worker requires deterministic variable Top-K to be enabled "
            "and fail-closed"
        )
    spec = _load_controller_spec(
        args.controller_spec, args.controller_spec_sha256
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.worker_gpu:
        raise ExecutionError(
            "worker CUDA_VISIBLE_DEVICES differs from the controller GPU selector"
        )
    if spec.get("gpu_selector") != args.worker_gpu:
        raise ExecutionError("worker GPU selector differs from controller spec")
    expected_engine_config = engine_config_contract(
        expected_engine,
        logical_model_sha256=spec["shared_contract"]["logical_model_sha256"],
        gpu_memory_utilization=spec["shared_contract"]["gpu_memory_utilization"],
    )
    if spec.get("engine_configs", {}).get(expected_engine) != expected_engine_config:
        raise ExecutionError("controller spec has a non-frozen engine config")
    expected_engine_config_hash = analysis.canonical_sha256(expected_engine_config)
    if (
        spec.get("engine_config_sha256", {}).get(expected_engine)
        != expected_engine_config_hash
    ):
        raise ExecutionError("controller engine-config object/hash mismatch")
    expected_source = spec["source_manifest_sha256"].get(expected_engine)
    actual_source = source_manifest(expected_engine)["sha256"]
    if expected_source != actual_source:
        raise ExecutionError(
            f"{expected_engine} runtime source changed before worker execution"
        )
    model_contract = normalized_model_contract(args.worker_model)
    vocab_size = int(model_contract["vocab_size"])
    if vocab_size != int(spec["generator"]["vocab_size"]):
        raise ExecutionError("worker model vocabulary differs from frozen workload")
    prompt = deterministic_token_ids(
        analysis.FORMAL_INPUT_TOKENS, vocab_size, FORMAL_SEED
    )
    prompt_hash = ordered_prompts_sha256(
        prompt, int(spec["workload"]["request_count"])
    )
    if prompt_hash != spec["workload"]["ordered_prompt_tokens_sha256"]:
        raise ExecutionError("worker regenerated different ordered token IDs")

    engine_result = _run_engine_worker(args, spec, prompt, expected_engine)
    elapsed = float(engine_result["measured_seconds"])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ExecutionError(f"worker measured invalid elapsed time: {elapsed!r}")
    if source_manifest(expected_engine)["sha256"] != expected_source:
        raise ExecutionError("runtime source changed while worker was active")
    request_count = int(spec["workload"]["request_count"])
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "arm": args.arm,
        "restart": args.restart,
        "controller_spec_sha256": args.controller_spec_sha256,
        "engine": expected_engine,
        "engine_version": engine_result["engine_version"],
        "engine_config_sha256": spec["engine_config_sha256"][expected_engine],
        "source_manifest_sha256": expected_source,
        "measured_seconds": elapsed,
        "completed_requests": int(engine_result["completed_requests"]),
        "total_input_tokens": request_count * analysis.FORMAL_INPUT_TOKENS,
        "total_output_tokens": request_count * analysis.FORMAL_OUTPUT_TOKENS,
        "request_order_sha256": spec["workload"][
            "ordered_request_ids_sha256"
        ],
        "ordered_prompt_tokens_sha256": prompt_hash,
        "completion_token_ids_sha256": engine_result[
            "completion_token_ids_sha256"
        ],
        "counters": engine_result["counters"],
        "backend_evidence": engine_result["backend_evidence"],
        "finished_at": utc_now(),
    }
    validate_worker_observation(
        observation,
        expected_arm=args.arm,
        expected_restart=args.restart,
        spec=spec,
    )
    atomic_write_json(args.worker_output, observation)
    return 0


def _run_engine_worker(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    prompt: list[int],
    expected_engine: str,
) -> dict[str, Any]:
    """Dispatch one engine and make formal selector failures non-retryable."""

    try:
        return (
            _run_fastdeploy_worker(args, spec, prompt)
            if expected_engine == "fastdeploy"
            else _run_vllm_worker(args, spec, prompt)
        )
    except RuntimeError as error:
        if not str(error).startswith(BTS_VARIABLE_TOPK_REQUIRED_ERROR_PREFIX):
            raise
        # main() maps ExecutionError to status 4; the controller reserves that
        # status for deterministic protocol/configuration failures and will not
        # retry forever when the required native selector cannot run.
        raise ExecutionError(str(error)) from error


_OBSERVATION_KEYS = {
    "schema_version",
    "arm",
    "restart",
    "controller_spec_sha256",
    "engine",
    "engine_version",
    "engine_config_sha256",
    "source_manifest_sha256",
    "measured_seconds",
    "completed_requests",
    "total_input_tokens",
    "total_output_tokens",
    "request_order_sha256",
    "ordered_prompt_tokens_sha256",
    "completion_token_ids_sha256",
    "counters",
    "backend_evidence",
    "finished_at",
}
_COUNTER_KEYS = {
    "dense_prefill_calls",
    "bts_prefill_calls",
    "dense_decode_calls",
    "bts_decode_calls",
}


def validate_worker_observation(
    raw: Any,
    *,
    expected_arm: str,
    expected_restart: int,
    spec: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _OBSERVATION_KEYS:
        raise ExecutionError("worker observation has wrong top-level fields")
    if raw["schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ExecutionError("worker observation has unsupported schema")
    if raw["arm"] != expected_arm or raw["restart"] != expected_restart:
        raise ExecutionError("worker observation arm/restart mismatch")
    engine = engine_for_arm(expected_arm)
    if raw["engine"] != engine:
        raise ExecutionError("worker observation engine mismatch")
    if raw["controller_spec_sha256"] != analysis.canonical_sha256(spec):
        raise ExecutionError("worker observation controller-spec hash mismatch")
    if raw["source_manifest_sha256"] != spec["source_manifest_sha256"][engine]:
        raise ExecutionError("worker observation source-manifest hash mismatch")
    expected_engine_config = engine_config_contract(
        engine,
        logical_model_sha256=spec["shared_contract"]["logical_model_sha256"],
        gpu_memory_utilization=spec["shared_contract"]["gpu_memory_utilization"],
    )
    declared_engine_config = spec.get("engine_configs", {}).get(engine)
    declared_engine_config_hash = spec.get("engine_config_sha256", {}).get(engine)
    if declared_engine_config != expected_engine_config:
        raise ExecutionError("controller spec has a non-frozen engine config")
    expected_engine_config_hash = analysis.canonical_sha256(expected_engine_config)
    if declared_engine_config_hash != expected_engine_config_hash:
        raise ExecutionError("controller engine-config object/hash mismatch")
    if raw["engine_config_sha256"] != expected_engine_config_hash:
        raise ExecutionError("worker observation engine-config hash mismatch")
    if not isinstance(raw["engine_version"], str) or not raw["engine_version"]:
        raise ExecutionError("worker observation lacks engine version")
    elapsed = raw["measured_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0
    ):
        raise ExecutionError("worker observation elapsed time is invalid")
    requests = int(spec["workload"]["request_count"])
    expected_values = {
        "completed_requests": requests,
        "total_input_tokens": requests * analysis.FORMAL_INPUT_TOKENS,
        "total_output_tokens": requests * analysis.FORMAL_OUTPUT_TOKENS,
        "request_order_sha256": spec["workload"][
            "ordered_request_ids_sha256"
        ],
        "ordered_prompt_tokens_sha256": spec["workload"][
            "ordered_prompt_tokens_sha256"
        ],
    }
    for field, expected in expected_values.items():
        if raw[field] != expected:
            raise ExecutionError(
                f"worker observation {field} mismatch: {raw[field]!r} != {expected!r}"
            )
    require_sha256(raw["completion_token_ids_sha256"], "completion token hash")
    counters = raw["counters"]
    if not isinstance(counters, dict) or set(counters) != _COUNTER_KEYS:
        raise ExecutionError("worker observation has wrong counter fields")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counters.values()
    ):
        raise ExecutionError("worker counters must be nonnegative integers")
    mode = mode_for_arm(expected_arm)
    if counters["dense_decode_calls"] <= 0 or counters["bts_decode_calls"] != 0:
        raise ExecutionError("worker did not prove native dense-only decode")
    if mode == "bts":
        if counters["bts_prefill_calls"] <= 0 or counters["dense_prefill_calls"] != 0:
            raise ExecutionError("BTS worker prefill counters are invalid")
    elif counters["dense_prefill_calls"] <= 0 or counters["bts_prefill_calls"] != 0:
        raise ExecutionError("dense worker prefill counters are invalid")
    evidence = raw["backend_evidence"]
    if not isinstance(evidence, dict):
        raise ExecutionError("worker observation lacks backend evidence")
    if evidence.get("requested_dense_attention_backend") != "flash_attention_2":
        raise ExecutionError("worker did not request FlashAttention 2")
    expected_selector_evidence = {
        "contract": BTS_VARIABLE_TOPK_EMIT_CONTRACT,
        "environment_variable": BTS_VARIABLE_TOPK_ENV,
        "environment_value": "1",
        "require_environment_variable": BTS_VARIABLE_TOPK_REQUIRE_ENV,
        "require_environment_value": "1",
        "default_on_when_unset": True,
    }
    if evidence.get("bts_variable_topk_emit") != expected_selector_evidence:
        raise ExecutionError(
            "worker did not prove the deterministic variable Top-K emit contract"
        )
    expected_sampling_mechanism = (
        "min_tokens_equals_max_tokens_fastdeploy_2_5"
        if engine == "fastdeploy"
        else "sampling_params_ignore_eos_true"
    )
    expected_sampling_contract = {
        "temperature": 0.0,
        "min_tokens": analysis.FORMAL_OUTPUT_TOKENS,
        "max_tokens": analysis.FORMAL_OUTPUT_TOKENS,
        "effective_ignore_eos": True,
        "mechanism": expected_sampling_mechanism,
    }
    if engine == "vllm":
        expected_sampling_contract["detokenize"] = False
    if evidence.get("sampling_contract") != expected_sampling_contract:
        raise ExecutionError("worker did not prove the frozen sampling semantics")
    if engine == "vllm":
        if evidence.get("fa2_compat_adapter") != {
            "name": VLLM_FA2_ADAPTER_NAME,
            "flash_attn_version": SUPPORTED_FLASH_ATTN_VERSION,
            "resolved_kv_cache_block_size": VLLM_KV_CACHE_BLOCK_SIZE,
        }:
            raise ExecutionError(
                "vLLM observation lacks the resolved CUDA-12 FA2 adapter contract"
            )
        plugin = evidence.get("plugin_provenance", {})
        if plugin.get("adapter_name") != VLLM_FA2_ADAPTER_NAME or plugin.get(
            "entry_points"
        ) != [
            {"name": "bts_vllm", "value": "bts_vllm.plugin:register"},
            {
                "name": "bts_vllm_fa2_compat",
                "value": "bts_vllm.fa2_compat:register",
            },
        ]:
            raise ExecutionError(
                "vLLM observation lacks exact submitted plugin provenance"
            )
        model_contract = spec.get("model_contract")
        if not isinstance(model_contract, dict) or not isinstance(
            model_contract.get("num_hidden_layers"), int
        ):
            raise ExecutionError("controller spec lacks vLLM model layer provenance")
        fa2_calls_per_request = int(model_contract["num_hidden_layers"]) * (
            1 if mode == "bts" else 2
        )
        _validate_vllm_fa2_evidence(
            evidence.get("fa2_adapter_calls"),
            expected_calls=fa2_calls_per_request * requests,
            expected_calls_per_request=fa2_calls_per_request,
        )
    if engine == "fastdeploy":
        _validate_fastdeploy_resolved_config(
            evidence.get("resolved_engine_config", {}), expected_engine_config
        )
        native = evidence.get("observed_native_attention_contract")
        if native != {
            "fastdeploy_flash_attn_version": 2,
            "paddle_flash_attn_version": 2,
            "dense_prefill_op": "flash_attn_unpadded",
            "decode_op": "append_attention",
        }:
            raise ExecutionError("FastDeploy observation lacks the native FA2 contract")
        if evidence.get("decode", {}).get("type") != (
            "completed_two_token_generation_plus_observed_native_contract"
        ):
            raise ExecutionError("FastDeploy decode evidence is not runtime-derived")
    elif mode == "bts":
        if evidence.get("prefill", {}).get("type") != "mmap_sparse_segment_delta":
            raise ExecutionError("vLLM BTS prefill lacks raw mmap evidence")
        if evidence.get("decode", {}).get("type") != (
            "mmap_native_dense_fallback_segment_delta"
        ):
            raise ExecutionError("vLLM BTS decode lacks raw mmap evidence")
    return raw


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ExecutionError(f"cannot query NVIDIA GPU inventory: {error}") from error
    rows = []
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) != 5:
            continue
        index, uuid, name, memory, capability = (
            field.strip() for field in fields
        )
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(memory),
                "compute_capability": capability,
            }
        )
    if not rows:
        raise ExecutionError("nvidia-smi returned no GPUs")
    return rows


def resolve_gpu(selector: str) -> dict[str, Any]:
    inventory = _gpu_inventory()
    if selector.isdigit():
        matches = [row for row in inventory if row["index"] == int(selector)]
    elif selector.startswith("GPU-"):
        matches = [row for row in inventory if row["uuid"].startswith(selector)]
    else:
        raise ExecutionError("--gpu must be one physical index or GPU UUID")
    if len(matches) != 1:
        raise ExecutionError(f"cannot resolve --gpu={selector!r} uniquely")
    gpu = matches[0]
    if "H100" not in gpu["name"] or gpu["compute_capability"] != "9.0":
        raise ExecutionError(
            "the frozen protocol requires one H100 (sm90), found "
            f"{gpu['name']} compute capability {gpu['compute_capability']}"
        )
    return gpu


def gpu_processes(gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--id={gpu['index']}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30
    )
    rows = []
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) != 3:
            continue
        raw_pid, name, raw_memory = (field.strip() for field in fields)
        if raw_pid.isdigit():
            rows.append(
                {
                    "pid": int(raw_pid),
                    "process_name": name,
                    "used_gpu_memory_mib": (
                        int(raw_memory) if raw_memory.isdigit() else raw_memory
                    ),
                }
            )
    return rows


def descendant_pids(root_pid: int) -> set[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        try:
            pid, parent = (int(value) for value in line.split())
        except (TypeError, ValueError):
            continue
        children.setdefault(parent, []).append(pid)
    found: set[int] = set()
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        if pid in found:
            continue
        found.add(pid)
        stack.extend(children.get(pid, ()))
    return found


def pids_with_run_token(token: str) -> set[int]:
    expected = f"{RUN_TOKEN_ENV}={token}".encode("utf-8")
    found = set()
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            values = path.read_bytes().split(b"\0")
        except (OSError, PermissionError, ProcessLookupError):
            continue
        if expected in values:
            found.add(int(path.parent.name))
    return found


def process_forensics(pid: int, *, token: str) -> dict[str, Any]:
    """Capture bounded /proc evidence without persisting environment contents."""

    evidence: dict[str, Any] = {"pid": pid, "captured_at": utc_now()}
    root = Path("/proc") / str(pid)
    try:
        raw = (root / "cmdline").read_bytes().split(b"\0")
        evidence["cmdline"] = [
            item.decode("utf-8", errors="replace") for item in raw if item
        ]
    except (OSError, PermissionError, ProcessLookupError):
        evidence["cmdline_unavailable"] = True
    try:
        status = {}
        for line in (root / "status").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        if "PPid" in status:
            evidence["ppid"] = int(status["PPid"])
        if "Uid" in status:
            evidence["uids"] = [int(value) for value in status["Uid"].split()]
    except (OSError, PermissionError, ProcessLookupError, ValueError):
        evidence["status_unavailable"] = True
    try:
        identity = _process_identity(pid)
        if identity is not None:
            evidence["process_state"] = identity[0]
            evidence["start_time_clock_ticks"] = identity[1]
    except (OSError, PermissionError, ProcessLookupError, ValueError):
        evidence["start_time_unavailable"] = True
    try:
        environment = (root / "environ").read_bytes().split(b"\0")
        prefix = f"{RUN_TOKEN_ENV}=".encode("utf-8")
        expected = prefix + token.encode("utf-8")
        evidence["has_three_way_run_token"] = any(
            value.startswith(prefix) for value in environment
        )
        evidence["matches_current_attempt_token"] = expected in environment
    except (OSError, PermissionError, ProcessLookupError):
        evidence["environment_token_state"] = "unavailable"
    try:
        evidence["process_group_id"] = os.getpgid(pid)
        evidence["session_id"] = os.getsid(pid)
    except (OSError, ProcessLookupError):
        evidence["process_group_unavailable"] = True
    return evidence


def _process_identity(pid: int) -> tuple[str, int] | None:
    """Return Linux process state/starttime without being confused by spaces."""

    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except (OSError, PermissionError, ProcessLookupError):
        return None
    # ``comm`` (field 2) is parenthesized and may itself contain spaces.  The
    # suffix starts at field 3 (state); starttime is field 22, suffix index 19.
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        return None
    suffix = stat[closing_parenthesis + 2 :].split()
    if len(suffix) <= 19:
        return None
    try:
        return suffix[0], int(suffix[19])
    except ValueError:
        return None


def external_gpu_process_error(
    *,
    stage: str,
    gpu: Mapping[str, Any],
    lease_pid: int,
    worker_pid: int | None,
    token: str,
    first_observed: Sequence[Mapping[str, Any]],
    confirmed_observed: Sequence[Mapping[str, Any]],
    allowed_pids: set[int],
    check_count: int,
    started_at: str,
    worker_gpu_pids: set[int],
) -> ExternalGpuProcessDetected:
    outsiders = [
        dict(row) for row in confirmed_observed if int(row["pid"]) not in allowed_pids
    ]
    evidence = {
        "schema_version": "bts_three_way_contaminated_attempt_v1",
        "status": "interrupted_by_external_gpu_process",
        "stage": stage,
        "detected_at": utc_now(),
        "started_at": started_at,
        "gpu_index": gpu["index"],
        "lease_pid": lease_pid,
        "worker_pid": worker_pid,
        "check_count": check_count,
        "all_checks_exclusive": False,
        "observation_eligible": False,
        "observed_worker_gpu_pids": sorted(worker_gpu_pids),
        "first_gpu_snapshot": [dict(row) for row in first_observed],
        "confirmed_gpu_snapshot": [dict(row) for row in confirmed_observed],
        "outsiders": outsiders,
        "outsider_process_forensics": [
            process_forensics(int(row["pid"]), token=token) for row in outsiders
        ],
    }
    return ExternalGpuProcessDetected(
        "selected GPU was contaminated by unrelated processes; "
        f"stage={stage}, outsiders={outsiders}",
        evidence=evidence,
    )


def _terminate_owned_processes(
    process: subprocess.Popen[Any], token: str
) -> dict[str, Any]:
    """Terminate only identity-verified processes owned by one worker attempt."""

    started_at = utc_now()
    identities: dict[int, int] = {}
    try:
        process.poll()
    except (AttributeError, ChildProcessError):
        pass
    # Once Popen has reaped the launcher, its numeric PID may be reused by an
    # unrelated process.  Across cleanup retries, never rediscover a reaped
    # launcher's new namesake or any of that process's descendants.
    root_identity = (
        _process_identity(process.pid) if process.returncode is None else None
    )
    root_start_time = None if root_identity is None else root_identity[1]

    def remember(pid: int) -> None:
        if pid == os.getpid() or pid in identities:
            return
        identity = _process_identity(pid)
        if identity is not None and identity[0] != "Z":
            identities[pid] = identity[1]

    def discover() -> None:
        candidates = set(pids_with_run_token(token))
        current_root = _process_identity(process.pid)
        if (
            process.returncode is None
            and root_start_time is not None
            and current_root is not None
            and current_root[1] == root_start_time
        ):
            candidates.add(process.pid)
            candidates.update(descendant_pids(process.pid))
        for owned_pid in candidates:
            remember(owned_pid)

    def alive() -> set[int]:
        # poll() reaps the direct child, so a dead launcher is never mistaken
        # for a live zombie by kill(pid, 0).
        try:
            process.poll()
        except (AttributeError, ChildProcessError):
            pass
        discover()
        result = set()
        for owned_pid, expected_start in identities.items():
            identity = _process_identity(owned_pid)
            if (
                identity is not None
                and identity[0] != "Z"
                and identity[1] == expected_start
            ):
                result.add(owned_pid)
        return result

    def signal_verified(pids: Iterable[int], signum: int) -> None:
        for owned_pid in sorted(pids):
            identity = _process_identity(owned_pid)
            if identity is None or identity[1] != identities.get(owned_pid):
                continue
            try:
                os.kill(owned_pid, signum)
            except ProcessLookupError:
                pass

    discover()
    initial_pids = sorted(identities)
    signal_verified(alive(), signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    remaining = alive()
    while remaining and time.monotonic() < deadline:
        time.sleep(0.2)
        remaining = alive()
    used_sigkill = bool(remaining)
    if remaining:
        signal_verified(remaining, signal.SIGKILL)
        deadline = time.monotonic() + 15.0
        while remaining and time.monotonic() < deadline:
            time.sleep(0.2)
            remaining = alive()
    # A final wait closes the small interval in which poll() ran just before
    # the launcher became a zombie and the subsequent /proc read saw ``Z``.
    if process.returncode is None and process.pid not in remaining:
        try:
            process.wait(timeout=1.0)
        except (subprocess.TimeoutExpired, ChildProcessError):
            process.poll()
    return {
        "started_at": started_at,
        "finished_at": utc_now(),
        "initial_owned_pids": initial_pids,
        "remaining_owned_pids": sorted(remaining),
        "used_sigkill": used_sigkill,
        "identity_check": "linux_proc_start_time_clock_ticks",
        "launcher_returncode": process.returncode,
    }


def python_library_path(python: Path, inherited: str = "") -> str:
    probe = (
        "import json,site,sys; "
        "print(json.dumps(list(dict.fromkeys(site.getsitepackages()+sys.path))))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        roots = json.loads(result.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        raise ExecutionError(f"cannot inspect Python environment {python}: {error}") from error
    candidates: list[Path] = []
    for raw in roots:
        if not raw:
            continue
        root = Path(raw)
        candidates.append(root / "torch" / "lib")
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            candidates.extend(sorted(nvidia.glob("*/lib")))
    values = [str(path.resolve()) for path in candidates if path.is_dir()]
    values.extend(item for item in inherited.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(values))


def worker_environment(
    *,
    python: Path,
    engine: str,
    gpu_selector: str,
    run_token: str,
    required_variable_topk_failure_marker: Path | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    # The GPU lease uses this helper too and must never inherit a stale marker.
    environment.pop(BTS_VARIABLE_TOPK_FAILURE_MARKER_ENV, None)
    source = FASTDEPLOY_SRC if engine == "fastdeploy" else VLLM_SRC
    python_paths = [str(SUBMISSION_ROOT), str(source)]
    python_paths.extend(
        item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item
    )
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_paths)),
            "LD_LIBRARY_PATH": python_library_path(
                python, environment.get("LD_LIBRARY_PATH", "")
            ),
            "CUDA_VISIBLE_DEVICES": gpu_selector,
            "BTS_SOURCE_ROOT": str(SUBMISSION_ROOT),
            # Pin the race-free native emit path for every formal worker.
            BTS_VARIABLE_TOPK_ENV: "1",
            BTS_VARIABLE_TOPK_REQUIRE_ENV: "1",
            "OMP_NUM_THREADS": "1",
            RUN_TOKEN_ENV: run_token,
        }
    )
    if engine == "vllm":
        environment["PYTORCH_ALLOC_CONF"] = FORMAL_PYTORCH_ALLOC_CONF
        environment["PYTORCH_CUDA_ALLOC_CONF"] = FORMAL_PYTORCH_ALLOC_CONF
    if required_variable_topk_failure_marker is not None:
        environment[BTS_VARIABLE_TOPK_FAILURE_MARKER_ENV] = str(
            required_variable_topk_failure_marker.resolve()
        )
    return environment


def gpu_lease_main(args: argparse.Namespace) -> int:
    """Hold one idle CUDA context so the GPU never looks free between arms."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.worker_gpu:
        raise ExecutionError(
            "GPU lease CUDA_VISIBLE_DEVICES differs from its physical selector"
        )
    # Linux parent-death signal prevents an orphaned lease from retaining the
    # GPU if the controller is killed abruptly.  Re-check PPID after prctl to
    # close the small race where the parent exits immediately before the call.
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() == 1:
            raise ExecutionError("GPU lease controller disappeared during startup")
    except (AttributeError, OSError) as error:
        raise ExecutionError(f"cannot arm GPU lease parent-death signal: {error}") from error
    try:
        import torch
    except ImportError as error:
        raise ExecutionError("GPU lease interpreter does not provide torch") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExecutionError("GPU lease must see exactly one CUDA device")
    # One byte is enough to create a persistent CUDA context and a visible
    # nvidia-smi compute PID.  The context is intentionally idle and consumes
    # no benchmark compute; keeping the allocation tiny preserves the same
    # usable-memory conditions for every arm.
    marker = torch.empty(1, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    ready = {
        "schema_version": GPU_LEASE_SCHEMA_VERSION,
        "status": "ready",
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "logical_cuda_device": 0,
        "marker_bytes": int(marker.numel() * marker.element_size()),
        "device_name": torch.cuda.get_device_name(0),
        "ready_at": utc_now(),
    }
    print(json.dumps(ready, sort_keys=True), flush=True)
    while True:
        signal.pause()


def _stop_gpu_lease(lease: Mapping[str, Any]) -> dict[str, Any]:
    process = lease["process"]
    token = lease["token"]
    cleanup = _terminate_owned_processes(process, token)
    if cleanup["remaining_owned_pids"]:
        raise ExecutionError(
            "persistent GPU lease left owned processes alive: "
            f"{cleanup['remaining_owned_pids']}"
        )
    stream = process.stdout
    trailing = "" if stream is None or stream.closed else stream.read()
    if stream is not None and not stream.closed:
        stream.close()
    remaining_token_pids = sorted(pids_with_run_token(token))
    if remaining_token_pids:
        raise ExecutionError(
            "persistent GPU lease token remains on processes: "
            f"{remaining_token_pids}"
        )
    return {
        "released_at": utc_now(),
        "returncode": process.returncode,
        "trailing_log": trailing,
        "remaining_token_pids": remaining_token_pids,
        "owned_process_cleanup": cleanup,
    }


def _ensure_attempt_worker_cleanup(
    error: BaseException,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> BaseException | None:
    """Keep the lease/locks while retrying cleanup of an owned failed worker."""

    evidence = getattr(error, "evidence", {})
    cleanup = evidence.get("owned_worker_cleanup")
    if not hasattr(error, "process") or not hasattr(error, "token"):
        # A prelaunch contamination has no worker to clean up.
        return None

    pending_control_flow: BaseException | None = None

    def remember_control_flow(candidate: BaseException) -> None:
        nonlocal pending_control_flow
        if (
            pending_control_flow is None
            and isinstance(candidate, (KeyboardInterrupt, SystemExit))
        ):
            pending_control_flow = candidate

    def owned_processes_are_gone() -> tuple[bool, list[int]]:
        try:
            error.process.poll()
        except (AttributeError, ChildProcessError):
            pass
        token_pids = sorted(pids_with_run_token(error.token))
        return error.process.returncode is not None and not token_pids, token_pids

    if isinstance(cleanup, Mapping) and not cleanup.get("remaining_owned_pids"):
        try:
            gone, _ = owned_processes_are_gone()
        except BaseException as postcheck_error:
            remember_control_flow(postcheck_error)
        else:
            if cleanup.get("status") != "cleanup_error" and gone:
                return pending_control_flow
    attempts: list[dict[str, Any]] = []
    while True:
        try:
            result = _terminate_owned_processes(error.process, error.token)
        except BaseException as cleanup_error:
            remember_control_flow(cleanup_error)
            result = {
                "status": "cleanup_error",
                "attempted_at": utc_now(),
                "error_type": type(cleanup_error).__name__,
                "error": str(cleanup_error),
                "remaining_owned_pids": [error.process.pid],
            }
        try:
            gone, token_pids = owned_processes_are_gone()
        except BaseException as postcheck_error:
            remember_control_flow(postcheck_error)
            gone = False
            token_pids = []
            result.update(
                status="cleanup_postcheck_error",
                postcheck_error_type=type(postcheck_error).__name__,
                postcheck_error=str(postcheck_error),
            )
            result.setdefault("remaining_owned_pids", [error.process.pid])
        result["postcheck_launcher_returncode"] = error.process.returncode
        result["postcheck_token_pids"] = token_pids
        if len(attempts) < 1000:
            attempts.append(dict(result))
        manifest["owned_worker_cleanup_recovery"] = {
            "status": (
                "completed"
                if not result.get("remaining_owned_pids") and gone
                else "waiting"
            ),
            "attempt_count": len(attempts),
            "latest": result,
            "attempts": attempts,
        }
        manifest_error = _best_effort_cleanup_manifest_write(
            manifest_path, manifest, context="owned worker cleanup"
        )
        if manifest_error is not None:
            remember_control_flow(manifest_error)
        if not result.get("remaining_owned_pids") and gone:
            return pending_control_flow
        try:
            print(
                "owned benchmark worker is still alive; retaining GPU lease and "
                "locks while cleanup retries: "
                f"remaining={result['remaining_owned_pids']}, token_pids={token_pids}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(5.0)
        except BaseException as pause_error:
            # Cleanup is an uninterruptible safety boundary.  Remember the first
            # control-flow exception and re-raise it only after all owned
            # processes are proven gone and the controller can unwind safely.
            remember_control_flow(pause_error)


def _best_effort_cleanup_manifest_write(
    path: Path, manifest: Mapping[str, Any], *, context: str
) -> BaseException | None:
    """Evidence I/O must never interrupt identity-safe process cleanup."""

    try:
        atomic_write_json(path, manifest)
        return None
    except BaseException as error:
        try:
            print(
                f"cannot persist {context} evidence; cleanup continues: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
        except BaseException:
            pass
        return error


def _ensure_gpu_lease_stopped(
    lease: Mapping[str, Any],
    *,
    gpu: Mapping[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[dict[str, Any], BaseException | None]:
    """Do not release formal locks while any lease-token process remains."""

    attempts: list[dict[str, Any]] = []
    pending_control_flow: BaseException | None = None

    def remember_control_flow(candidate: BaseException) -> None:
        nonlocal pending_control_flow
        if (
            pending_control_flow is None
            and isinstance(candidate, (KeyboardInterrupt, SystemExit))
        ):
            pending_control_flow = candidate

    while True:
        try:
            release = _stop_gpu_lease(lease)
        except BaseException as error:
            remember_control_flow(error)
            process = lease["process"]
            try:
                process.poll()
                token_pids = sorted(pids_with_run_token(lease["token"]))
                launcher_alive = process.returncode is None
                postcheck_error: BaseException | None = None
            except BaseException as check_error:
                remember_control_flow(check_error)
                # An inconclusive postcheck is never permission to unlock.
                token_pids = []
                launcher_alive = True
                postcheck_error = check_error
            record = {
                "attempted_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "launcher_alive": launcher_alive,
                "remaining_token_pids": token_pids,
            }
            if postcheck_error is not None:
                record.update(
                    postcheck_error_type=type(postcheck_error).__name__,
                    postcheck_error=str(postcheck_error),
                )
            if len(attempts) < 1000:
                attempts.append(record)
            manifest["gpu_lease_cleanup_recovery"] = {
                "status": "waiting" if launcher_alive or token_pids else "failed",
                "attempt_count": len(attempts),
                "latest": record,
                "attempts": attempts,
            }
            manifest_error = _best_effort_cleanup_manifest_write(
                manifest_path, manifest, context="GPU lease cleanup"
            )
            if manifest_error is not None:
                remember_control_flow(manifest_error)
            if not launcher_alive and not token_pids:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    # The interrupt may have landed after termination but before
                    # release accounting completed.  The lease is proven gone;
                    # synthesize bounded evidence and defer the control flow
                    # until after the formal locks are released.
                    release = {
                        "released_at": utc_now(),
                        "returncode": process.returncode,
                        "trailing_log": "",
                        "remaining_token_pids": [],
                        "owned_process_cleanup": {
                            "status": "release_accounting_interrupted",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "remaining_owned_pids": [],
                        },
                    }
                    break
                # The lease is definitely gone, so unlocking is safe, but an
                # unrelated release-accounting error must still fail the run.
                raise
            try:
                print(
                    "GPU lease cleanup is incomplete; retaining formal locks and "
                    f"retrying: launcher_alive={launcher_alive}, token_pids={token_pids}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(5.0)
            except BaseException as pause_error:
                remember_control_flow(pause_error)
            continue
        break
    while True:
        try:
            observed = gpu_processes(gpu)
        except (KeyboardInterrupt, SystemExit) as poll_interrupt:
            remember_control_flow(poll_interrupt)
            continue
        if lease["process"].pid not in {
            int(row["pid"]) for row in observed
        }:
            return release, pending_control_flow
        record = {
            "attempted_at": utc_now(),
            "status": "waiting_for_gpu_context_removal",
            "lease_pid": lease["process"].pid,
            "observed_gpu_processes": observed,
        }
        if len(attempts) < 1000:
            attempts.append(record)
        manifest["gpu_lease_cleanup_recovery"] = {
            "status": "waiting",
            "attempt_count": len(attempts),
            "latest": record,
            "attempts": attempts,
        }
        manifest_error = _best_effort_cleanup_manifest_write(
            manifest_path, manifest, context="GPU context removal"
        )
        if manifest_error is not None:
            remember_control_flow(manifest_error)
        while True:
            try:
                time.sleep(1.0)
                break
            except BaseException as pause_error:
                remember_control_flow(pause_error)


def _gpu_lease_record(lease: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": GPU_LEASE_SCHEMA_VERSION,
        "status": "active",
        "pid": lease["process"].pid,
        "run_token_sha256": hashlib.sha256(
            lease["token"].encode("utf-8")
        ).hexdigest(),
        "command": list(lease["command"]),
        "started_at": lease["started_at"],
        "acquired_at": lease["acquired_at"],
        "ready": dict(lease["ready"]),
        "observed_process": dict(lease["observed_process"]),
        "startup_log": list(lease["startup_log"]),
        "semantics": (
            "idle one-byte CUDA marker retained continuously across all fresh "
            "benchmark workers"
        ),
    }


def _start_gpu_lease_once(
    *, gpu: Mapping[str, Any], python: Path
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    occupied = gpu_processes(gpu)
    if occupied:
        return None, occupied
    token = secrets.token_hex(16)
    selector = str(gpu["index"])
    environment = worker_environment(
        python=python,
        engine="vllm",
        gpu_selector=selector,
        run_token=token,
    )
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "--internal-gpu-lease",
        "--worker-gpu",
        selector,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        text=True,
        start_new_session=True,
        bufsize=1,
    )
    lease: dict[str, Any] = {
        "process": process,
        "token": token,
        "command": command,
        "started_at": utc_now(),
    }
    lines: list[str] = []
    deadline = time.monotonic() + 120.0
    try:
        ready_record: Mapping[str, Any] | None = None
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                raise ExecutionError(
                    "GPU lease exited before readiness with status "
                    f"{returncode}: {' | '.join(lines[-20:])}"
                )
            assert process.stdout is not None
            readable, _, _ = select.select([process.stdout], [], [], 0.25)
            if not readable:
                continue
            line = process.stdout.readline()
            if not line:
                continue
            lines.append(line.rstrip("\n"))
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(candidate, dict)
                and candidate.get("schema_version") == GPU_LEASE_SCHEMA_VERSION
                and candidate.get("status") == "ready"
            ):
                ready_record = candidate
                break
        if ready_record is None:
            raise ExecutionError(
                "GPU lease did not become ready within 120 seconds: "
                + " | ".join(lines[-20:])
            )
        if ready_record.get("pid") != process.pid:
            raise ExecutionError("GPU lease readiness PID does not match its process")
        observed = gpu_processes(gpu)
        lease_rows = [row for row in observed if row["pid"] == process.pid]
        outsiders = [row for row in observed if row["pid"] != process.pid]
        if outsiders:
            # Another job won the acquisition race.  Release our marker and
            # wait for a genuinely exclusive opportunity instead of mixing.
            lease["startup_log"] = lines
            _stop_gpu_lease(lease)
            return None, outsiders
        if len(lease_rows) != 1:
            raise ExecutionError(
                "GPU lease reported ready but nvidia-smi did not observe it"
            )
        lease.update(
            ready=dict(ready_record),
            startup_log=lines,
            observed_process=lease_rows[0],
            acquired_at=utc_now(),
        )
        return lease, []
    except BaseException:
        if process.poll() is None:
            _stop_gpu_lease(lease)
        raise


def acquire_gpu_lease(
    *,
    gpu: Mapping[str, Any],
    python: Path,
    wait_seconds: float,
    poll_seconds: float,
    idle_stability_seconds: float,
) -> dict[str, Any]:
    """Wait for stable idleness, then atomically-ish establish a CUDA marker."""

    deadline = time.monotonic() + wait_seconds
    last_pids: tuple[int, ...] | None = None
    idle_since: float | None = None
    while True:
        occupied = gpu_processes(gpu)
        now = time.monotonic()
        if occupied:
            idle_since = None
        else:
            if idle_since is None:
                idle_since = now
                print(
                    f"GPU {gpu['index']} entered candidate-idle state; requiring "
                    f"{idle_stability_seconds:g}s continuous idleness",
                    file=sys.stderr,
                    flush=True,
                )
            if now - idle_since >= idle_stability_seconds:
                lease, race_occupants = _start_gpu_lease_once(
                    gpu=gpu, python=python
                )
                if lease is not None:
                    print(
                        f"acquired persistent GPU lease on index {gpu['index']} "
                        f"with PID {lease['process'].pid}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return lease
                occupied = race_occupants
                idle_since = None
        pids = tuple(sorted(int(row["pid"]) for row in occupied))
        if pids != last_pids:
            if occupied:
                print(
                    f"waiting for exclusive GPU {gpu['index']}; occupants={occupied}",
                    file=sys.stderr,
                    flush=True,
                )
            last_pids = pids
        remaining = deadline - now
        if wait_seconds <= 0 or remaining <= 0:
            raise ExecutionError(
                f"timed out waiting for exclusive GPU {gpu['index']}; "
                f"occupants={occupied}"
            )
        time.sleep(min(poll_seconds, remaining))


def wait_for_lease_only_stable(
    *,
    gpu: Mapping[str, Any],
    lease: Mapping[str, Any],
    poll_seconds: float,
    stability_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Keep the lease alive while unrelated GPU work drains, then re-stabilize."""

    process = lease["process"]
    lease_pid = int(process.pid)
    started_at = utc_now()
    deadline = (
        None if timeout_seconds == 0 else time.monotonic() + timeout_seconds
    )
    stable_since: float | None = None
    checks = 0
    transitions: list[dict[str, Any]] = []
    last_state: tuple[tuple[int, Any], ...] | None = None
    while True:
        if process.poll() is not None:
            raise ExecutionError(
                "persistent GPU lease exited while waiting for contamination to clear"
            )
        observed = gpu_processes(gpu)
        checks += 1
        observed_pids = {int(row["pid"]) for row in observed}
        if lease_pid not in observed_pids:
            raise ExecutionError(
                "persistent GPU lease disappeared while waiting for contamination "
                f"to clear; lease_pid={lease_pid}, observed={observed}"
            )
        outsiders = [dict(row) for row in observed if int(row["pid"]) != lease_pid]
        state = tuple(
            sorted(
                (
                    int(row["pid"]),
                    row.get("used_gpu_memory_mib"),
                )
                for row in outsiders
            )
        )
        now = time.monotonic()
        if state != last_state:
            transition = {
                "observed_at": utc_now(),
                "outsiders": outsiders,
            }
            if len(transitions) < 1000:
                transitions.append(transition)
            print(
                f"GPU {gpu['index']} recovery state changed; outsiders={outsiders}",
                file=sys.stderr,
                flush=True,
            )
            last_state = state
        if outsiders:
            stable_since = None
        else:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stability_seconds:
                return {
                    "schema_version": "bts_three_way_gpu_recovery_v1",
                    "status": "lease_only_stable",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "lease_pid": lease_pid,
                    "check_count": checks,
                    "required_stability_seconds": stability_seconds,
                    "timeout_seconds": timeout_seconds,
                    "transitions": transitions,
                    "transition_limit_reached": len(transitions) >= 1000,
                }
        if deadline is not None and now >= deadline:
            raise ExecutionError(
                "timed out waiting for GPU contamination to clear while retaining "
                f"lease PID {lease_pid}; outsiders={outsiders}"
            )
        sleep_for = poll_seconds
        if deadline is not None:
            sleep_for = min(sleep_for, max(0.0, deadline - now))
        time.sleep(sleep_for)


def _required_variable_topk_marker_error(
    environment: Mapping[str, str],
) -> NonRetryableWorkerExit | None:
    """Read the atomic marker emitted by a spawned FastDeploy GPU worker."""

    raw_path = environment.get(BTS_VARIABLE_TOPK_FAILURE_MARKER_ENV)
    if not raw_path:
        return None
    path = Path(raw_path)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return None
    except OSError as error:
        return NonRetryableWorkerExit(
            f"cannot inspect required variable Top-K failure marker {path}: {error}"
        )
    if not 0 < size <= 4096:
        return NonRetryableWorkerExit(
            "required variable Top-K failure marker has an invalid size: "
            f"{path} ({size} bytes)"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return NonRetryableWorkerExit(
            f"cannot read required variable Top-K failure marker {path}: {error}"
        )
    if not text.endswith("\n"):
        return NonRetryableWorkerExit(
            f"required variable Top-K failure marker is malformed: {path}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return NonRetryableWorkerExit(
            f"required variable Top-K failure marker is malformed: {path}"
        )
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "error", "pid"}
        or payload.get("schema_version")
        != BTS_VARIABLE_TOPK_FAILURE_MARKER_SCHEMA
        or not isinstance(payload.get("error"), str)
        or not isinstance(payload.get("pid"), int)
        or isinstance(payload.get("pid"), bool)
        or payload["pid"] <= 0
    ):
        return NonRetryableWorkerExit(
            f"required variable Top-K failure marker has the wrong contract: {path}"
        )
    message = payload["error"]
    if not message.startswith(BTS_VARIABLE_TOPK_REQUIRED_ERROR_PREFIX):
        return NonRetryableWorkerExit(
            f"required variable Top-K failure marker has the wrong contract: {path}"
        )
    return NonRetryableWorkerExit(message)


def run_worker_subprocess(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    gpu: Mapping[str, Any],
    lease_pid: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    token = environment[RUN_TOKEN_ENV]
    started_at = utc_now()
    occupied = gpu_processes(gpu)
    occupied_pids = {int(row["pid"]) for row in occupied}
    if lease_pid not in occupied_pids:
        raise ExecutionError(
            "persistent GPU lease disappeared before worker launch; "
            f"lease_pid={lease_pid}, observed={occupied}"
        )
    if occupied_pids != {lease_pid}:
        first_observed = occupied
        time.sleep(0.25)
        confirmed = gpu_processes(gpu)
        confirmed_pids = {int(row["pid"]) for row in confirmed}
        if lease_pid not in confirmed_pids:
            raise ExecutionError(
                "persistent GPU lease disappeared before worker launch; "
                f"lease_pid={lease_pid}, observed={confirmed}"
            )
        if confirmed_pids != {lease_pid}:
            raise external_gpu_process_error(
                stage="prelaunch",
                gpu=gpu,
                lease_pid=lease_pid,
                worker_pid=None,
                token=token,
                first_observed=first_observed,
                confirmed_observed=confirmed,
                allowed_pids={lease_pid},
                check_count=2,
                started_at=started_at,
                worker_gpu_pids=set(),
            )
    process: subprocess.Popen[Any] | None = None
    checks = 0
    worker_gpu_pids: set[int] = set()
    worker_gpu_missing_since: float | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(environment),
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout_seconds
            while True:
                marker_error = _required_variable_topk_marker_error(environment)
                if marker_error is not None:
                    raise marker_error
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                allowed = {
                    lease_pid,
                    process.pid,
                    *descendant_pids(process.pid),
                    *pids_with_run_token(token),
                }
                observed = gpu_processes(gpu)
                checks += 1
                if lease_pid not in {row["pid"] for row in observed}:
                    raise ExecutionError(
                        "persistent GPU lease disappeared while an arm was active"
                    )
                outsiders = [row for row in observed if row["pid"] not in allowed]
                if outsiders:
                    first_observed = observed
                    time.sleep(0.25)
                    allowed.update(descendant_pids(process.pid))
                    allowed.update(pids_with_run_token(token))
                    observed = gpu_processes(gpu)
                    checks += 1
                    if lease_pid not in {row["pid"] for row in observed}:
                        raise ExecutionError(
                            "persistent GPU lease disappeared while an arm was active"
                        )
                    outsiders = [row for row in observed if row["pid"] not in allowed]
                if outsiders:
                    raise external_gpu_process_error(
                        stage="running",
                        gpu=gpu,
                        lease_pid=lease_pid,
                        worker_pid=process.pid,
                        token=token,
                        first_observed=first_observed,
                        confirmed_observed=observed,
                        allowed_pids=allowed,
                        check_count=checks,
                        started_at=started_at,
                        worker_gpu_pids=worker_gpu_pids,
                    )
                marker_error = _required_variable_topk_marker_error(environment)
                if marker_error is not None:
                    raise marker_error
                owned_gpu_pids = {
                    int(row["pid"])
                    for row in observed
                    if row["pid"] in allowed and row["pid"] != lease_pid
                }
                if owned_gpu_pids:
                    worker_gpu_pids.update(owned_gpu_pids)
                    worker_gpu_missing_since = None
                elif worker_gpu_pids:
                    now = time.monotonic()
                    if worker_gpu_missing_since is None:
                        worker_gpu_missing_since = now
                    elif (
                        now - worker_gpu_missing_since
                        >= GPU_WORKER_DISAPPEAR_GRACE_SECONDS
                    ):
                        raise ExecutionError(
                            "owned GPU worker disappeared while its launcher remained "
                            "alive; the engine worker likely crashed or OOMed"
                        )
                try:
                    returncode = process.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    continue
        marker_error = _required_variable_topk_marker_error(environment)
        if marker_error is not None:
            raise marker_error
        if returncode != 0:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                    -40:
                ]
            )
            failure_type = (
                NonRetryableWorkerExit if returncode in {2, 4} else ExecutionError
            )
            raise failure_type(
                f"worker exited with status {returncode}; log tail:\n{tail}"
            )
        lingering = pids_with_run_token(token) - {process.pid}
        deadline = time.monotonic() + 60.0
        while lingering and time.monotonic() < deadline:
            time.sleep(0.5)
            lingering = pids_with_run_token(token) - {process.pid}
        if lingering:
            raise ExecutionError(
                f"worker left token-owned subprocesses alive: {sorted(lingering)}"
            )
        final_processes = gpu_processes(gpu)
        final_pids = {int(row["pid"]) for row in final_processes}
        if lease_pid not in final_pids:
            raise ExecutionError(
                "persistent GPU lease disappeared after worker teardown; "
                f"lease_pid={lease_pid}, observed={final_processes}"
            )
        if final_pids != {lease_pid}:
            first_observed = final_processes
            time.sleep(0.25)
            confirmed = gpu_processes(gpu)
            confirmed_pids = {int(row["pid"]) for row in confirmed}
            if lease_pid not in confirmed_pids:
                raise ExecutionError(
                    "persistent GPU lease disappeared after worker teardown; "
                    f"lease_pid={lease_pid}, observed={confirmed}"
                )
            if confirmed_pids != {lease_pid}:
                owned_after_exit = {process.pid, *pids_with_run_token(token)}
                outsiders = [
                    row
                    for row in confirmed
                    if int(row["pid"]) not in {lease_pid, *owned_after_exit}
                ]
                if outsiders:
                    raise external_gpu_process_error(
                        stage="post_worker",
                        gpu=gpu,
                        lease_pid=lease_pid,
                        worker_pid=process.pid,
                        token=token,
                        first_observed=first_observed,
                        confirmed_observed=confirmed,
                        allowed_pids={lease_pid, *owned_after_exit},
                        check_count=checks + 2,
                        started_at=started_at,
                        worker_gpu_pids=worker_gpu_pids,
                    )
                raise ExecutionError(
                    "GPU retained token-owned worker processes after teardown: "
                    f"{confirmed}"
                )
        if not worker_gpu_pids:
            raise ExecutionError("GPU monitor never observed an owned compute process")
        return {
            "mode": "sampled_compute_process_monitor_with_persistent_lease",
            "started_at": started_at,
            "finished_at": utc_now(),
            "check_count": checks,
            "all_checks_exclusive": True,
            "persistent_lease_pid": lease_pid,
            "lease_alive_after_worker": True,
            "observed_worker_gpu_pids": sorted(worker_gpu_pids),
            "post_worker_gpu_state": "lease_only",
        }
    except BaseException as error:
        if not isinstance(
            error,
            (
                ExternalGpuProcessDetected,
                NonRetryableWorkerExit,
                KeyboardInterrupt,
                SystemExit,
            ),
        ):
            marker_error = _required_variable_topk_marker_error(environment)
            if marker_error is not None:
                error = marker_error
        cleanup: dict[str, Any] | None = None
        if process is not None:
            try:
                cleanup = _terminate_owned_processes(process, token)
            except BaseException as cleanup_error:
                cleanup = {
                    "status": "cleanup_error",
                    "error_type": type(cleanup_error).__name__,
                    "error": str(cleanup_error),
                }
        if isinstance(error, ExternalGpuProcessDetected):
            error.evidence["owned_worker_cleanup"] = cleanup
            if process is not None:
                error.process = process
                error.token = token
            raise
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            if process is not None:
                # Preserve the control-flow exception exactly, including a
                # SystemExit status, but retain private handles so the
                # controller can finish identity-verified cleanup before it
                # ever releases the persistent GPU lease.
                error.evidence = {
                    "schema_version": "bts_three_way_failed_attempt_v1",
                    "status": "controller_interrupted",
                    "failed_at": utc_now(),
                    "started_at": started_at,
                    "failure_type": type(error).__name__,
                    "failure": str(error),
                    "observation_eligible": False,
                    "owned_worker_cleanup": cleanup,
                }
                error.process = process
                error.token = token
            raise
        if process is not None:
            evidence = {
                "schema_version": "bts_three_way_failed_attempt_v1",
                "status": (
                    "non_retryable_failure"
                    if isinstance(error, NonRetryableWorkerExit)
                    else "worker_failed"
                ),
                "failed_at": utc_now(),
                "started_at": started_at,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "observation_eligible": False,
                "owned_worker_cleanup": cleanup,
            }
            wrapped_failure = (
                NonRetryableWorkerAttemptFailed
                if isinstance(error, NonRetryableWorkerExit)
                else WorkerAttemptFailed
            )
            raise wrapped_failure(
                f"{type(error).__name__}: {error}",
                evidence=evidence,
                process=process,
                token=token,
            ) from error
        raise


def parse_vllm_fa2_log(
    path: Path, *, expected_measured_pids: set[int] | None = None
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    versions = [int(value) for value in _FA_VERSION_PATTERN.findall(text)]
    if not versions:
        raise ExecutionError(
            "vLLM model worker log did not report its resolved FlashAttention version"
        )
    if set(versions) != {2}:
        raise ExecutionError(
            f"vLLM resolved a non-FA2 backend: observed versions={versions}"
        )
    adapter_calls: list[dict[str, Any]] = []
    invalid_markers: list[str] = []
    for line in text.splitlines():
        marker_at = line.find(VLLM_FA2_LOG_MARKER_PREFIX)
        if marker_at < 0:
            continue
        payload = line[marker_at + len(VLLM_FA2_LOG_MARKER_PREFIX) :]
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            invalid_markers.append(line)
            continue
        if not isinstance(parsed, dict):
            invalid_markers.append(line)
            continue
        adapter_calls.append(parsed)
    if invalid_markers:
        raise ExecutionError(
            "vLLM emitted malformed FA2 adapter markers: "
            f"count={len(invalid_markers)}"
        )
    if not adapter_calls:
        raise ExecutionError(
            "vLLM log did not prove that the submitted CUDA-12 FA2 adapter "
            "executed"
        )
    required_adapter_call = {
        "schema_version": VLLM_FA2_MARKER_SCHEMA_VERSION,
        "adapter": VLLM_FA2_ADAPTER_NAME,
        "vllm_version": SUPPORTED_VLLM_VERSION,
        "flash_attn_version": SUPPORTED_FLASH_ATTN_VERSION,
        "upstream_argument_count": 21,
        "page_size": VLLM_KV_CACHE_BLOCK_SIZE,
        "upstream_extension_distribution": "flash-attn",
        "upstream_extension_in_distribution": True,
    }
    mismatches = [
        {
            key: {"expected": expected, "observed": row.get(key)}
            for key, expected in required_adapter_call.items()
            if row.get(key) != expected
        }
        for row in adapter_calls
    ]
    mismatches = [row for row in mismatches if row]
    if mismatches:
        raise ExecutionError(
            "vLLM FA2 adapter marker violated the frozen runtime contract: "
            f"{mismatches}"
        )
    invalid_identity = []
    for row in adapter_calls:
        pid = row.get("pid")
        calls = row.get("cumulative_calls")
        paged_calls = row.get("cumulative_paged_calls")
        extension_file = row.get("upstream_extension_file")
        distribution_file = row.get("upstream_extension_distribution_file")
        extension_sha256 = row.get("upstream_extension_sha256")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(calls, bool)
            or not isinstance(calls, int)
            or calls <= 0
            or isinstance(paged_calls, bool)
            or not isinstance(paged_calls, int)
            or paged_calls <= 0
            or paged_calls > calls
            or not isinstance(extension_file, str)
            or not Path(extension_file).is_absolute()
            or not isinstance(distribution_file, str)
            or not distribution_file
            or not isinstance(extension_sha256, str)
            or _HEX_SHA256.fullmatch(extension_sha256) is None
        ):
            invalid_identity.append(row)
    if invalid_identity:
        raise ExecutionError(
            "vLLM FA2 adapter marker lacks valid binary/call provenance: "
            f"count={len(invalid_identity)}"
        )
    marker_pids = {int(row["pid"]) for row in adapter_calls}
    if expected_measured_pids is not None and marker_pids != expected_measured_pids:
        raise ExecutionError(
            "vLLM FA2 marker PIDs differ from timed mmap worker PIDs: "
            f"marker={sorted(marker_pids)}, measured={sorted(expected_measured_pids)}"
        )
    extension_identities = {
        (
            row["upstream_extension_file"],
            row["upstream_extension_distribution_file"],
            row["upstream_extension_sha256"],
        )
        for row in adapter_calls
    }
    if len(extension_identities) != 1:
        raise ExecutionError("vLLM workers resolved different upstream FA2 binaries")
    extension_file, distribution_file, extension_sha256 = next(
        iter(extension_identities)
    )
    return {
        "type": "model_worker_log",
        "pattern": _FA_VERSION_PATTERN.pattern,
        "observed_versions": versions,
        "resolved_backend": "flash_attention_2",
        "adapter_marker_prefix": VLLM_FA2_LOG_MARKER_PREFIX,
        "adapter_calls": adapter_calls,
        "adapter_contract": required_adapter_call,
        "measured_worker_pids": sorted(marker_pids),
        "upstream_extension_identity": {
            "distribution": "flash-attn",
            "distribution_file": distribution_file,
            "file": extension_file,
            "sha256": extension_sha256,
            "in_distribution": True,
        },
        "sha256": file_sha256(path),
    }


def _build_controller_spec(
    plan: Mapping[str, Any],
    *,
    shared_contract: Mapping[str, Any],
    source_manifests: Mapping[str, Mapping[str, Any]],
    gpu_selector: str,
) -> dict[str, Any]:
    return {
        "schema_version": WORKER_SPEC_SCHEMA_VERSION,
        "protocol": EXECUTION_PROTOCOL,
        "shared_contract": dict(shared_contract),
        "shared_contract_sha256": analysis.canonical_sha256(shared_contract),
        "workload": dict(plan["workload"]),
        "workload_sha256": analysis.canonical_sha256(plan["workload"]),
        "generator": dict(plan["generator"]),
        "model_contract": plan["identity"]["model_contract"],
        "source_manifest_sha256": {
            engine: manifest["sha256"]
            for engine, manifest in source_manifests.items()
        },
        "engine_configs": {
            engine: dict(config) for engine, config in plan["engine_configs"].items()
        },
        "engine_config_sha256": dict(plan["engine_config_sha256"]),
        "gpu_selector": gpu_selector,
        "gpu_memory_utilization": plan["fixed_contract"][
            "gpu_memory_utilization"
        ],
        "warmups": FORMAL_WARMUPS,
    }


def _runtime_from_observations(
    arm: str,
    observations: Sequence[Mapping[str, Any]],
    source_sha256: str,
) -> dict[str, Any]:
    versions = {row["engine_version"] for row in observations}
    if len(versions) != 1:
        raise ExecutionError(f"{arm} engine version changed across restarts")
    engine, mode = analysis.ARM_SPECS[arm]
    engine_config_hashes = {row["engine_config_sha256"] for row in observations}
    if len(engine_config_hashes) != 1:
        raise ExecutionError(f"{arm} engine config changed across restarts")
    return {
        "engine": engine,
        "engine_version": next(iter(versions)),
        "attention_mode": mode,
        "prefill_backend": "bts" if mode == "bts" else "flash_attention_2",
        "requested_dense_attention_backend": "flash_attention_2",
        "observed_dense_attention_backend": "flash_attention_2",
        "decode_backend": (
            "native_append_attention"
            if engine == "fastdeploy"
            else "native_vllm_flash_attention_2"
        ),
        "bts_prefill_enabled": mode == "bts",
        "bts_decode_enabled": False,
        "engine_config_sha256": next(iter(engine_config_hashes)),
        "source_manifest_sha256": source_sha256,
    }


def _restart_from_observation(
    observation: Mapping[str, Any], monitor: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        not monitor.get("all_checks_exclusive")
        or not monitor.get("lease_alive_after_worker")
        or monitor.get("post_worker_gpu_state") != "lease_only"
    ):
        raise ExecutionError("GPU monitor cannot support an exclusive restart claim")
    counters = observation["counters"]
    return {
        "restart": observation["restart"],
        "measured_seconds": observation["measured_seconds"],
        "completed_requests": observation["completed_requests"],
        "total_input_tokens": observation["total_input_tokens"],
        "total_output_tokens": observation["total_output_tokens"],
        "fresh_process": True,
        "gpu_exclusive": True,
        "request_order_sha256": observation["request_order_sha256"],
        "runtime_source_sha256": observation["source_manifest_sha256"],
        **counters,
    }


def run_arm_with_retries(
    *,
    args: argparse.Namespace,
    output: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    restart: int,
    arm: str,
    engine: str,
    python: Path,
    model: Path,
    spec: Mapping[str, Any],
    spec_sha256: str,
    worker_gpu_selector: str,
    gpu: Mapping[str, Any],
    lease: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    """Run one logical arm, retaining the lease across recoverable attempts."""

    arm_dir = output / f"restart-{restart}" / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    attempt = 0
    contamination_retries = 0
    worker_failure_retries = 0
    contamination_recovery_elapsed_seconds = 0.0
    worker_failure_recovery_elapsed_seconds = 0.0
    while True:
        attempt_dir = arm_dir / f"attempt-{attempt:03d}"
        attempt_dir.mkdir()
        observation_path = attempt_dir / "observation.json"
        log_path = attempt_dir / "worker.log"
        monitor_path = attempt_dir / "gpu-monitor.json"
        attempt_path = attempt_dir / "attempt.json"
        command = _worker_command(
            python=python,
            arm=arm,
            restart=restart,
            model=model,
            bts_config=args.bts_config.resolve(),
            controller_spec=output / "controller-spec.json",
            spec_sha256=spec_sha256,
            observation=observation_path,
            gpu=worker_gpu_selector,
        )
        atomic_write_json(attempt_dir / "command.json", {"command": command})
        token = secrets.token_hex(16)
        attempt_record: dict[str, Any] = {
            "schema_version": "bts_three_way_arm_attempt_v1",
            "restart": restart,
            "arm": arm,
            "engine": engine,
            "attempt": attempt,
            "status": "worker_running",
            "started_at": utc_now(),
            "persistent_lease_pid": lease["process"].pid,
            "run_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "command_path": str((attempt_dir / "command.json").relative_to(output)),
            "worker_log_path": str(log_path.relative_to(output)),
            "observation_path": str(observation_path.relative_to(output)),
            "gpu_monitor_path": str(monitor_path.relative_to(output)),
            "required_variable_topk_failure_marker_path": str(
                (
                    attempt_dir / BTS_VARIABLE_TOPK_FAILURE_MARKER_FILENAME
                ).relative_to(output)
            ),
            "observation_eligible": False,
        }
        atomic_write_json(attempt_path, attempt_record)
        manifest["active_arm"] = {
            "restart": restart,
            "arm": arm,
            "attempt": attempt,
            "phase": "worker_running",
            "started_at": attempt_record["started_at"],
            "persistent_lease_pid": lease["process"].pid,
            "attempt_path": str(attempt_path.relative_to(output)),
        }
        manifest["status"] = "running"
        atomic_write_json(manifest_path, manifest)
        environment = worker_environment(
            python=python,
            engine=engine,
            gpu_selector=worker_gpu_selector,
            run_token=token,
            required_variable_topk_failure_marker=(
                attempt_dir / BTS_VARIABLE_TOPK_FAILURE_MARKER_FILENAME
            ),
        )
        try:
            monitor = run_worker_subprocess(
                command,
                environment=environment,
                log_path=log_path,
                gpu=gpu,
                lease_pid=lease["process"].pid,
                timeout_seconds=args.worker_timeout_seconds,
            )
        except ExternalGpuProcessDetected as error:
            cleanup_control_flow = _ensure_attempt_worker_cleanup(
                error,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            atomic_write_json(monitor_path, error.evidence)
            attempt_record.update(
                status="contaminated",
                finished_at=utc_now(),
                contamination_stage=error.evidence["stage"],
                contamination_monitor_path=str(monitor_path.relative_to(output)),
                observation_eligible=False,
            )
            if observation_path.exists():
                attempt_record["discarded_observation_sha256"] = file_sha256(
                    observation_path
                )
            summary = {
                "restart": restart,
                "arm": arm,
                "attempt": attempt,
                "status": "contaminated",
                "attempt_path": str(attempt_path.relative_to(output)),
                "monitor_path": str(monitor_path.relative_to(output)),
                "detected_at": error.evidence["detected_at"],
                "stage": error.evidence["stage"],
                "outsiders": error.evidence["outsiders"],
                "contamination_retry_index": contamination_retries,
                "observation_eligible": False,
            }
            manifest["arm_attempts"].append(summary)
            atomic_write_json(attempt_path, attempt_record)
            if cleanup_control_flow is not None:
                raise cleanup_control_flow
            maximum = args.max_contamination_retries_per_arm
            if maximum > 0 and contamination_retries >= maximum:
                manifest["active_arm"].update(
                    phase="contamination_retry_exhausted",
                    latest_outsiders=error.evidence["outsiders"],
                )
                atomic_write_json(manifest_path, manifest)
                raise ExecutionError(
                    f"{arm} restart {restart} exhausted {maximum} contamination "
                    "retries while retaining the lease"
                ) from error
            recovery_timeout = args.gpu_recovery_timeout_seconds
            if recovery_timeout > 0:
                recovery_timeout -= contamination_recovery_elapsed_seconds
                if recovery_timeout <= 0:
                    raise ExecutionError(
                        f"{arm} restart {restart} exhausted its cumulative GPU "
                        "recovery timeout while retaining the lease"
                    ) from error
            manifest["status"] = "recovering_gpu_exclusivity"
            manifest["active_arm"].update(
                phase="waiting_for_lease_only",
                latest_outsiders=error.evidence["outsiders"],
            )
            atomic_write_json(manifest_path, manifest)
            recovery_started = time.monotonic()
            try:
                recovery = wait_for_lease_only_stable(
                    gpu=gpu,
                    lease=lease,
                    poll_seconds=args.gpu_poll_seconds,
                    stability_seconds=args.gpu_idle_stability_seconds,
                    timeout_seconds=recovery_timeout,
                )
            finally:
                contamination_recovery_elapsed_seconds += (
                    time.monotonic() - recovery_started
                )
            recovery_path = attempt_dir / "gpu-recovery.json"
            atomic_write_json(recovery_path, recovery)
            attempt_record.update(
                status="contaminated_recovered",
                recovery_path=str(recovery_path.relative_to(output)),
                recovered_at=recovery["finished_at"],
            )
            summary.update(
                status="contaminated_recovered",
                recovery_path=str(recovery_path.relative_to(output)),
                recovered_at=recovery["finished_at"],
            )
            atomic_write_json(attempt_path, attempt_record)
            manifest["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            contamination_retries += 1
            attempt += 1
            continue
        except (KeyboardInterrupt, SystemExit):
            raise
        except NonRetryableWorkerAttemptFailed as error:
            cleanup_control_flow = _ensure_attempt_worker_cleanup(
                error,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            evidence = error.evidence
            failure_type = str(evidence.get("failure_type", type(error).__name__))
            failure = str(evidence.get("failure", error))
            atomic_write_json(monitor_path, evidence)
            attempt_record.update(
                status="failed",
                finished_at=utc_now(),
                failure_type=failure_type,
                failure=failure,
                failure_evidence_path=str(monitor_path.relative_to(output)),
                observation_eligible=False,
            )
            if observation_path.exists():
                attempt_record["discarded_observation_sha256"] = file_sha256(
                    observation_path
                )
            atomic_write_json(attempt_path, attempt_record)
            manifest["arm_attempts"].append(
                {
                    "restart": restart,
                    "arm": arm,
                    "attempt": attempt,
                    "status": "failed",
                    "attempt_path": str(attempt_path.relative_to(output)),
                    "monitor_path": str(monitor_path.relative_to(output)),
                    "failed_at": attempt_record["finished_at"],
                    "failure_type": failure_type,
                    "observation_eligible": False,
                }
            )
            manifest["active_arm"].update(
                phase="non_retryable_failure",
                failure_type=failure_type,
                failure=failure,
            )
            atomic_write_json(manifest_path, manifest)
            if cleanup_control_flow is not None:
                raise cleanup_control_flow
            raise
        except WorkerAttemptFailed as error:
            # A failed launcher or engine worker is recoverable only after all
            # token-owned processes are proven gone.  Keep the same lease and
            # formal locks throughout cleanup, the stability wait, and retry.
            cleanup_control_flow = _ensure_attempt_worker_cleanup(
                error,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            evidence = error.evidence
            failure_type = str(evidence.get("failure_type", type(error).__name__))
            failure = str(evidence.get("failure", error))
            atomic_write_json(monitor_path, evidence)
            attempt_record.update(
                status="worker_failed",
                finished_at=utc_now(),
                failure_type=failure_type,
                failure=failure,
                failure_evidence_path=str(monitor_path.relative_to(output)),
                worker_failure_retry_index=worker_failure_retries,
                observation_eligible=False,
            )
            if observation_path.exists():
                attempt_record["discarded_observation_sha256"] = file_sha256(
                    observation_path
                )
            summary = {
                "restart": restart,
                "arm": arm,
                "attempt": attempt,
                "status": "worker_failed",
                "attempt_path": str(attempt_path.relative_to(output)),
                "monitor_path": str(monitor_path.relative_to(output)),
                "failed_at": attempt_record["finished_at"],
                "failure_type": failure_type,
                "worker_failure_retry_index": worker_failure_retries,
                "observation_eligible": False,
            }
            manifest["arm_attempts"].append(summary)
            atomic_write_json(attempt_path, attempt_record)
            if cleanup_control_flow is not None:
                raise cleanup_control_flow

            maximum = args.max_worker_failure_retries_per_arm
            if maximum > 0 and worker_failure_retries >= maximum:
                manifest["active_arm"].update(
                    phase="worker_failure_retry_exhausted",
                    failure_type=failure_type,
                    failure=failure,
                )
                atomic_write_json(manifest_path, manifest)
                raise ExecutionError(
                    f"{arm} restart {restart} exhausted {maximum} worker-failure "
                    "retries while retaining the lease"
                ) from error

            recovery_timeout = args.gpu_recovery_timeout_seconds
            if recovery_timeout > 0:
                recovery_timeout -= worker_failure_recovery_elapsed_seconds
                if recovery_timeout <= 0:
                    raise ExecutionError(
                        f"{arm} restart {restart} exhausted its cumulative GPU "
                        "recovery timeout while retaining the lease"
                    ) from error
            manifest["status"] = "recovering_worker_failure"
            manifest["active_arm"].update(
                phase="waiting_for_lease_only_after_worker_failure",
                failure_type=failure_type,
                failure=failure,
            )
            atomic_write_json(manifest_path, manifest)
            recovery_started = time.monotonic()
            try:
                recovery = wait_for_lease_only_stable(
                    gpu=gpu,
                    lease=lease,
                    poll_seconds=args.gpu_poll_seconds,
                    stability_seconds=args.gpu_idle_stability_seconds,
                    timeout_seconds=recovery_timeout,
                )
            finally:
                worker_failure_recovery_elapsed_seconds += (
                    time.monotonic() - recovery_started
                )
            recovery_path = attempt_dir / "gpu-recovery.json"
            atomic_write_json(recovery_path, recovery)
            attempt_record.update(
                status="worker_failed_recovered",
                recovery_path=str(recovery_path.relative_to(output)),
                recovered_at=recovery["finished_at"],
            )
            summary.update(
                status="worker_failed_recovered",
                recovery_path=str(recovery_path.relative_to(output)),
                recovered_at=recovery["finished_at"],
            )
            atomic_write_json(attempt_path, attempt_record)
            manifest["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            worker_failure_retries += 1
            attempt += 1
            continue
        except Exception as error:
            # Protocol/configuration/validation errors are deterministic and
            # must fail closed instead of being hidden by an infinite retry.
            evidence = {
                "schema_version": "bts_three_way_failed_attempt_v1",
                "status": "non_retryable_failure",
                "failed_at": utc_now(),
                "failure_type": type(error).__name__,
                "failure": str(error),
                "observation_eligible": False,
            }
            atomic_write_json(monitor_path, evidence)
            attempt_record.update(
                status="failed",
                finished_at=utc_now(),
                failure_type=type(error).__name__,
                failure=str(error),
                failure_evidence_path=str(monitor_path.relative_to(output)),
                observation_eligible=False,
            )
            if observation_path.exists():
                attempt_record["discarded_observation_sha256"] = file_sha256(
                    observation_path
                )
            atomic_write_json(attempt_path, attempt_record)
            manifest["arm_attempts"].append(
                {
                    "restart": restart,
                    "arm": arm,
                    "attempt": attempt,
                    "status": "failed",
                    "attempt_path": str(attempt_path.relative_to(output)),
                    "monitor_path": str(monitor_path.relative_to(output)),
                    "failed_at": attempt_record["finished_at"],
                    "failure_type": type(error).__name__,
                    "observation_eligible": False,
                }
            )
            manifest["active_arm"].update(
                phase="non_retryable_failure",
                failure_type=type(error).__name__,
                failure=str(error),
            )
            atomic_write_json(manifest_path, manifest)
            raise

        atomic_write_json(monitor_path, monitor)
        manifest["active_arm"]["phase"] = "validating"
        atomic_write_json(manifest_path, manifest)
        try:
            try:
                observation = json.loads(observation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ExecutionError(
                    f"cannot load worker observation {observation_path}: {error}"
                ) from error
            validate_worker_observation(
                observation,
                expected_arm=arm,
                expected_restart=restart,
                spec=spec,
            )
            if engine == "vllm":
                measured_pids = {
                    int(row["pid"])
                    for row in observation["backend_evidence"][
                        "fa2_adapter_calls"
                    ]["final_audits"]
                }
                log_evidence = parse_vllm_fa2_log(
                    log_path, expected_measured_pids=measured_pids
                )
                atomic_write_json(
                    attempt_dir / "vllm-fa2-log-evidence.json", log_evidence
                )
        except Exception as error:
            validation_failure_path = attempt_dir / "validation-failure.json"
            evidence = {
                "schema_version": "bts_three_way_failed_attempt_v1",
                "status": "non_retryable_failure",
                "stage": "post_worker_validation",
                "failed_at": utc_now(),
                "failure_type": type(error).__name__,
                "failure": str(error),
                "observation_eligible": False,
            }
            atomic_write_json(validation_failure_path, evidence)
            attempt_record.update(
                status="failed",
                finished_at=evidence["failed_at"],
                failure_type=type(error).__name__,
                failure=str(error),
                failure_evidence_path=str(
                    validation_failure_path.relative_to(output)
                ),
                observation_eligible=False,
            )
            if observation_path.exists():
                attempt_record["discarded_observation_sha256"] = file_sha256(
                    observation_path
                )
            atomic_write_json(attempt_path, attempt_record)
            manifest["arm_attempts"].append(
                {
                    "restart": restart,
                    "arm": arm,
                    "attempt": attempt,
                    "status": "failed",
                    "attempt_path": str(attempt_path.relative_to(output)),
                    "monitor_path": str(monitor_path.relative_to(output)),
                    "failure_evidence_path": str(
                        validation_failure_path.relative_to(output)
                    ),
                    "failed_at": evidence["failed_at"],
                    "failure_type": type(error).__name__,
                    "observation_eligible": False,
                }
            )
            manifest["active_arm"].update(
                phase="non_retryable_failure",
                failure_type=type(error).__name__,
                failure=str(error),
            )
            atomic_write_json(manifest_path, manifest)
            raise
        attempt_record.update(
            status="completed",
            finished_at=utc_now(),
            observation_eligible=True,
            observation_sha256=file_sha256(observation_path),
        )
        atomic_write_json(attempt_path, attempt_record)
        manifest["arm_attempts"].append(
            {
                "restart": restart,
                "arm": arm,
                "attempt": attempt,
                "status": "completed",
                "attempt_path": str(attempt_path.relative_to(output)),
                "monitor_path": str(monitor_path.relative_to(output)),
                "observation_path": str(observation_path.relative_to(output)),
                "completed_at": attempt_record["finished_at"],
                "observation_eligible": True,
            }
        )
        return observation, monitor, attempt


def _close_progress_display(display: Any | None) -> BaseException | None:
    """Stop optional terminal UI without letting it skip resource cleanup."""

    if display is None:
        return None
    try:
        display.close()
    except BaseException as error:
        return error
    return None


def execute_controller(args: argparse.Namespace) -> int:
    plan = build_plan(args, dry_run=False)
    gpu = resolve_gpu(args.gpu)
    # FastDeploy 2.5 parses CUDA_VISIBLE_DEVICES as an integer in one worker
    # path.  Resolve a user-supplied UUID once, retain it in provenance, and
    # pass the unambiguous physical index to both engines.
    worker_gpu_selector = str(gpu["index"])
    plan["requested_gpu_selector"] = args.gpu
    plan["worker_gpu_selector"] = worker_gpu_selector
    source_manifests = {
        engine: source_manifest(engine) for engine in ("fastdeploy", "vllm")
    }
    shared_contract = analysis.build_shared_contract(
        logical_model_sha256=plan["identity"]["logical_model_sha256"],
        tokenizer_sha256=plan["identity"]["tokenizer_sha256"],
        bts_config_sha256=plan["identity"]["bts_config_sha256"],
        gpu_uuid=gpu["uuid"],
    )
    spec = _build_controller_spec(
        plan,
        shared_contract=shared_contract,
        source_manifests=source_manifests,
        gpu_selector=worker_gpu_selector,
    )
    spec_sha256 = analysis.canonical_sha256(spec)
    for row in plan["commands"]:
        row["command"] = [
            spec_sha256 if value == "<controller-spec-sha256>" else value
            for value in row["command"]
        ]
        worker_gpu_index = row["command"].index("--worker-gpu") + 1
        row["command"][worker_gpu_index] = worker_gpu_selector
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ExecutionError(f"output directory already exists: {output}") from error
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "waiting_for_gpu",
        "started_at": utc_now(),
        "gpu": gpu,
        "gpu_wait_seconds": args.wait_for_gpu_seconds,
        "gpu_poll_seconds": args.gpu_poll_seconds,
        "gpu_idle_stability_seconds": args.gpu_idle_stability_seconds,
        "contamination_retry_policy": {
            "action": "discard_attempt_keep_lease_wait_and_retry",
            "max_retries_per_arm": args.max_contamination_retries_per_arm,
            "zero_max_retries_means_unlimited": True,
            "gpu_recovery_timeout_seconds": args.gpu_recovery_timeout_seconds,
            "zero_recovery_timeout_means_unlimited": True,
            "required_lease_only_stability_seconds": (
                args.gpu_idle_stability_seconds
            ),
        },
        "worker_failure_retry_policy": {
            "action": "discard_attempt_keep_lease_cleanup_wait_and_retry",
            "max_retries_per_arm": args.max_worker_failure_retries_per_arm,
            "zero_max_retries_means_unlimited": True,
            "gpu_recovery_timeout_seconds": args.gpu_recovery_timeout_seconds,
            "zero_recovery_timeout_means_unlimited": True,
            "required_lease_only_stability_seconds": (
                args.gpu_idle_stability_seconds
            ),
        },
        "plan": {key: value for key, value in plan.items() if key != "commands"},
        "controller_spec_sha256": spec_sha256,
        "source_manifests": source_manifests,
        "completed_arms": [],
        "arm_attempts": [],
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output / "controller-spec.json", spec)
    atomic_write_json(output / "plan.json", plan)

    observations: dict[str, list[Mapping[str, Any]]] = {
        arm: [] for arm in analysis.ARMS
    }
    monitors: dict[tuple[int, str], Mapping[str, Any]] = {}
    lease: dict[str, Any] | None = None
    formal_locks: list[Any] = []
    controller_error: BaseException | None = None
    progress_display: Any | None = None
    try:
        manifest["status"] = "waiting_for_repository_gpu_locks"
        atomic_write_json(manifest_path, manifest)
        formal_locks = acquire_formal_run_lock(
            gpu=gpu,
            wait_seconds=args.wait_for_gpu_seconds,
            poll_seconds=args.gpu_poll_seconds,
        )
        manifest["formal_run_locks"] = {
            "status": "held",
            "acquired_at": utc_now(),
            "paths": [str(path) for path in formal_run_lock_paths(gpu)],
        }
        manifest["status"] = "waiting_for_gpu"
        atomic_write_json(manifest_path, manifest)
        lease = acquire_gpu_lease(
            gpu=gpu,
            python=absolute_executable_path(args.vllm_python),
            wait_seconds=args.wait_for_gpu_seconds,
            poll_seconds=args.gpu_poll_seconds,
            idle_stability_seconds=args.gpu_idle_stability_seconds,
        )
        lease_record = _gpu_lease_record(lease)
        atomic_write_json(output / "gpu-lease.json", lease_record)
        manifest.update(status="running", gpu_lease=lease_record)
        atomic_write_json(manifest_path, manifest)
        try:
            from eval.ThreeWay.progress import ManifestProgress

            progress_display = ManifestProgress(
                manifest_path,
                total=analysis.FORMAL_RESTARTS * len(analysis.ARMS),
                restart_count=analysis.FORMAL_RESTARTS,
            )
            progress_display.start()
        except Exception as error:
            # Progress is presentation only.  It must never invalidate a timed
            # measurement or interfere with worker/lease cleanup.
            print(f"progress display unavailable: {error}", file=sys.stderr)

        # The wait can be long.  Revalidate every mutable input while the lease
        # is already holding the GPU, before any timed worker is launched.
        for engine, original in source_manifests.items():
            if source_manifest(engine)["sha256"] != original["sha256"]:
                raise ExecutionError(
                    f"{engine} source changed while waiting for the GPU"
                )
        acquired_identity = derive_cross_engine_identity(
            args.fastdeploy_model.resolve(),
            args.vllm_model.resolve(),
            plan["identity"]["model_contract"],
        )
        for field in ("logical_model_sha256", "tokenizer_sha256"):
            if acquired_identity[field] != plan["identity"][field]:
                raise ExecutionError(
                    f"cross-engine {field} changed while waiting for the GPU"
                )

        for restart in range(analysis.FORMAL_RESTARTS):
            for arm in rotated_arm_order(restart):
                engine = engine_for_arm(arm)
                python = (
                    args.fastdeploy_python
                    if engine == "fastdeploy"
                    else args.vllm_python
                )
                python = absolute_executable_path(python)
                model = (
                    args.fastdeploy_model
                    if engine == "fastdeploy"
                    else args.vllm_model
                ).resolve()
                observation, monitor, successful_attempt = run_arm_with_retries(
                    args=args,
                    output=output,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    restart=restart,
                    arm=arm,
                    engine=engine,
                    python=python,
                    model=model,
                    spec=spec,
                    spec_sha256=spec_sha256,
                    worker_gpu_selector=worker_gpu_selector,
                    gpu=gpu,
                    lease=lease,
                )
                monitors[(restart, arm)] = monitor
                observations[arm].append(observation)
                manifest["completed_arms"].append(
                    {
                        "restart": restart,
                        "arm": arm,
                        "successful_attempt": successful_attempt,
                        "attempt_count": successful_attempt + 1,
                        "completed_at": utc_now(),
                    }
                )
                manifest.pop("active_arm", None)
                atomic_write_json(manifest_path, manifest)

        for engine, original in source_manifests.items():
            current = source_manifest(engine)
            if current["sha256"] != original["sha256"]:
                raise ExecutionError(f"{engine} source changed during the benchmark")
        final_identity = derive_cross_engine_identity(
            args.fastdeploy_model.resolve(),
            args.vllm_model.resolve(),
            plan["identity"]["model_contract"],
        )
        for field in ("logical_model_sha256", "tokenizer_sha256"):
            if final_identity[field] != plan["identity"][field]:
                raise ExecutionError(
                    f"cross-engine {field} changed during the benchmark"
                )

        arm_reports = {}
        for arm in analysis.ARMS:
            engine = engine_for_arm(arm)
            rows = sorted(observations[arm], key=lambda row: row["restart"])
            runtime = _runtime_from_observations(
                arm, rows, source_manifests[engine]["sha256"]
            )
            restarts = [
                _restart_from_observation(
                    row, monitors[(int(row["restart"]), arm)]
                )
                for row in rows
            ]
            report = analysis.build_arm_report(
                arm=arm,
                shared_contract=shared_contract,
                workload=plan["workload"],
                runtime=runtime,
                restarts=restarts,
            )
            arm_reports[arm] = report
            atomic_write_json(output / f"{arm}.json", report)
        result = analysis.analyze_arm_reports(arm_reports)
        atomic_write_json(output / "three-comparison.json", result)
        manifest.update(
            status="completed",
            completed_at=utc_now(),
            analysis_status=result["overall_status"],
            analysis_path="three-comparison.json",
        )
        atomic_write_json(manifest_path, manifest)
        progress_error = _close_progress_display(progress_display)
        progress_display = None
        if isinstance(progress_error, (KeyboardInterrupt, SystemExit)):
            raise progress_error
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if result["overall_status"] == "PASS" else 2
    except BaseException as error:
        controller_error = error
        manifest.update(
            status="failed",
            failed_at=utc_now(),
            failure_type=type(error).__name__,
            failure=str(error),
        )
        try:
            atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        raise
    finally:
        cleanup_errors: list[str] = []
        cleanup_control_flow: BaseException | None = None

        def remember_cleanup_control_flow(candidate: BaseException) -> None:
            nonlocal cleanup_control_flow
            if (
                cleanup_control_flow is None
                and isinstance(candidate, (KeyboardInterrupt, SystemExit))
            ):
                cleanup_control_flow = candidate

        progress_error = _close_progress_display(progress_display)
        progress_display = None
        if progress_error is not None:
            # Ordinary UI failures are intentionally ignored.  A user control-flow
            # exception is deferred until every worker, lease, and lock is safe.
            remember_cleanup_control_flow(progress_error)

        if (
            controller_error is not None
            and hasattr(controller_error, "process")
            and hasattr(controller_error, "token")
        ):
            while True:
                try:
                    deferred = _ensure_attempt_worker_cleanup(
                        controller_error,
                        manifest=manifest,
                        manifest_path=manifest_path,
                    )
                    if deferred is not None:
                        remember_cleanup_control_flow(deferred)
                    break
                except BaseException as cleanup_error:
                    remember_cleanup_control_flow(cleanup_error)
                    manifest["owned_worker_cleanup_retry_error"] = {
                        "error_type": type(cleanup_error).__name__,
                        "error": str(cleanup_error),
                    }
                    manifest_error = _best_effort_cleanup_manifest_write(
                        manifest_path,
                        manifest,
                        context="controller worker cleanup retry",
                    )
                    if manifest_error is not None:
                        remember_cleanup_control_flow(manifest_error)
                    try:
                        time.sleep(1.0)
                    except BaseException as pause_error:
                        remember_cleanup_control_flow(pause_error)
        if lease is not None:
            lease_cleanup_confirmed = False
            while not lease_cleanup_confirmed:
                try:
                    release, deferred = _ensure_gpu_lease_stopped(
                        lease,
                        gpu=gpu,
                        manifest=manifest,
                        manifest_path=manifest_path,
                    )
                    if deferred is not None:
                        remember_cleanup_control_flow(deferred)
                    if release["remaining_token_pids"]:
                        raise ExecutionError(
                            "GPU lease token-owned processes remain after release: "
                            f"{release['remaining_token_pids']}"
                        )
                    post_release = gpu_processes(gpu)
                    if lease["process"].pid in {
                        int(row["pid"]) for row in post_release
                    }:
                        raise ExecutionError(
                            "GPU lease PID remains visible after release: "
                            f"{lease['process'].pid}"
                        )
                    lease_record = dict(
                        manifest.get("gpu_lease", _gpu_lease_record(lease))
                    )
                    lease_record.update(
                        status="released",
                        release=release,
                        post_release_gpu_processes=post_release,
                        gpu_idle_after_release=not post_release,
                    )
                    manifest["gpu_lease"] = lease_record
                    atomic_write_json(output / "gpu-lease.json", lease_record)
                    atomic_write_json(manifest_path, manifest)
                    lease_cleanup_confirmed = True
                except BaseException as release_error:
                    remember_cleanup_control_flow(release_error)
                    message = f"{type(release_error).__name__}: {release_error}"
                    manifest["gpu_lease_release_error"] = message
                    process = lease["process"]
                    try:
                        process.poll()
                        token_pids = sorted(pids_with_run_token(lease["token"]))
                        lease_is_gone = (
                            process.returncode is not None and not token_pids
                        )
                    except BaseException as postcheck_error:
                        remember_cleanup_control_flow(postcheck_error)
                        lease_is_gone = False
                        token_pids = []
                    if lease_is_gone:
                        cleanup_errors.append(
                            f"GPU lease release failed after termination: {message}"
                        )
                        lease_cleanup_confirmed = True
                    else:
                        manifest["gpu_lease_release_retry"] = {
                            "status": "waiting",
                            "remaining_token_pids": token_pids,
                            "error": message,
                        }
                    manifest_error = _best_effort_cleanup_manifest_write(
                        manifest_path, manifest, context="GPU lease release retry"
                    )
                    if manifest_error is not None:
                        remember_cleanup_control_flow(manifest_error)
                    if not lease_cleanup_confirmed:
                        try:
                            time.sleep(1.0)
                        except BaseException as pause_error:
                            remember_cleanup_control_flow(pause_error)
        # Formal locks are deliberately the final resource released, and only
        # after the lease and all token-owned processes are proven gone.
        if formal_locks:
            try:
                release_formal_run_lock(formal_locks)
                lock_record = dict(manifest.get("formal_run_locks", {}))
                lock_record.update(status="released", released_at=utc_now())
                manifest["formal_run_locks"] = lock_record
                atomic_write_json(manifest_path, manifest)
            except BaseException as lock_error:
                remember_cleanup_control_flow(lock_error)
                message = f"{type(lock_error).__name__}: {lock_error}"
                cleanup_errors.append(f"formal lock release failed: {message}")
                manifest["formal_run_lock_release_error"] = message
                try:
                    atomic_write_json(manifest_path, manifest)
                except Exception:
                    pass
        if cleanup_errors:
            manifest["cleanup_errors"] = cleanup_errors
            if controller_error is None and cleanup_control_flow is None:
                manifest.update(
                    status="failed",
                    failed_at=utc_now(),
                    failure_type="ControllerCleanupError",
                    failure="; ".join(cleanup_errors),
                )
            try:
                atomic_write_json(manifest_path, manifest)
            except Exception:
                pass
        if cleanup_control_flow is not None and controller_error is None:
            manifest.update(
                status="failed",
                failed_at=utc_now(),
                failure_type=type(cleanup_control_flow).__name__,
                failure=str(cleanup_control_flow),
            )
            _best_effort_cleanup_manifest_write(
                manifest_path, manifest, context="cleanup control flow"
            )
            raise cleanup_control_flow
        if cleanup_errors and controller_error is None:
            raise ExecutionError("; ".join(cleanup_errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastdeploy-python", type=Path)
    parser.add_argument("--vllm-python", type=Path)
    parser.add_argument("--fastdeploy-model", type=Path)
    parser.add_argument("--vllm-model", type=Path)
    parser.add_argument("--bts-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument(
        "--logical-model-sha256",
        help="optional expected digest; actual runs derive and verify it from both models",
    )
    parser.add_argument(
        "--tokenizer-sha256",
        help="optional expected digest; actual runs derive and verify it from both models",
    )
    parser.add_argument("--request-count", type=int, default=FORMAL_REQUEST_COUNT)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=FORMAL_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--worker-timeout-seconds", type=float, default=14400.0)
    parser.add_argument(
        "--wait-for-gpu-seconds",
        type=float,
        default=0.0,
        help="wait this long for the selected GPU, then retain a CUDA lease across all arms",
    )
    parser.add_argument(
        "--gpu-poll-seconds",
        type=float,
        default=5.0,
        help="poll interval while waiting for the selected GPU",
    )
    parser.add_argument(
        "--gpu-idle-stability-seconds",
        type=float,
        default=30.0,
        help="require this many continuously idle seconds before acquiring the lease",
    )
    parser.add_argument(
        "--max-contamination-retries-per-arm",
        type=int,
        default=0,
        help=(
            "retry a contaminated arm this many times while retaining the lease; "
            "0 means unlimited"
        ),
    )
    parser.add_argument(
        "--max-worker-failure-retries-per-arm",
        type=int,
        default=0,
        help=(
            "retry a crashed/failed worker this many times after identity-verified "
            "cleanup while retaining the lease; 0 means unlimited"
        ),
    )
    parser.add_argument(
        "--gpu-recovery-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "cumulative time allowed for one arm to recover from GPU contamination; "
            "0 means wait indefinitely while retaining the lease"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")

    # Hidden worker arguments.  The controller supplies every one explicitly.
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-gpu-lease", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--arm", choices=analysis.ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--restart", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-bts-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--controller-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--controller-spec-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-gpu", help=argparse.SUPPRESS)
    return parser


def _require_controller_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    required = (
        "fastdeploy_python",
        "vllm_python",
        "fastdeploy_model",
        "vllm_model",
        "bts_config",
        "output",
        "gpu",
    )
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error("controller requires " + ", ".join(missing))


def _require_worker_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    required = (
        "arm",
        "restart",
        "worker_model",
        "worker_bts_config",
        "controller_spec",
        "controller_spec_sha256",
        "worker_output",
        "worker_gpu",
    )
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error("internal worker requires " + ", ".join(missing))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    internal_mode = args.internal_gpu_lease or args.internal_worker
    try:
        if args.internal_gpu_lease:
            if args.worker_gpu is None:
                parser.error("internal GPU lease requires --worker-gpu")
            return gpu_lease_main(args)
        if args.internal_worker:
            _require_worker_args(parser, args)
            return worker_main(args)
        _require_controller_args(parser, args)
        if (
            not math.isfinite(args.worker_timeout_seconds)
            or args.worker_timeout_seconds <= 0
        ):
            raise ExecutionError("--worker-timeout-seconds must be positive")
        if (
            not math.isfinite(args.wait_for_gpu_seconds)
            or args.wait_for_gpu_seconds < 0
        ):
            raise ExecutionError("--wait-for-gpu-seconds must be nonnegative")
        if not math.isfinite(args.gpu_poll_seconds) or args.gpu_poll_seconds <= 0:
            raise ExecutionError("--gpu-poll-seconds must be positive")
        if (
            not math.isfinite(args.gpu_idle_stability_seconds)
            or args.gpu_idle_stability_seconds < 0
        ):
            raise ExecutionError(
                "--gpu-idle-stability-seconds must be nonnegative"
            )
        if args.max_contamination_retries_per_arm < 0:
            raise ExecutionError(
                "--max-contamination-retries-per-arm must be nonnegative"
            )
        if args.max_worker_failure_retries_per_arm < 0:
            raise ExecutionError(
                "--max-worker-failure-retries-per-arm must be nonnegative"
            )
        if (
            not math.isfinite(args.gpu_recovery_timeout_seconds)
            or args.gpu_recovery_timeout_seconds < 0
        ):
            raise ExecutionError(
                "--gpu-recovery-timeout-seconds must be nonnegative"
            )
        if args.dry_run:
            plan = build_plan(args, dry_run=True)
            print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
            controller_status = 0
        else:
            controller_status = execute_controller(args)
    except (ExecutionError, analysis.ProtocolError, OSError, subprocess.SubprocessError) as error:
        print(f"execution error: {error}", file=sys.stderr)
        if internal_mode:
            return 4
        controller_status = 4
    except KeyboardInterrupt:
        if internal_mode:
            raise
        print("execution interrupted", file=sys.stderr)
        controller_status = 130

    try:
        # Keep the machine-readable JSON on stdout ordered before the human
        # summary on stderr even when both streams are redirected to one log.
        sys.stdout.flush()
        print(file=sys.stderr)
        print(
            terminal_summary.render_run_summary(
                output_dir=args.output,
                controller_exit_code=controller_status,
                dry_run=args.dry_run,
            ),
            file=sys.stderr,
        )
    except (terminal_summary.SummaryError, analysis.ProtocolError) as error:
        print(f"terminal summary error: {error}", file=sys.stderr)
        return 4
    return controller_status


if __name__ == "__main__":
    raise SystemExit(main())
