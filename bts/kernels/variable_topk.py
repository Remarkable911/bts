"""Exact FP32 per-row-K selection for long block_count candidate rows.

The caller supplies unique candidate blocks, valid global row IDs and budgets
already bounded by the number of finite scores. Scores contain finite values
or -inf masks. Output is prefilled with kv_len; only each row's valid prefix is
written. The deterministic native entry point is enabled by default. It emits
larger-than-boundary items in candidate-index order and resolves boundary ties
by smaller candidate index. Set ``BTS_DETERMINISTIC_VARIABLE_TOPK=0`` only for
diagnostic comparison with the legacy cross-warp atomic emit path.

The tested SM90 path supports K up to 65536 and uses this project's independent
CUDA radix selector. Unsupported layouts/devices or an unavailable build
toolchain return False so the caller retains its existing selector. CUDA
execution errors propagate: they must not be hidden by a fallback.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

import torch
import triton
import triton.language as tl


# Keep this validated allocation bound in sync with variable_topk.cu.
# Radix selection sizes shared memory from N, not the selected output width K.
_MAX_TOPK = 65536
_DETERMINISTIC_ENV = "BTS_DETERMINISTIC_VARIABLE_TOPK"
_REQUIRE_DETERMINISTIC_ENV = "BTS_REQUIRE_DETERMINISTIC_VARIABLE_TOPK"
_REQUIRED_ERROR_PREFIX = "BTS required deterministic variable Top-K: "
_REQUIRED_ERROR_MARKER_ENV = "BTS_REQUIRED_VARIABLE_TOPK_FAILURE_MARKER"
_REQUIRED_ERROR_MARKER_SCHEMA = "bts_required_variable_topk_failure_v1"
_TRUE_ENV_VALUES = frozenset(("1", "true", "yes", "on"))


def deterministic_variable_topk_enabled():
    """Return whether the default-on deterministic native emit path is enabled."""
    raw = os.environ.get(_DETERMINISTIC_ENV)
    if raw is None:
        return True
    return raw.strip().lower() in _TRUE_ENV_VALUES


def deterministic_variable_topk_required():
    """Return whether unsupported/fallback selector routes must fail closed."""
    return (
        os.environ.get(_REQUIRE_DETERMINISTIC_ENV, "").strip().lower()
        in _TRUE_ENV_VALUES
    )


def _raise_required_error(reason):
    """Publish a cross-process failure marker, then fail the formal route."""

    message = _REQUIRED_ERROR_PREFIX + reason
    raw_marker = os.environ.get(_REQUIRED_ERROR_MARKER_ENV)
    temporary = None
    if raw_marker:
        marker = Path(raw_marker)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=marker.parent,
                prefix=f".{marker.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    {
                        "schema_version": _REQUIRED_ERROR_MARKER_SCHEMA,
                        "error": message,
                        "pid": os.getpid(),
                    },
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, marker)
            temporary = None
        except OSError as marker_error:
            warnings.warn(
                "could not publish the required variable Top-K failure marker "
                f"{marker}: {marker_error}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    raise RuntimeError(message)


@lru_cache(maxsize=None)
def build_variable_topk(cuda_arch="90"):
    from torch.utils.cpp_extension import load

    if cuda_arch != "90":
        raise ValueError("Variable Top-K currently supports the validated SM90 path only.")
    directory = Path(__file__).resolve().parent
    sources = [directory / "variable_topk.cpp", directory / "variable_topk.cu"]
    digest = hashlib.sha256(f"{torch.__version__}:{torch.version.cuda}:{cuda_arch}".encode())
    for source in sources:
        if source.is_file():
            digest.update(source.read_bytes())
    name = "bts_variable_topk_" + digest.hexdigest()[:12]
    build = Path(tempfile.gettempdir()) / "bts_extensions" / name
    build.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=list(map(str, sources)), build_directory=str(build),
        extra_cflags=["-O3"], with_cuda=True,
        extra_cuda_cflags=["-O3", "--ftz=false", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"])


@lru_cache(maxsize=None)
def _extension_or_none():
    try:
        return build_variable_topk()
    except (RuntimeError, OSError, ImportError) as exc:
        warnings.warn(f"BTS variable Top-K build unavailable; using the existing selector: {exc}",
                      RuntimeWarning, stacklevel=2)
        return None


@lru_cache(maxsize=None)
def _supported_device(device):
    return torch.cuda.get_device_capability(device) == (9, 0)


def variable_topk_supported(scores, blocks, rows, budgets, output, *, topk, block_size, kv_len):
    tensors = (scores, blocks, rows, budgets, output)
    if any(t is None or not t.is_cuda or not t.is_contiguous() for t in tensors):
        return False
    if any(t.device != scores.device for t in tensors):
        return False
    if (scores.ndim != 2 or blocks.ndim != 2 or rows.ndim != 1 or budgets.ndim != 1
            or output.ndim != 2 or scores.dtype != torch.float32
            or blocks.dtype not in (torch.int32, torch.int64)
            or rows.dtype not in (torch.int32, torch.int64)
            or budgets.dtype != torch.int32 or output.dtype != torch.int32):
        return False
    r, n = scores.shape
    if (r == 0 or not 8192 < n <= 131072 or not 0 < topk <= min(n, _MAX_TOPK)
            or block_size not in (64, 128, 256) or not 0 < kv_len < 2**31
            or blocks.shape[0] != r or blocks.shape[1] * block_size < n
            or rows.numel() != r or output.shape[1] < topk
            or budgets.numel() != output.shape[0] or r * topk >= 2**31):
        return False
    return _supported_device(scores.device)


# Row counts and budgets vary between layers/documents. They affect bounds,
# not the tile: compiling their Cartesian product defeats the JIT cache.
@triton.jit(do_not_specialize=["R", "CAP"])
def _gather_budgets(ROWS, COUNTS, LOCAL, R, CAP, T: tl.constexpr):
    x = tl.program_id(0) * T + tl.arange(0, T)
    row = tl.load(ROWS + x, x < R, 0)
    count = tl.load(COUNTS + row, x < R, 0)
    # Empty rows use dummy K=1. Mapping below discards their selected item.
    tl.store(LOCAL + x, tl.minimum(tl.maximum(count, 1), CAP).to(tl.int32), x < R)


@triton.jit(do_not_specialize=["K", "BW", "OS", "KV"])
def _map_indices(INDEX, BLOCKS, ROWS, COUNTS, OUT,
                 K, BW, OS, BS: tl.constexpr, KV, T: tl.constexpr):
    r = tl.program_id(0)
    rank = tl.program_id(1) * T + tl.arange(0, T)
    row = tl.load(ROWS + r).to(tl.int64)
    count = tl.minimum(tl.maximum(tl.load(COUNTS + row), 0), K)
    keep = rank < count
    index = tl.load(INDEX + r * K + rank, keep, 0)
    block = tl.load(BLOCKS + r * BW + index // BS, keep, 0).to(tl.int64)
    position = block * BS + index % BS
    tl.store(OUT + row * OS + rank, position.to(tl.int32),
             keep & (position >= 0) & (position < KV))


def try_variable_topk_to_token_positions(
    scores, blocks, rows, budgets, output, *, topk, block_size, kv_len
):
    """Write exact per-row-K prefixes, or return False without modifying output."""
    required = deterministic_variable_topk_required()
    deterministic = deterministic_variable_topk_enabled()
    if required and not deterministic:
        _raise_required_error(
            f"{_REQUIRE_DETERMINISTIC_ENV}=1 requires {_DETERMINISTIC_ENV}=1"
        )
    if not variable_topk_supported(scores, blocks, rows, budgets, output,
            topk=topk, block_size=block_size, kv_len=kv_len):
        if required:
            _raise_required_error("the selector route is unsupported")
        return False
    module = _extension_or_none()
    if module is None:
        if required:
            _raise_required_error("the native extension is unavailable")
        return False
    indices = torch.empty((scores.shape[0], topk), device=scores.device, dtype=torch.int32)
    local = torch.empty(scores.shape[0], device=scores.device, dtype=torch.int32)
    with torch.cuda.device(scores.device):
        _gather_budgets[(triton.cdiv(rows.numel(), 256),)](rows, budgets, local,
            R=rows.numel(), CAP=topk, T=256)
        # The caller maps only selected indices into global token positions;
        # selected score values are never consumed.  The native selector has
        # an explicit optional-values path, so avoid a second [rows, topk]
        # buffer (2.71 GiB in the 128K FastDeploy profile).
        if deterministic:
            module.select_deterministic(scores, local, indices, None)
        else:
            module.select(scores, local, indices, None)
        _map_indices[(rows.numel(), triton.cdiv(topk, 256))](indices, blocks, rows, budgets, output,
            K=topk, BW=blocks.shape[1], OS=output.shape[1], BS=block_size, KV=kv_len, T=256)
    return True
