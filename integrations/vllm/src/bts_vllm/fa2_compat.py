"""Fail-closed vLLM 0.17 to upstream FlashAttention 2 adapter.

The CUDA extension bundled by the audited vLLM wheel exposes a 22-argument
``varlen_fwd`` op.  The separately installed upstream FlashAttention 2.8.2
extension exposes the equivalent 21-argument ABI: it omits vLLM's
``num_splits`` argument and returns four tensors instead of two.  This module
keeps vLLM's Python-level contract while routing FA2 execution through the
upstream extension.

Registration is deliberately fail-closed.  A missing/wrong upstream package,
an unexpected vLLM wrapper/op schema, or an incompatible paged-KV block size
raises before the vendored CUDA op can be used.
"""

from __future__ import annotations

import hashlib
import inspect
from importlib.machinery import EXTENSION_SUFFIXES
import json
import mmap
import os
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
import struct
import sys
import threading
from types import ModuleType
from typing import Any, Callable

from .compat import require_supported_vllm


ADAPTER_NAME = "bts_vllm_upstream_flash_attn_2_cuda"
LOG_MARKER_PREFIX = "BTS_VLLM_FA2_ADAPTER_CALL "
MARKER_SCHEMA_VERSION = "bts_vllm_fa2_adapter_call_v2"
SUPPORTED_FLASH_ATTN_VERSION = "2.8.2"
VLLM_OP_ARGUMENT_COUNT = 22
UPSTREAM_ARGUMENT_COUNT = 21
PAGED_KV_BLOCK_ALIGNMENT = 256
FA2_AUDIT_ENV = "BTS_VLLM_FA2_AUDIT_DIR"

_FA2_AUDIT_MAGIC = b"BTSFA2\0\0"
_FA2_AUDIT_SCHEMA_VERSION = 1
_FA2_AUDIT_RECORD = struct.Struct("<8sIIQQ")
_FA2_AUDIT_MMAP_BYTES = 4096

_WRAPPER_PARAMETER_NAMES = (
    "q",
    "k",
    "v",
    "max_seqlen_q",
    "cu_seqlens_q",
    "max_seqlen_k",
    "cu_seqlens_k",
    "seqused_k",
    "q_v",
    "dropout_p",
    "softmax_scale",
    "causal",
    "window_size",
    "softcap",
    "alibi_slopes",
    "deterministic",
    "return_attn_probs",
    "block_table",
    "return_softmax_lse",
    "out",
    "scheduler_metadata",
    "q_descale",
    "k_descale",
    "v_descale",
    "num_splits",
    "fa_version",
    "s_aux",
    "cp_world_size",
    "cp_rank",
    "cp_tot_seqused_k",
)
_VLLM_OP_PARAMETER_NAMES = (
    "q",
    "k",
    "v",
    "out",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_k",
    "leftpad_k",
    "block_table",
    "alibi_slopes",
    "max_seqlen_q",
    "max_seqlen_k",
    "p_dropout",
    "softmax_scale",
    "zero_tensors",
    "is_causal",
    "window_size_left",
    "window_size_right",
    "softcap",
    "return_softmax",
    "num_splits",
    "gen",
)
_ALIAS_MODULE_NAMES = (
    "vllm.vllm_flash_attn.flash_attn_interface",
    "vllm.vllm_flash_attn",
    "vllm.v1.attention.backends.fa_utils",
    "vllm.v1.attention.backends.flash_attn",
)

_LOCK = threading.RLock()
_INSTALLED_PID: int | None = None
_MARKER_EMITTED_PID: int | None = None
_ORIGINAL_FLASH_ATTN_VARLEN_FUNC: Callable[..., Any] | None = None
_UPSTREAM_EXTENSION: ModuleType | Any | None = None
_TORCH: ModuleType | Any | None = None
_PROVENANCE: dict[str, Any] | None = None
_AUDIT_PID: int | None = None
_AUDIT_CALLS = 0
_AUDIT_PAGED_CALLS = 0
_AUDIT_MAPPING: mmap.mmap | None = None
_AUDIT_FILE: Any | None = None
_AUDIT_PATH: Path | None = None


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"{ADAPTER_NAME} requires the installed {name} distribution"
        ) from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_extension_distribution(extension: Any) -> dict[str, Any]:
    """Bind the imported native module to the installed flash-attn dist."""

    raw_file = getattr(extension, "__file__", None)
    if not isinstance(raw_file, str) or not raw_file:
        raise RuntimeError("flash_attn_2_cuda does not expose a binary file path")
    try:
        extension_file = Path(raw_file).resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"flash_attn_2_cuda binary does not exist: {raw_file!r}"
        ) from error
    if not extension_file.is_file() or not any(
        extension_file.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES
    ):
        raise RuntimeError(
            "flash_attn_2_cuda must resolve to a native Python extension; "
            f"found {extension_file}"
        )

    try:
        installed = distribution("flash-attn")
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"{ADAPTER_NAME} requires the installed flash-attn distribution"
        ) from error
    files = installed.files
    if files is None:
        raise RuntimeError(
            "installed flash-attn distribution has no file manifest; cannot "
            "prove flash_attn_2_cuda ownership"
        )
    owning_file = None
    for candidate in files:
        try:
            candidate_path = Path(installed.locate_file(candidate)).resolve(strict=True)
        except OSError:
            continue
        if candidate_path == extension_file:
            owning_file = str(candidate)
            break
    if owning_file is None:
        raise RuntimeError(
            "imported flash_attn_2_cuda binary is not owned by the installed "
            f"flash-attn distribution: {extension_file}"
        )
    return {
        "upstream_extension_distribution": "flash-attn",
        "upstream_extension_distribution_file": owning_file,
        "upstream_extension_file": str(extension_file),
        "upstream_extension_sha256": _file_sha256(extension_file),
        "upstream_extension_in_distribution": True,
    }


def _validate_wrapper_signature(callable_object: Callable[..., Any]) -> None:
    observed = tuple(inspect.signature(callable_object).parameters)
    if observed != _WRAPPER_PARAMETER_NAMES:
        raise RuntimeError(
            "vLLM FlashAttention wrapper signature differs from the audited "
            f"0.17.0 contract: expected={_WRAPPER_PARAMETER_NAMES!r}, "
            f"observed={observed!r}"
        )


def _validate_vllm_op_schema(torch_module: Any) -> None:
    try:
        schema = torch_module.ops._vllm_fa2_C.varlen_fwd.default._schema
        observed = tuple(argument.name for argument in schema.arguments)
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            "vLLM 0.17 did not register the audited _vllm_fa2_C::varlen_fwd op"
        ) from error
    if observed != _VLLM_OP_PARAMETER_NAMES:
        raise RuntimeError(
            "vLLM FA2 op schema differs from the audited 22-argument ABI: "
            f"expected={_VLLM_OP_PARAMETER_NAMES!r}, observed={observed!r}"
        )


def _validate_upstream_abi(extension: Any) -> None:
    operation = getattr(extension, "varlen_fwd", None)
    if operation is None or not callable(operation):
        raise RuntimeError("flash_attn_2_cuda does not expose callable varlen_fwd")
    # pybind11 builtins do not expose inspect.signature.  Their generated
    # docstring does enumerate arg0..arg20, which is stable for the pinned
    # 2.8.2 binary and lets us fail before the first CUDA invocation.
    doc = getattr(operation, "__doc__", "") or ""
    if "arg20:" not in doc or "arg21:" in doc:
        raise RuntimeError(
            "flash_attn_2_cuda.varlen_fwd does not expose the audited "
            "21-argument upstream ABI"
        )


def _patch_loaded_aliases() -> tuple[str, ...]:
    patched = []
    for name in _ALIAS_MODULE_NAMES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if not hasattr(module, "flash_attn_varlen_func"):
            continue
        setattr(module, "flash_attn_varlen_func", flash_attn_varlen_func)
        patched.append(name)
    return tuple(patched)


def _load_upstream_extension() -> Any:
    try:
        return import_module("flash_attn_2_cuda")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"{ADAPTER_NAME} requires importable flash_attn_2_cuda from "
            f"flash-attn=={SUPPORTED_FLASH_ATTN_VERSION}; refusing to fall "
            "back to vLLM's vendored FA2 extension"
        ) from error


def _close_audit_locked() -> None:
    global _AUDIT_MAPPING
    global _AUDIT_FILE
    global _AUDIT_PATH

    if _AUDIT_MAPPING is not None:
        _AUDIT_MAPPING.close()
    if _AUDIT_FILE is not None:
        _AUDIT_FILE.close()
    _AUDIT_MAPPING = None
    _AUDIT_FILE = None
    _AUDIT_PATH = None


def _write_audit_locked() -> None:
    if _AUDIT_MAPPING is None or _AUDIT_PID is None:
        return
    _FA2_AUDIT_RECORD.pack_into(
        _AUDIT_MAPPING,
        0,
        _FA2_AUDIT_MAGIC,
        _FA2_AUDIT_SCHEMA_VERSION,
        _AUDIT_PID,
        _AUDIT_CALLS,
        _AUDIT_PAGED_CALLS,
    )


def _ensure_audit_process_locked() -> None:
    global _AUDIT_PID
    global _AUDIT_CALLS
    global _AUDIT_PAGED_CALLS
    global _AUDIT_MAPPING
    global _AUDIT_FILE
    global _AUDIT_PATH

    pid = os.getpid()
    if _AUDIT_PID != pid:
        _close_audit_locked()
        _AUDIT_PID = pid
        _AUDIT_CALLS = 0
        _AUDIT_PAGED_CALLS = 0

    requested_dir = os.environ.get(FA2_AUDIT_ENV)
    if _AUDIT_MAPPING is not None or not requested_dir:
        return
    directory = Path(requested_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"fa2-worker-{pid}.mmap"
    file_object = path.open("w+b")
    file_object.truncate(_FA2_AUDIT_MMAP_BYTES)
    mapping = mmap.mmap(
        file_object.fileno(), _FA2_AUDIT_MMAP_BYTES, access=mmap.ACCESS_WRITE
    )
    _AUDIT_FILE = file_object
    _AUDIT_MAPPING = mapping
    _AUDIT_PATH = path
    _write_audit_locked()


def _record_usage(page_size: int | None) -> tuple[int, int]:
    global _AUDIT_CALLS
    global _AUDIT_PAGED_CALLS

    with _LOCK:
        _ensure_audit_process_locked()
        _AUDIT_CALLS += 1
        if page_size is not None:
            _AUDIT_PAGED_CALLS += 1
        _write_audit_locked()
        return _AUDIT_CALLS, _AUDIT_PAGED_CALLS


def read_call_audit(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one process-local cumulative adapter-call record."""

    audit_path = Path(path)
    raw = audit_path.read_bytes()[: _FA2_AUDIT_RECORD.size]
    if len(raw) != _FA2_AUDIT_RECORD.size:
        raise ValueError(f"truncated BTS vLLM FA2 audit record: {audit_path}")
    magic, schema, pid, calls, paged_calls = _FA2_AUDIT_RECORD.unpack(raw)
    if magic != _FA2_AUDIT_MAGIC or schema != _FA2_AUDIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported BTS vLLM FA2 audit record: {audit_path}")
    if paged_calls > calls:
        raise ValueError(f"invalid BTS vLLM FA2 audit counters: {audit_path}")
    return {
        "path": str(audit_path),
        "schema_version": schema,
        "pid": pid,
        "calls": calls,
        "paged_calls": paged_calls,
    }


def read_call_audits(directory: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read all cumulative adapter-call records in stable path order."""

    return [
        read_call_audit(path)
        for path in sorted(Path(directory).glob("fa2-worker-*.mmap"))
    ]


def register() -> None:
    """Install the adapter in one vLLM process.

    vLLM loads general plugins independently in the frontend, EngineCore, and
    model-worker processes.  PID-scoped idempotence therefore handles both
    repeated loads and forked process state without allowing an inherited
    registration to masquerade as a child-process installation.
    """

    global _INSTALLED_PID
    global _MARKER_EMITTED_PID
    global _ORIGINAL_FLASH_ATTN_VARLEN_FUNC
    global _UPSTREAM_EXTENSION
    global _TORCH
    global _PROVENANCE

    pid = os.getpid()
    with _LOCK:
        if _INSTALLED_PID == pid:
            patched = _patch_loaded_aliases()
            if _PROVENANCE is not None:
                _PROVENANCE["patched_aliases"] = list(patched)
            return

        observed_vllm = require_supported_vllm()
        observed_flash_attn = _distribution_version("flash-attn")
        if observed_flash_attn != SUPPORTED_FLASH_ATTN_VERSION:
            raise RuntimeError(
                f"{ADAPTER_NAME} is locked to flash-attn=="
                f"{SUPPORTED_FLASH_ATTN_VERSION}; found {observed_flash_attn!r}"
            )

        torch_module = import_module("torch")
        # Importing the package also binds the interface function at package
        # scope, so both aliases must be patched below.
        interface = import_module(
            "vllm.vllm_flash_attn.flash_attn_interface"
        )
        import_module("vllm.vllm_flash_attn")
        current = interface.flash_attn_varlen_func
        if current is flash_attn_varlen_func:
            original = _ORIGINAL_FLASH_ATTN_VARLEN_FUNC
            if original is None:
                raise RuntimeError("inherited FA2 adapter has no original wrapper")
        else:
            _validate_wrapper_signature(current)
            original = current

        _validate_vllm_op_schema(torch_module)
        upstream = _load_upstream_extension()
        _validate_upstream_abi(upstream)
        extension_identity = _validate_extension_distribution(upstream)

        # Commit global state only after every fail-closed check succeeds.
        _ORIGINAL_FLASH_ATTN_VARLEN_FUNC = original
        _UPSTREAM_EXTENSION = upstream
        _TORCH = torch_module
        _INSTALLED_PID = pid
        _MARKER_EMITTED_PID = None
        patched = _patch_loaded_aliases()
        _PROVENANCE = {
            "schema_version": "bts_vllm_fa2_compat_v1",
            "pid": pid,
            "adapter": ADAPTER_NAME,
            "adapter_file": str(__file__),
            "vllm_version": observed_vllm,
            "flash_attn_version": observed_flash_attn,
            **extension_identity,
            "vllm_op_argument_count": VLLM_OP_ARGUMENT_COUNT,
            "upstream_argument_count": UPSTREAM_ARGUMENT_COUNT,
            "paged_kv_block_alignment": PAGED_KV_BLOCK_ALIGNMENT,
            "patched_aliases": list(patched),
            "fallback_to_vendored_fa2": False,
        }


def provenance() -> dict[str, Any]:
    """Return process-local installation evidence or fail if not registered."""

    with _LOCK:
        if _INSTALLED_PID != os.getpid() or _PROVENANCE is None:
            raise RuntimeError(f"{ADAPTER_NAME} is not registered in this process")
        return dict(_PROVENANCE)


def _runtime_state() -> tuple[Any, Callable[..., Any], dict[str, Any]]:
    # vLLM may fork EngineCore after the frontend has already set its global
    # ``plugins_loaded`` flag.  In that case the child inherits our patched
    # function alias and parent-scoped state, but vLLM deliberately skips a
    # second plugin callback.  Re-run every fail-closed registration check in
    # the child before its first FA2 call; ``register`` is PID-idempotent and
    # the re-entrant lock keeps concurrent first calls safe.
    inherited_registration = (
        _INSTALLED_PID is not None
        and _UPSTREAM_EXTENSION is not None
        and _TORCH is not None
        and _ORIGINAL_FLASH_ATTN_VARLEN_FUNC is not None
        and _PROVENANCE is not None
    )
    if _INSTALLED_PID != os.getpid() and inherited_registration:
        register()
    with _LOCK:
        if (
            _INSTALLED_PID != os.getpid()
            or _UPSTREAM_EXTENSION is None
            or _TORCH is None
            or _ORIGINAL_FLASH_ATTN_VARLEN_FUNC is None
            or _PROVENANCE is None
        ):
            raise RuntimeError(
                f"{ADAPTER_NAME} was not registered in the current process; "
                "refusing to use vLLM's vendored FA2 extension"
            )
        return (
            _UPSTREAM_EXTENSION,
            _ORIGINAL_FLASH_ATTN_VARLEN_FUNC,
            dict(_PROVENANCE),
        )


def _maybe_contiguous(tensor: Any) -> Any:
    return (
        tensor.contiguous()
        if tensor is not None and tensor.stride(-1) != 1
        else tensor
    )


def _emit_usage_marker(
    provenance_record: dict[str, Any],
    page_size: int | None,
    call_counts: tuple[int, int],
) -> None:
    global _MARKER_EMITTED_PID

    pid = os.getpid()
    with _LOCK:
        if _MARKER_EMITTED_PID == pid:
            return
        payload = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "pid": pid,
            "adapter": ADAPTER_NAME,
            "vllm_version": provenance_record["vllm_version"],
            "flash_attn_version": provenance_record["flash_attn_version"],
            "upstream_argument_count": UPSTREAM_ARGUMENT_COUNT,
            "page_size": page_size,
            "cumulative_calls": call_counts[0],
            "cumulative_paged_calls": call_counts[1],
            "upstream_extension_distribution": provenance_record[
                "upstream_extension_distribution"
            ],
            "upstream_extension_distribution_file": provenance_record[
                "upstream_extension_distribution_file"
            ],
            "upstream_extension_file": provenance_record[
                "upstream_extension_file"
            ],
            "upstream_extension_sha256": provenance_record[
                "upstream_extension_sha256"
            ],
            "upstream_extension_in_distribution": provenance_record[
                "upstream_extension_in_distribution"
            ],
        }
        print(
            LOG_MARKER_PREFIX + json.dumps(payload, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        _MARKER_EMITTED_PID = pid


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: list[int] | None = None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    fa_version: int = 2,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
):
    """Dispatch vLLM FA2 calls through the pinned upstream 21-arg ABI."""

    upstream, original, provenance_record = _runtime_state()
    if fa_version != 2:
        return original(
            q,
            k,
            v,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            q_v=q_v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            block_table=block_table,
            return_softmax_lse=return_softmax_lse,
            out=out,
            scheduler_metadata=scheduler_metadata,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=num_splits,
            fa_version=fa_version,
            s_aux=s_aux,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            cp_tot_seqused_k=cp_tot_seqused_k,
        )

    if cu_seqlens_k is None and seqused_k is None:
        raise ValueError("cu_seqlens_k or seqused_k must be provided")
    if cu_seqlens_k is not None and seqused_k is not None:
        raise ValueError("cu_seqlens_k and seqused_k cannot both be provided")
    if block_table is not None and seqused_k is None:
        raise ValueError("paged KV requires seqused_k")
    unsupported = {
        "q_v": q_v,
        "s_aux": s_aux,
        "cp_tot_seqused_k": cp_tot_seqused_k,
    }
    present = sorted(name for name, value in unsupported.items() if value is not None)
    if present:
        raise NotImplementedError(
            "upstream FA2 compatibility path does not support: " + ", ".join(present)
        )
    # vLLM supplies q/k/v descale placeholder tensors on the ordinary BF16
    # path even though FA2 does not consume them.  Match the audited wrapper:
    # ignore those placeholders unless scheduler metadata activates the
    # quantized scheduling contract that upstream FA2 cannot implement.
    if scheduler_metadata is not None and all(
        value is not None for value in (q_descale, k_descale, v_descale)
    ):
        raise NotImplementedError(
            "upstream FA2 compatibility path does not support scheduler_metadata "
            "with q/k/v descale tensors"
        )
    if deterministic or return_attn_probs:
        raise NotImplementedError(
            "upstream FA2 compatibility path does not support deterministic or "
            "return_attn_probs"
        )
    if num_splits not in (0, 1):
        raise NotImplementedError(
            "upstream FA2 compatibility path requires num_splits 0 or 1"
        )
    if cp_world_size != 1 or cp_rank != 0:
        raise NotImplementedError(
            "upstream FA2 compatibility path does not support context parallelism"
        )

    page_size = None
    if block_table is not None:
        if getattr(k, "ndim", 0) < 2:
            raise ValueError("paged KV tensors must expose a page dimension")
        page_size = int(k.shape[1])
        if page_size % PAGED_KV_BLOCK_ALIGNMENT != 0:
            raise ValueError(
                "upstream FlashAttention 2 paged KV page size must be divisible "
                f"by {PAGED_KV_BLOCK_ALIGNMENT}; found {page_size}"
            )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if window_size is None:
        window_left, window_right = -1, -1
    else:
        if len(window_size) != 2:
            raise ValueError("window_size must contain exactly two integers")
        window_left, window_right = int(window_size[0]), int(window_size[1])
    q, k, v = (_maybe_contiguous(tensor) for tensor in (q, k, v))

    torch_module = _TORCH
    if torch_module is None:  # Guarded by _runtime_state; retained for typing.
        raise RuntimeError("torch runtime disappeared after FA2 adapter registration")
    effective_cu_seqlens_k = (
        cu_seqlens_k
        if cu_seqlens_k is not None
        else torch_module.zeros_like(cu_seqlens_q)
    )
    upstream_args = (
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        effective_cu_seqlens_k,
        seqused_k,
        None,  # leftpad_k
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,  # zero_tensors
        causal,
        window_left,
        window_right,
        softcap,
        return_softmax_lse and dropout_p > 0,
        None,  # Generator
    )
    if len(upstream_args) != UPSTREAM_ARGUMENT_COUNT:
        raise RuntimeError("internal upstream FA2 argument mapping is inconsistent")
    result = upstream.varlen_fwd(*upstream_args)
    if not isinstance(result, (list, tuple)) or len(result) != 4:
        raise RuntimeError(
            "flash_attn_2_cuda.varlen_fwd must return four tensors under the "
            "pinned 2.8.2 ABI"
        )
    output, softmax_lse, _, _ = result
    call_counts = _record_usage(page_size)
    _emit_usage_marker(provenance_record, page_size, call_counts)
    return (output, softmax_lse) if return_softmax_lse else output


__all__ = [
    "ADAPTER_NAME",
    "FA2_AUDIT_ENV",
    "LOG_MARKER_PREFIX",
    "MARKER_SCHEMA_VERSION",
    "PAGED_KV_BLOCK_ALIGNMENT",
    "SUPPORTED_FLASH_ATTN_VERSION",
    "UPSTREAM_ARGUMENT_COUNT",
    "VLLM_OP_ARGUMENT_COUNT",
    "flash_attn_varlen_func",
    "provenance",
    "read_call_audit",
    "read_call_audits",
    "register",
]
