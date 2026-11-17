"""vLLM general-plugin entry point for the BTS custom attention backend."""

from __future__ import annotations

import os

from .audit import AUDIT
from .compat import validate_vllm_api_contract
from .fa2_compat import register as register_fa2_compat


_REGISTERED_PID: int | None = None
_BACKEND_PATH = "bts_vllm.backend.BTSFlashAttentionBackend"


def register() -> None:
    """Register BTS as ``AttentionBackendEnum.CUSTOM`` in every vLLM process."""

    global _REGISTERED_PID
    pid = os.getpid()
    # Dense fallback and decode must use the same audited upstream FA2 adapter
    # as the standalone dense arm.  Register it before importing vLLM's
    # attention backend so all module-level aliases bind to the adapter.
    register_fa2_compat()
    if _REGISTERED_PID == pid:
        return

    validate_vllm_api_contract()
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    custom = AttentionBackendEnum.CUSTOM
    if custom.is_overridden():
        existing = custom.get_path()
        if existing != _BACKEND_PATH:
            raise RuntimeError(
                "AttentionBackendEnum.CUSTOM is already registered by another "
                f"plugin: {existing}"
            )
    else:
        register_backend(custom, _BACKEND_PATH)
    _REGISTERED_PID = pid
    AUDIT.record(backend_activations=1)


__all__ = ["register"]
