"""Explicit worker-local installation through FastDeploy's plugin loader."""

import importlib
from importlib.metadata import version
import logging
import os

SUPPORTED_FASTDEPLOY = "2.5.0"
REQUIRED_FLASH_ATTN_VERSION = 2


def _require_chunked_prefill_disabled():
    """Set the FastDeploy 2.5 opt-out, or reject a contradictory setting."""
    current = os.environ.get("FD_DISABLE_CHUNKED_PREFILL")
    if current is None:
        os.environ["FD_DISABLE_CHUNKED_PREFILL"] = "1"
    elif current != "1":
        raise ValueError(
            "BTS FastDeploy requires FD_DISABLE_CHUNKED_PREFILL=1; "
            f"found {current!r}"
        )


def _force_flash_attention_2(module):
    """Keep FastDeploy's selector and Paddle's dispatch flag on FA2.

    FastDeploy 2.5 independently selects FA3 on SM90 through the module global
    ``FLASH_ATTN_VERSION``.  ``FLAGS_flash_attn_version`` alone does not change
    that choice and would leave the two selectors inconsistent.  The benchmark
    contract requires FA2 for every native dense prefill/fallback, so set and
    verify both selectors in every launcher and spawned GPU worker.
    """
    import paddle

    module.FLASH_ATTN_VERSION = REQUIRED_FLASH_ATTN_VERSION
    paddle.set_flags({"FLAGS_flash_attn_version": REQUIRED_FLASH_ATTN_VERSION})
    paddle_version = paddle.base.framework.get_flags(
        ["FLAGS_flash_attn_version"]
    )["FLAGS_flash_attn_version"]
    if (
        module.FLASH_ATTN_VERSION != REQUIRED_FLASH_ATTN_VERSION
        or int(paddle_version) != REQUIRED_FLASH_ATTN_VERSION
    ):
        raise RuntimeError(
            "Could not enforce FastDeploy/Paddle FlashAttention version 2: "
            f"fastdeploy={module.FLASH_ATTN_VERSION!r}, paddle={paddle_version!r}"
        )
    return {
        "fastdeploy_flash_attn_version": int(module.FLASH_ATTN_VERSION),
        "paddle_flash_attn_version": int(paddle_version),
        "dense_prefill_op": "flash_attn_unpadded",
        "decode_op": "append_attention",
    }


def register():
    # Entry points may be discovered automatically. Installation alone must not
    # change either the research process or an ordinary FastDeploy invocation.
    if os.environ.get("BTS_FASTDEPLOY_ENABLE") != "1":
        return
    # FastDeploy 2.5 consults this while constructing EngineArgs and otherwise
    # re-enables chunked prefill for CUDA mixed-mode engines. Apply it before
    # importing an engine backend or constructing any BTS configuration.
    _require_chunked_prefill_disabled()
    actual = version("fastdeploy-gpu")
    if actual.split("+")[0] != SUPPORTED_FASTDEPLOY:
        raise RuntimeError(f"BTS adapter supports fastdeploy-gpu=={SUPPORTED_FASTDEPLOY}; found {actual}")
    if os.environ.get("FD_ATTENTION_BACKEND") != "FLASH_ATTN":
        raise ValueError("BTS FastDeploy requires FD_ATTENTION_BACKEND=FLASH_ATTN")
    mode = os.environ.get("BTS_FASTDEPLOY_MODE", "bts")
    if mode not in ("bts", "dense_rope_compat"):
        raise ValueError(
            "BTS_FASTDEPLOY_MODE must be 'bts' or 'dense_rope_compat'; "
            f"found {mode!r}"
        )
    module = importlib.import_module("fastdeploy.model_executor.layers.attention.flash_attn_backend")
    attention_contract = _force_flash_attention_2(module)
    existing_mode = getattr(
        module.FlashAttentionBackend, "_bts_fastdeploy_mode", None
    )
    if existing_mode is not None:
        if existing_mode == mode:
            return
        raise RuntimeError(
            "FastDeploy backend is already installed in mode "
            f"{existing_mode!r}; dense and BTS require fresh processes"
        )
    from .backend import make_backend, make_dense_backend
    replacement = (
        make_backend(module.FlashAttentionBackend)
        if mode == "bts"
        else make_dense_backend(module.FlashAttentionBackend)
    )
    replacement._bts_native_attention_contract = attention_contract
    module.FlashAttentionBackend = replacement
    package = importlib.import_module("fastdeploy.model_executor.layers.attention")
    package.FlashAttentionBackend = replacement
    # The platform selector resolves this class by its module path at worker
    # startup. Clear any early cached resolution made before plugin loading.
    selector = importlib.import_module("fastdeploy.model_executor.layers.attention.attention_selecter")
    selector._get_attn_backend.cache_clear()
    if mode == "bts":
        message = "BTS FastDeploy adapter enabled (full prefill; native decode)"
    else:
        message = (
            "FastDeploy dense baseline enabled with PaddleFormers identity-RoPE "
            "compatibility guard (native dense attention)"
        )
    logging.getLogger(__name__).warning(message)
