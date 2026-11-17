"""Fail-closed compatibility checks for the private vLLM 0.17.0 API surface."""

from __future__ import annotations

from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
import inspect


SUPPORTED_VLLM_VERSION = "0.17.0"


def installed_vllm_version() -> str:
    """Return the installed distribution version without importing vLLM."""

    try:
        return version("vllm")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "BTS vLLM backend requires the installed vllm==0.17.0 distribution"
        ) from error


def require_supported_vllm() -> str:
    """Reject every vLLM release other than the audited 0.17.0 build."""

    observed = installed_vllm_version()
    if observed != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            "BTS vLLM backend is locked to vllm==0.17.0; "
            f"found {observed!r}"
        )
    return observed


def _parameter_names(callable_object) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_object).parameters)


def validate_vllm_api_contract() -> None:
    """Verify the audited methods and metadata fields before registration.

    These are deliberately private vLLM interfaces.  The exact-version check
    catches normal upgrades; the shape check also catches locally patched
    0.17.0 installations that changed the integration contract.
    """

    require_supported_vllm()

    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionImpl,
        FlashAttentionMetadata,
        FlashAttentionMetadataBuilder,
    )

    expected_signatures = {
        "FlashAttentionImpl.__init__": (
            "self",
            "num_heads",
            "head_size",
            "scale",
            "num_kv_heads",
            "alibi_slopes",
            "sliding_window",
            "kv_cache_dtype",
            "logits_soft_cap",
            "attn_type",
            "kv_sharing_target_layer_name",
            "sinks",
        ),
        "FlashAttentionImpl.forward": (
            "self",
            "layer",
            "query",
            "key",
            "value",
            "kv_cache",
            "attn_metadata",
            "output",
            "output_scale",
            "output_block_scale",
        ),
        "FlashAttentionMetadataBuilder.__init__": (
            "self",
            "kv_cache_spec",
            "layer_names",
            "vllm_config",
            "device",
        ),
        "FlashAttentionMetadataBuilder.build": (
            "self",
            "common_prefix_len",
            "common_attn_metadata",
            "fast_build",
        ),
    }
    observed_signatures = {
        "FlashAttentionImpl.__init__": _parameter_names(FlashAttentionImpl.__init__),
        "FlashAttentionImpl.forward": _parameter_names(FlashAttentionImpl.forward),
        "FlashAttentionMetadataBuilder.__init__": _parameter_names(
            FlashAttentionMetadataBuilder.__init__
        ),
        "FlashAttentionMetadataBuilder.build": _parameter_names(
            FlashAttentionMetadataBuilder.build
        ),
    }
    mismatches = [
        f"{name}: expected={expected_signatures[name]!r}, observed={observed!r}"
        for name, observed in observed_signatures.items()
        if observed != expected_signatures[name]
    ]

    expected_metadata_fields = (
        "num_actual_tokens",
        "max_query_len",
        "query_start_loc",
        "max_seq_len",
        "seq_lens",
        "block_table",
        "slot_mapping",
        "use_cascade",
        "common_prefix_len",
        "cu_prefix_query_lens",
        "prefix_kv_lens",
        "suffix_kv_lens",
        "max_dcp_context_kv_len",
        "dcp_context_kv_lens",
        "scheduler_metadata",
        "prefix_scheduler_metadata",
        "max_num_splits",
        "causal",
    )
    observed_metadata_fields = tuple(
        field.name for field in fields(FlashAttentionMetadata)
    )
    if observed_metadata_fields != expected_metadata_fields:
        mismatches.append(
            "FlashAttentionMetadata fields: "
            f"expected={expected_metadata_fields!r}, "
            f"observed={observed_metadata_fields!r}"
        )
    if FlashAttentionBackend.forward_includes_kv_cache_update is not False:
        mismatches.append(
            "FlashAttentionBackend.forward_includes_kv_cache_update must be False"
        )
    if mismatches:
        raise RuntimeError(
            "installed vLLM does not match the audited 0.17.0 attention API: "
            + "; ".join(mismatches)
        )
