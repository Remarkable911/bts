"""Frozen comparison baselines for the standalone submission.

Modules are intentionally lazy: listing methods does not import CUDA
extensions.
"""

from .registry import (
    ATTENTION_NAMES,
    BASELINE_NAMES,
    DENSITY_DEFINITIONS,
    PreparedAttention,
    configure_environment,
    dependency_report,
    get_spec,
    install_runtime_hooks,
    load_forward,
    method_contract,
    normalize_name,
    prepare_attention,
)

__all__ = (
    "ATTENTION_NAMES",
    "BASELINE_NAMES",
    "DENSITY_DEFINITIONS",
    "PreparedAttention",
    "configure_environment",
    "dependency_report",
    "get_spec",
    "install_runtime_hooks",
    "load_forward",
    "method_contract",
    "normalize_name",
    "prepare_attention",
)
