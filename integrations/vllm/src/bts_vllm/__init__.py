"""Out-of-tree BTS attention backend for exactly vLLM 0.17.0.

The package is intentionally inert on import.  vLLM activates it through a
``vllm.general_plugins`` entry point that targets :func:`bts_vllm.plugin.register`.
"""

from .compat import SUPPORTED_VLLM_VERSION
from .fa2_compat import ADAPTER_NAME, LOG_MARKER_PREFIX

__all__ = ["ADAPTER_NAME", "LOG_MARKER_PREFIX", "SUPPORTED_VLLM_VERSION"]
