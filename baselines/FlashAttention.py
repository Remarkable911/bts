"""Explicit FlashAttention-2 baseline contract.

FlashAttention is selected through Transformers' native
``attn_implementation="flash_attention_2"`` path.  There is intentionally
no monkey-patched forward in this module.
"""

from __future__ import annotations

import importlib.metadata


ATTN_IMPLEMENTATION = "flash_attention_2"
DISPLAY_NAME = "FlashAttention-2"


def model_args() -> dict[str, str]:
    """Return the exact Hugging Face model-load override."""
    return {"attn_implementation": ATTN_IMPLEMENTATION}


def preflight_backend() -> dict[str, object]:
    """Import FlashAttention before allocating model weights."""
    import flash_attn

    return {
        "callable": callable(getattr(flash_attn, "flash_attn_func", None)),
        "attn_implementation": ATTN_IMPLEMENTATION,
        "distribution": "flash-attn",
        "version": importlib.metadata.version("flash-attn"),
    }
