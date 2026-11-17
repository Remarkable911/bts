"""Per-request cache lifecycle required by the long-context MInference run."""

from __future__ import annotations

import gc
from functools import wraps
from typing import Any


def install_hflm_cache_lifecycle(hflm_class: type | None = None) -> dict[str, object]:
    """Install the audited request-local DynamicCache lifecycle once.

    MInference's 128K evaluation otherwise retains request KV tensors long
    enough to accumulate into an OOM.  The hook does not change prompts,
    generation parameters, attention selection, or returned token IDs.
    """
    import torch
    from transformers.cache_utils import DynamicCache

    if hflm_class is None:
        from lm_eval.models.huggingface import HFLM

        hflm_class = HFLM

    if getattr(hflm_class, "_minference_explicit_cache_lifecycle", False):
        return {
            "installed": True,
            "already_installed": True,
            "policy": "request_local_dynamic_cache_clear_layers",
        }

    original = hflm_class._model_generate

    @wraps(original)
    def generate_with_request_cache(self: Any, *args: Any, **kwargs: Any):
        if kwargs.get("past_key_values") is not None:
            return original(self, *args, **kwargs)

        cache = DynamicCache(config=self.model.config)
        kwargs["past_key_values"] = cache
        try:
            result = original(self, *args, **kwargs)
            if isinstance(result, torch.Tensor) and result.is_cuda:
                result = result.cpu()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return result
        finally:
            kwargs.pop("past_key_values", None)
            layers = getattr(cache, "layers", None)
            if layers is None:
                raise RuntimeError(
                    "installed Transformers DynamicCache has no layers collection"
                )
            layers.clear()
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    hflm_class._model_generate = generate_with_request_cache
    hflm_class._minference_explicit_cache_lifecycle = True
    return {
        "installed": True,
        "already_installed": False,
        "policy": "request_local_dynamic_cache_clear_layers",
    }
