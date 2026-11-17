"""Version-scoped replacement of only FastDeploy's prefill attention call.

Clone the upstream method with a private globals dictionary. Cache write,
decoder and output merge remain in FastDeploy. When PaddleFormers has already
applied RoPE, an identity rotary table prevents the fused FastDeploy path from
rotating Q/K a second time. No module-level attention function is replaced,
and importing this module never modifies FastDeploy.
"""

from contextvars import ContextVar
import functools
import inspect
import os
from types import FunctionType

from .audit import Audit
from .config import config_digest, load_config, validate_engine

_active = ContextVar("bts_fastdeploy_backend", default=None)


class _IdentityRotaryMixin:
    """Prevent FastDeploy from rotating PaddleFormers Q/K a second time."""

    def _get_identity_rotary_embs(self, original_rotary_embs):
        import paddle

        spec = (tuple(original_rotary_embs.shape),
                str(original_rotary_embs.dtype),
                str(original_rotary_embs.place))
        identity = getattr(self, "_bts_identity_rotary_embs", None)
        if (identity is None
                or getattr(self, "_bts_identity_rotary_embs_spec", None) != spec):
            identity = paddle.zeros_like(original_rotary_embs)
            identity[0] = 1.0
            identity[1] = 0.0
            self._bts_identity_rotary_embs = identity
            self._bts_identity_rotary_embs_spec = spec
        return identity

    def _install_identity_rotary(self, forward_meta, layer_id):
        original = getattr(forward_meta, "rotary_embs", None)
        if not (getattr(forward_meta, "rope_already_applied", False)
                and original is not None):
            return None
        identity = self._get_identity_rotary_embs(original)
        self.bts_audit.record_rope_guard(layer_id)
        forward_meta.rotary_embs = identity
        return original

    def _record_native_attention_contract(self):
        contract = getattr(type(self), "_bts_native_attention_contract", None)
        # Direct construction is used by framework-free unit tests. Production
        # construction always comes through plugin.register(), which installs
        # the verified runtime contract on the replacement class.
        if contract is None:
            # The mmap transport writes static JSON metadata separately from
            # its live counters.  Persist any metadata the caller populated
            # before this hook (notably dense native-kernel provenance), even
            # for direct framework-free construction in unit tests.
            self.bts_audit.flush()
            return
        if not isinstance(contract, dict):
            raise RuntimeError("BTS FastDeploy native attention contract is missing")
        if (
            contract.get("fastdeploy_flash_attn_version") != 2
            or contract.get("paddle_flash_attn_version") != 2
            or contract.get("dense_prefill_op") != "flash_attn_unpadded"
            or contract.get("decode_op") != "append_attention"
        ):
            raise RuntimeError(
                f"BTS FastDeploy requires the frozen FA2/native-decode contract: {contract!r}"
            )
        self.bts_audit.data["native_attention_contract"] = dict(contract)
        self.bts_audit.flush()


def clone_prefill_method(original, attention):
    names = original.__code__.co_names
    required = {"gqa_rope_write_cache", "flash_attn_func", "append_attention", "merge_prefill_decode_output"}
    if not required.issubset(names):
        raise RuntimeError("Unsupported FastDeploy FlashAttentionBackend.forward_mixed implementation")
    expected = ("self", "q", "k", "v", "qkv", "compressed_kv", "k_pe", "layer", "forward_meta")
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError("FastDeploy forward_mixed signature changed")
    namespace = dict(original.__globals__)
    namespace["flash_attn_func"] = attention
    clone = FunctionType(original.__code__, namespace, original.__name__,
                         original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    return functools.update_wrapper(clone, original)


def _bts_flash_attention(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None,
                          max_seqlen_q=None, max_seqlen_k=None, attn_mask_q=None,
                          causal=True, num_heads=None, kv_num_heads=None, head_dim=128, version=None):
    from .bridge import prefill_paddle
    state = _active.get()
    if state is None:
        raise RuntimeError("BTS prefill called outside its FastDeploy backend")
    backend, layer_id, forward_meta = state
    output, lengths = prefill_paddle(
        q, k, v, cu_seqlens_q, cu_seqlens_k, config=backend.bts_config,
        num_heads=num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim,
        causal=causal, attn_mask_q=attn_mask_q,
        max_num_seqs=backend.bts_max_num_seqs,
        seq_lens_encoder=getattr(forward_meta, "seq_lens_encoder", None),
        seq_lens_decoder=getattr(forward_meta, "seq_lens_decoder", None),
    )
    backend.bts_audit.record(layer_id, lengths)
    # FastDeploy's caller consumes element 0 only; no fabricated softmax/LSE.
    return (output,)


def make_backend(original_class):
    forward = clone_prefill_method(original_class.forward_mixed, _bts_flash_attention)

    class BTSFlashAttentionBackend(_IdentityRotaryMixin, original_class):
        _bts_fastdeploy_adapter = True
        _bts_fastdeploy_dense_compat = False
        _bts_fastdeploy_mode = "bts"

        def __init__(self, fd_config, *args, **kwargs):
            validate_engine(fd_config)
            self.bts_max_num_seqs = fd_config.scheduler_config.max_num_seqs
            self.bts_config = load_config(os.environ.get("BTS_FASTDEPLOY_CONFIG"))
            self.bts_audit = Audit(config_digest(self.bts_config), os.environ.get("BTS_FASTDEPLOY_AUDIT_DIR"))
            self._record_native_attention_contract()
            super().__init__(fd_config, *args, **kwargs)

        def forward_mixed(self, q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta):
            cache_type = getattr(layer, "cache_quant_type_str", "none")
            if cache_type not in (None, "none"):
                raise ValueError("BTS FastDeploy requires unquantized KV cache")
            if getattr(forward_meta, "attn_mask_offsets", None) is not None:
                raise ValueError("BTS FastDeploy does not support custom attention masks")
            original_rotary_embs = self._install_identity_rotary(
                forward_meta, layer.layer_id
            )
            token = _active.set((self, layer.layer_id, forward_meta))
            before = self.bts_audit.data["prefill_calls"]
            try:
                result = forward(self, q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
                if self.bts_audit.data["prefill_calls"] == before:
                    self.bts_audit.record(layer.layer_id)
            finally:
                try:
                    if original_rotary_embs is not None:
                        forward_meta.rotary_embs = original_rotary_embs
                finally:
                    _active.reset(token)
            return result

    BTSFlashAttentionBackend.__module__ = __name__
    return BTSFlashAttentionBackend


def make_dense_backend(original_class):
    """Wrap native dense attention with only the PaddleFormers RoPE guard.

    FastDeploy 2.5's AppendAttention backend already has this guard, but its
    FlashAttention backend does not.  This wrapper deliberately calls the
    original class method, so cache handling, native dense FlashAttention,
    decode and output merge all remain upstream FastDeploy code.
    """

    native_flash_attention = original_class.forward_mixed.__globals__.get(
        "flash_attn_func"
    )
    if not callable(native_flash_attention):
        raise RuntimeError(
            "FastDeploy dense forward_mixed no longer resolves native flash_attn_func"
        )

    class DenseFlashAttentionBackend(_IdentityRotaryMixin, original_class):
        _bts_fastdeploy_adapter = False
        _bts_fastdeploy_dense_compat = True
        _bts_fastdeploy_mode = "dense_rope_compat"
        _bts_native_flash_attn_preserved = True

        def __init__(self, fd_config, *args, **kwargs):
            validate_engine(fd_config)
            self.bts_audit = Audit(
                "not-applicable-dense",
                os.environ.get("BTS_FASTDEPLOY_AUDIT_DIR"),
                method_display_name="FastDeploy dense FlashAttention",
                backend_mode="dense_rope_compat",
            )
            preserved = (
                original_class.forward_mixed.__globals__.get("flash_attn_func")
                is native_flash_attention
            )
            if not preserved:
                raise RuntimeError("native FastDeploy flash_attn_func was replaced")
            self.bts_audit.data["native_flash_attention"] = {
                "module": getattr(native_flash_attention, "__module__", None),
                "qualname": getattr(native_flash_attention, "__qualname__", None),
                "preserved": True,
            }
            self._record_native_attention_contract()
            super().__init__(fd_config, *args, **kwargs)

        def forward_mixed(self, q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta):
            original_rotary_embs = self._install_identity_rotary(
                forward_meta, layer.layer_id
            )
            try:
                result = super().forward_mixed(
                    q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta
                )
            finally:
                if original_rotary_embs is not None:
                    forward_meta.rotary_embs = original_rotary_embs
            return result

    DenseFlashAttentionBackend.__module__ = __name__
    return DenseFlashAttentionBackend
