from types import SimpleNamespace as NS

import pytest

from bts_fastdeploy.backend import make_backend, make_dense_backend
from bts_fastdeploy.config import validate_engine


def engine_config(max_num_seqs=1):
    return NS(cache_config=NS(enable_chunked_prefill=False, enable_prefix_caching=False),
              parallel_config=NS(tensor_parallel_size=1,data_parallel_size=1,pd_disaggregation_mode="mixed"),
              graph_opt_config=NS(use_cudagraph=False,graph_opt_level=0),
              model_config=NS(model_type="qwen3",causal=True,dtype="bfloat16",quantization_config=None),
              scheduler_config=NS(max_num_seqs=max_num_seqs), speculative_config=NS(method=None))


# Sentinels exercise the exact function-cloning boundary: original globals,
# cache preparation and decoder must remain independent of the sparse callback.
def gqa_rope_write_cache(value, rotary_embs=None):
    return value + ["rope_and_cache" if rotary_embs is None else rotary_embs]


def flash_attn_func(*args, **kwargs):
    return (["native_prefill"],)


def append_attention(value):
    return value + ["native_decode"]


def merge_prefill_decode_output(prefill, decode):
    return prefill + decode


class OriginalBackend:
    def __init__(self, fd_config):
        self.original_initialized = True

    def forward_mixed(self,q,k,v,qkv,compressed_kv,k_pe,layer,forward_meta):
        prepared = gqa_rope_write_cache(qkv)
        decoder = append_attention(prepared)
        if forward_meta.prefill:
            encoder = flash_attn_func(q,k,v,head_dim=128)[0]
            return merge_prefill_decode_output(encoder,decoder)
        return decoder


class RopeAwareBackend:
    def __init__(self, fd_config):
        self.original_initialized = True

    def forward_mixed(self,q,k,v,qkv,compressed_kv,k_pe,layer,forward_meta):
        prepared = gqa_rope_write_cache(qkv, forward_meta.rotary_embs)
        decoder = append_attention(prepared)
        if forward_meta.prefill:
            encoder = flash_attn_func(q,k,v,head_dim=128)[0]
            return merge_prefill_decode_output(encoder,decoder)
        return decoder


class ExplodingDenseBackend:
    def __init__(self, fd_config):
        self.original_initialized = True

    def forward_mixed(self,q,k,v,qkv,compressed_kv,k_pe,layer,forward_meta):
        # Keep the upstream global lookup contract visible to make_dense_backend.
        flash_attn_func(q, k, v)
        raise RuntimeError("native dense failed")


def test_only_prefill_changes_and_exception_does_not_fall_back(monkeypatch):
    from bts_fastdeploy import bridge
    original_fn = OriginalBackend.forward_mixed
    replacement = make_backend(OriginalBackend)
    backend = replacement(engine_config())
    monkeypatch.setattr(bridge,"prefill_paddle",lambda *a,**k: (["bts_prefill"],(129,)))
    args = (None,None,None,[],None,None,NS(layer_id=3),NS(prefill=True))
    assert backend.forward_mixed(*args) == ["bts_prefill","rope_and_cache","native_decode"]
    assert OriginalBackend(engine_config()).forward_mixed(*args)[0] == "native_prefill"
    assert OriginalBackend.forward_mixed is original_fn
    assert original_fn.__globals__["flash_attn_func"] is flash_attn_func
    args = (*args[:-1],NS(prefill=False))
    assert backend.forward_mixed(*args) == ["rope_and_cache","native_decode"]
    assert backend.bts_audit.data["prefill_calls"] == 1
    assert backend.bts_audit.data["native_decode_calls"] == 1
    def fail(*a,**k):
        raise RuntimeError("selector failed")
    monkeypatch.setattr(bridge,"prefill_paddle",fail)
    with pytest.raises(RuntimeError,match="selector failed"):
        backend.forward_mixed(*args[:-1],NS(prefill=True))
    from bts_fastdeploy.backend import _active
    assert _active.get() is None


def test_bts_backend_passes_validated_batch_limit_to_bridge(monkeypatch):
    from bts_fastdeploy import bridge

    observed = {}

    def packed_prefill(*args, **kwargs):
        observed.update(kwargs)
        return ["bts_prefill"], (17, 129, 0)

    backend = make_backend(OriginalBackend)(engine_config(max_num_seqs=16))
    monkeypatch.setattr(bridge, "prefill_paddle", packed_prefill)
    backend.forward_mixed(
        None, None, None, [], None, None, NS(layer_id=3), NS(prefill=True)
    )

    assert observed["max_num_seqs"] == 16
    assert backend.bts_audit.data["prefill_calls"] == 2
    assert backend.bts_audit.data["layers"] == {"3": 2}
    assert backend.bts_audit.data["last_lengths"] == [17, 129, 0]


def test_bts_backend_updates_mmap_audit(monkeypatch, tmp_path):
    from bts_fastdeploy import bridge
    from bts_fastdeploy.audit import read_audits

    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_FLUSH", "mmap")
    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_NUM_LAYERS", "4")
    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_DIR", str(tmp_path / "audits"))
    monkeypatch.setattr(
        bridge, "prefill_paddle", lambda *args, **kwargs: (["bts_prefill"], (254,))
    )
    backend = make_backend(OriginalBackend)(engine_config())
    backend.forward_mixed(
        None,
        None,
        None,
        [],
        None,
        None,
        NS(layer_id=3),
        NS(prefill=True),
    )
    saved = read_audits(tmp_path / "audits")[0]
    assert saved["layers"] == {"3": 1}


@pytest.mark.parametrize("prefill,identity_index", [(True,1), (False,0)])
def test_preapplied_rope_uses_identity_and_restores_metadata(
        monkeypatch, prefill, identity_index):
    from bts_fastdeploy import bridge
    replacement = make_backend(RopeAwareBackend)
    backend = replacement(engine_config())
    identity = object()
    original = object()
    monkeypatch.setattr(backend, "_get_identity_rotary_embs", lambda value: identity)
    monkeypatch.setattr(bridge,"prefill_paddle",lambda *a,**k: (["bts_prefill"],(129,)))
    meta = NS(prefill=prefill, rope_already_applied=True, rotary_embs=original)
    result = backend.forward_mixed(None,None,None,[],None,None,NS(layer_id=3),meta)
    assert result[identity_index] is identity
    assert meta.rotary_embs is original


def test_non_preapplied_rope_remains_unchanged(monkeypatch):
    replacement = make_backend(RopeAwareBackend)
    backend = replacement(engine_config())
    original = object()
    monkeypatch.setattr(
        backend,
        "_get_identity_rotary_embs",
        lambda value: pytest.fail("identity RoPE must not be constructed"),
    )
    meta = NS(prefill=False, rope_already_applied=False, rotary_embs=original)
    result = backend.forward_mixed(None,None,None,[],None,None,NS(layer_id=3),meta)
    assert result[0] is original
    assert meta.rotary_embs is original


def test_preapplied_rope_restores_metadata_on_prefill_error(monkeypatch):
    from bts_fastdeploy import bridge
    from bts_fastdeploy.backend import _active
    replacement = make_backend(RopeAwareBackend)
    backend = replacement(engine_config())
    identity = object()
    original = object()
    monkeypatch.setattr(backend, "_get_identity_rotary_embs", lambda value: identity)
    monkeypatch.setattr(
        bridge,
        "prefill_paddle",
        lambda *a,**k: (_ for _ in ()).throw(RuntimeError("selector failed")),
    )
    meta = NS(prefill=True, rope_already_applied=True, rotary_embs=original)
    with pytest.raises(RuntimeError,match="selector failed"):
        backend.forward_mixed(None,None,None,[],None,None,NS(layer_id=3),meta)
    assert meta.rotary_embs is original
    assert _active.get() is None


def test_identity_rope_values_and_cache_reuse():
    paddle = pytest.importorskip("paddle")
    replacement = make_backend(RopeAwareBackend)
    backend = replacement(engine_config())
    original = paddle.randn([2,1,3,1,4], dtype="float32")
    original_copy = original.clone()
    identity = backend._get_identity_rotary_embs(original)
    assert tuple(identity.shape) == tuple(original.shape)
    assert identity.dtype == original.dtype
    assert bool(paddle.all(identity[0] == 1.0).item())
    assert bool(paddle.all(identity[1] == 0.0).item())
    assert bool(paddle.all(original == original_copy).item())
    assert backend._get_identity_rotary_embs(original) is identity
    different_dtype = paddle.randn([2,1,3,1,4], dtype="float64")
    assert backend._get_identity_rotary_embs(different_dtype) is not identity


def test_dense_rope_compat_preserves_native_attention_and_restores_metadata(monkeypatch):
    replacement = make_dense_backend(RopeAwareBackend)
    backend = replacement(engine_config())
    identity = object()
    original = object()
    monkeypatch.setattr(backend, "_get_identity_rotary_embs", lambda value: identity)
    meta = NS(prefill=True, rope_already_applied=True, rotary_embs=original)

    result = backend.forward_mixed(
        None, None, None, [], None, None, NS(layer_id=3), meta
    )

    assert result == ["native_prefill", identity, "native_decode"]
    assert meta.rotary_embs is original
    assert replacement._bts_fastdeploy_mode == "dense_rope_compat"
    assert replacement._bts_fastdeploy_dense_compat is True
    assert replacement._bts_fastdeploy_adapter is False
    assert replacement._bts_native_flash_attn_preserved is True
    assert backend.bts_audit.data["native_flash_attention"]["preserved"] is True
    assert backend.bts_audit.data["rope_guard_layers"] == {"3": 1}
    assert RopeAwareBackend.forward_mixed.__globals__["flash_attn_func"] is flash_attn_func


def test_dense_backend_updates_mmap_audit(monkeypatch, tmp_path):
    from bts_fastdeploy.audit import read_audits

    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_FLUSH", "mmap")
    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_NUM_LAYERS", "4")
    monkeypatch.setenv("BTS_FASTDEPLOY_AUDIT_DIR", str(tmp_path / "audits"))
    backend = make_dense_backend(RopeAwareBackend)(engine_config())
    original = object()
    monkeypatch.setattr(backend, "_get_identity_rotary_embs", lambda value: object())
    backend.forward_mixed(
        None,
        None,
        None,
        [],
        None,
        None,
        NS(layer_id=3),
        NS(prefill=True, rope_already_applied=True, rotary_embs=original),
    )
    saved = read_audits(tmp_path / "audits")[0]
    assert saved["rope_guard_layers"] == {"3": 1}
    assert saved["native_flash_attention"]["preserved"] is True


def test_dense_rope_compat_restores_metadata_when_native_forward_raises(monkeypatch):
    replacement = make_dense_backend(ExplodingDenseBackend)
    backend = replacement(engine_config())
    identity = object()
    original = object()
    monkeypatch.setattr(backend, "_get_identity_rotary_embs", lambda value: identity)
    meta = NS(prefill=True, rope_already_applied=True, rotary_embs=original)

    with pytest.raises(RuntimeError, match="native dense failed"):
        backend.forward_mixed(
            None, None, None, [], None, None, NS(layer_id=7), meta
        )
    assert meta.rotary_embs is original


@pytest.mark.parametrize(
    "meta",
    [
        NS(prefill=True, rope_already_applied=False, rotary_embs=object()),
        NS(prefill=True, rotary_embs=object()),
        NS(prefill=True, rope_already_applied=True, rotary_embs=None),
    ],
)
def test_dense_rope_compat_non_preapplied_inputs_are_passthrough(monkeypatch, meta):
    replacement = make_dense_backend(RopeAwareBackend)
    backend = replacement(engine_config())
    original = meta.rotary_embs
    monkeypatch.setattr(
        backend,
        "_get_identity_rotary_embs",
        lambda value: pytest.fail("identity RoPE must not be constructed"),
    )
    backend.forward_mixed(None, None, None, [], None, None, NS(layer_id=3), meta)
    assert meta.rotary_embs is original
    assert backend.bts_audit.data["rope_guard_calls"] == 0


@pytest.mark.parametrize("section,key,value", [
    ("cache_config","enable_chunked_prefill",True),
    ("cache_config","enable_prefix_caching",True),
    ("graph_opt_config","use_cudagraph",True),
    ("parallel_config","tensor_parallel_size",2),
    ("model_config","model_type","deepseek_v3"),
    ("model_config","dtype","float16"),
    ("model_config","quantization_config",{"weight": "int8"}),
])
def test_unsupported_engine_modes_fail(section,key,value):
    cfg = engine_config()
    setattr(getattr(cfg,section),key,value)
    with pytest.raises(ValueError,match="unsupported"):
        validate_engine(cfg)


@pytest.mark.parametrize("max_num_seqs", [1, 2, 8, 16])
def test_validated_multi_request_batch_sizes_are_accepted(max_num_seqs):
    validate_engine(engine_config(max_num_seqs=max_num_seqs))


@pytest.mark.parametrize("max_num_seqs", [-1, 0, 17, True])
def test_unvalidated_batch_sizes_fail_closed(max_num_seqs):
    with pytest.raises(ValueError, match="max_num_seqs"):
        validate_engine(engine_config(max_num_seqs=max_num_seqs))


@pytest.mark.parametrize("hf_config", [
    NS(model_type="qwen3", head_dim=64),
    NS(model_type="qwen3", hidden_size=512, num_attention_heads=8),
])
def test_incompatible_head_dimension_fails_when_model_metadata_exposes_it(hf_config):
    cfg = engine_config()
    cfg.model_config.hf_config = hf_config
    with pytest.raises(ValueError, match="head_dim=64"):
        validate_engine(cfg)


def test_missing_head_dimension_metadata_is_not_guessed():
    validate_engine(engine_config())
