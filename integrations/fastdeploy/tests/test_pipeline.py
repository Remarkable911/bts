from dataclasses import FrozenInstanceError, asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from bts import describe, bts_prefill_attention
from bts_fastdeploy.config import config_digest, load_config, validate_algorithm
from bts_fastdeploy.pipeline import prefill_attention


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def qkv(length=17, dtype=torch.bfloat16):
    torch.manual_seed(31)
    return (
        torch.randn(1, 2, length, 128, dtype=dtype),
        torch.randn(1, 1, length, 128, dtype=dtype),
        torch.randn(1, 1, length, 128, dtype=dtype),
    )


def test_default_config_is_exact_public_frozen_description():
    config = load_config()
    assert asdict(config) == describe()
    assert config_digest(config) == config_digest(load_config())
    with pytest.raises(FrozenInstanceError):
        config.token_ratio = 0.5


def test_matching_partial_and_saved_config_resolve_to_one_canonical_config(tmp_path):
    canonical = load_config()
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({
        "token": {
            "token_select": "block_count",
            "token_ratio": 0.4,
            "dense_tail_tokens": 256,
        }
    }))
    assert load_config(partial) == canonical

    saved = tmp_path / "runtime.json"
    saved.write_text(json.dumps({"runtime_config": asdict(canonical)}))
    assert load_config(saved) == canonical
    assert config_digest(load_config(saved)) == config_digest(canonical)


@pytest.mark.parametrize("data,match", [
    ({"token": {"token_select": "topp"}}, "token.token_select"),
    ({"token": {"token_ratio": 0.5}}, "token.token_ratio"),
    ({"token": {"score_mode": "qk_last"}}, "token.score_mode"),
    ({"token_sparse": False}, "token_sparse"),
    ({"block": {"attention_backend": "flex"}}, "attention_backend"),
    ({"token_min_seq_len": 8192}, "token_min_seq_len"),
])
def test_research_algorithm_switches_are_rejected(tmp_path, data, match):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_modified_config_object_is_rejected():
    with pytest.raises(ValueError, match="token_ratio"):
        validate_algorithm(replace(load_config(), token_ratio=0.5))


def test_pipeline_dispatches_once_through_public_api_without_mutation():
    tensors = qkv()
    snapshots = [tensor.clone() for tensor in tensors]
    sentinel = torch.empty_like(tensors[0])
    with patch("bts.bts_prefill_attention", return_value=sentinel) as public:
        actual = prefill_attention(*tensors, config=load_config())
    assert actual is sentinel
    assert public.call_count == 1
    for passed, original in zip(public.call_args.args, tensors):
        assert passed is original
    for tensor, snapshot in zip(tensors, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


def test_pipeline_output_equals_frozen_public_api_on_cpu_bf16():
    tensors = qkv()
    expected = bts_prefill_attention(*tensors)
    actual = prefill_attention(*tensors, config=load_config())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pipeline_rejects_non_frozen_dtype_and_cached_prefix():
    with pytest.raises(TypeError, match="BF16"):
        prefill_attention(*qkv(dtype=torch.float16), config=load_config())
    query, key, value = qkv()
    key = torch.cat([key, key], dim=2)
    value = torch.cat([value, value], dim=2)
    with pytest.raises(ValueError, match="square"):
        prefill_attention(query, key, value, config=load_config())


def test_import_and_disabled_plugin_have_no_framework_side_effects():
    adapter_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(adapter_src), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    )
    code = (
        "import sys, os; os.environ.pop('BTS_FASTDEPLOY_ENABLE', None); "
        "os.environ['FD_DISABLE_CHUNKED_PREFILL']='0'; "
        "import bts_fastdeploy; import bts_fastdeploy.config; "
        "from bts_fastdeploy.plugin import register; register(); "
        "assert os.environ['FD_DISABLE_CHUNKED_PREFILL']=='0'; "
        "assert not any(x in sys.modules for x in ('torch','paddle','bts','fastdeploy'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, env=env)
