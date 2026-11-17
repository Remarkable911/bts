import gc
import os
from contextlib import nullcontext

import pytest
import torch

from bts_fastdeploy.bridge import prefill_paddle, shared_stream, to_paddle, to_torch
from bts_fastdeploy.config import load_config
from bts_fastdeploy.pipeline import prefill_attention


@pytest.mark.parametrize("device", ["cpu", "gpu", "gpu_stream"])
def test_paddle_roundtrip_and_attention(device):
    gpu = device.startswith("gpu")
    if gpu and os.environ.get("BTS_FASTDEPLOY_TEST_CUDA") != "1":
        pytest.skip("GPU validation requires an explicitly allocated idle device")
    paddle = pytest.importorskip("paddle")
    paddle.set_device("gpu" if gpu else "cpu")
    torch_device = "cuda:0" if gpu else "cpu"
    dtype = torch.bfloat16
    torch.manual_seed(44)
    config = load_config()
    inputs = [torch.randn(129,h,128,device=torch_device,dtype=dtype) for h in (2,1,1)]
    # Ensure producers finish on the default Torch stream before handing the
    # fixtures to Paddle. Actual runtime uses one shared stream in both directions.
    if gpu:
        torch.cuda.synchronize()
    pd_inputs = [to_paddle(t) for t in inputs]
    stream_guard = paddle.device.cuda.stream_guard(paddle.device.cuda.Stream()) if device == "gpu_stream" else nullcontext()
    with stream_guard, shared_stream(pd_inputs[0]):
        lengths = paddle.to_tensor([0,129],dtype="int32")
        view = to_torch(pd_inputs[0])
        assert view.data_ptr() == inputs[0].data_ptr()
        expected = prefill_attention(*[t.transpose(0,1).unsqueeze(0) for t in inputs],config=config)
        output, seqs = prefill_paddle(*pd_inputs,lengths,lengths,config=config,
                                      num_heads=2,kv_num_heads=1,head_dim=128,
                                      max_num_seqs=1)
        actual = to_torch(output)
        del output
        gc.collect()
        torch.testing.assert_close(actual,expected.squeeze(0).transpose(0,1),rtol=0,atol=0)
    assert seqs == (129,)


def test_non_square_bridge_fails():
    paddle = pytest.importorskip("paddle")
    paddle.set_device("cpu")
    q = paddle.zeros([2,2,128])
    k = paddle.zeros([3,1,128])
    with pytest.raises(ValueError,match="Cached prefixes"):
        prefill_paddle(q,k,k,paddle.to_tensor([0,2]),paddle.to_tensor([0,3]),
                       config=load_config(),num_heads=2,kv_num_heads=1,head_dim=128,
                       max_num_seqs=1)


def test_ragged_packed_prefill_matches_independent_sequences():
    paddle = pytest.importorskip("paddle")
    paddle.set_device("cpu")
    torch.manual_seed(45)
    config = load_config()
    sequence_lengths = (17, 33)
    total = sum(sequence_lengths)
    inputs = [torch.randn(total,h,128,dtype=torch.bfloat16) for h in (2,1,1)]
    pd_inputs = [to_paddle(tensor) for tensor in inputs]
    boundaries = paddle.to_tensor([[0], [17], [50]], dtype="int32")

    expected = []
    start = 0
    for length in sequence_lengths:
        end = start + length
        expected.append(
            prefill_attention(
                *[
                    tensor[start:end].transpose(0, 1).unsqueeze(0)
                    for tensor in inputs
                ],
                config=config,
            ).squeeze(0).transpose(0, 1)
        )
        start = end

    output, observed_lengths = prefill_paddle(
        *pd_inputs,
        boundaries,
        boundaries,
        config=config,
        num_heads=2,
        kv_num_heads=1,
        head_dim=128,
        max_num_seqs=16,
        seq_lens_encoder=paddle.to_tensor(
            [*sequence_lengths, *([0] * 14)], dtype="int32"
        ),
        seq_lens_decoder=paddle.zeros([16], dtype="int32"),
    )

    torch.testing.assert_close(
        to_torch(output), torch.cat(expected), rtol=0, atol=0
    )
    assert observed_lengths == sequence_lengths


def test_sixteen_slots_preserve_boundaries_and_skip_empty_slots(monkeypatch):
    from bts_fastdeploy import pipeline

    paddle = pytest.importorskip("paddle")
    paddle.set_device("cpu")
    lengths = (2, 0, 1, 3, 0, 2, 1, 0, 4, 1, 2, 0, 3, 1, 0, 2)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    q = paddle.zeros([boundaries[-1], 2, 128], dtype="float32")
    k = paddle.zeros([boundaries[-1], 1, 128], dtype="float32")
    calls = []

    def marked_prefill(query, key, value, *, config):
        calls.append(query.shape[2])
        return torch.full_like(query, len(calls))

    monkeypatch.setattr(pipeline, "prefill_attention", marked_prefill)
    cumulative = paddle.to_tensor(boundaries, dtype="int32").reshape([-1, 1])
    sequence_state = paddle.to_tensor(lengths, dtype="int32").reshape([-1, 1])
    zeros = paddle.zeros_like(sequence_state)
    output, observed_lengths = prefill_paddle(
        q,
        k,
        k,
        cumulative,
        cumulative,
        config=load_config(),
        num_heads=2,
        kv_num_heads=1,
        head_dim=128,
        max_num_seqs=16,
        seq_lens_encoder=sequence_state,
        seq_lens_decoder=zeros,
    )

    actual = to_torch(output)
    start = 0
    call_index = 0
    for length in lengths:
        if length:
            call_index += 1
            assert torch.all(actual[start : start + length] == call_index)
        start += length
    assert calls == [length for length in lengths if length]
    assert observed_lengths == lengths


def test_mixed_batch_runs_bts_only_for_prefill_slots(monkeypatch):
    from bts_fastdeploy import pipeline

    paddle = pytest.importorskip("paddle")
    paddle.set_device("cpu")
    q = paddle.zeros([6, 2, 128], dtype="float32")
    k = paddle.zeros([6, 1, 128], dtype="float32")
    calls = []

    def marked_prefill(query, key, value, *, config):
        calls.append(query.shape[2])
        return torch.full_like(query, 10 * len(calls))

    monkeypatch.setattr(pipeline, "prefill_attention", marked_prefill)
    cumulative = paddle.to_tensor([0, 2, 3, 6], dtype="int32")
    output, observed_lengths = prefill_paddle(
        q,
        k,
        k,
        cumulative,
        cumulative,
        config=load_config(),
        num_heads=2,
        kv_num_heads=1,
        head_dim=128,
        max_num_seqs=16,
        seq_lens_encoder=paddle.to_tensor([2, 0, 3], dtype="int32"),
        seq_lens_decoder=paddle.to_tensor([0, 100, 0], dtype="int32"),
    )

    actual = to_torch(output)
    assert calls == [2, 3]
    assert torch.all(actual[:2] == 10)
    assert torch.all(actual[3:] == 20)
    assert observed_lengths == (2, 0, 3)


def test_packed_sequence_count_cannot_exceed_engine_limit():
    paddle = pytest.importorskip("paddle")
    paddle.set_device("cpu")
    q = paddle.zeros([2, 2, 128])
    k = paddle.zeros([2, 1, 128])
    cumulative = paddle.to_tensor([0, 1, 2], dtype="int32")
    with pytest.raises(ValueError, match="max_num_seqs"):
        prefill_paddle(
            q,
            k,
            k,
            cumulative,
            cumulative,
            config=load_config(),
            num_heads=2,
            kv_num_heads=1,
            head_dim=128,
            max_num_seqs=1,
        )
