"""BTS Paddle/Torch interop on Paddle's current CUDA stream."""

from contextlib import contextmanager


@contextmanager
def shared_stream(tensor):
    import paddle
    import torch

    if tensor.place.is_gpu_place():
        index = tensor.place.gpu_device_id()
        stream = paddle.device.cuda.current_stream(index)
        external = torch.cuda.ExternalStream(stream.cuda_stream, device=index)
        with torch.cuda.device(index), torch.cuda.stream(external):
            if torch.cuda.current_stream(index).cuda_stream != stream.cuda_stream:
                raise RuntimeError("Paddle and Torch did not enter the same CUDA stream")
            yield
    elif tensor.place.is_cpu_place():
        yield
    else:
        raise ValueError("BTS bridge supports CPU reference tests and NVIDIA CUDA only")


def to_torch(tensor):
    import torch

    # Passing the producer object (rather than a pre-built capsule) lets the
    # standard DLPack protocol negotiate the consumer stream when these helpers
    # are also used outside ``shared_stream``.
    return torch.utils.dlpack.from_dlpack(tensor)


def to_paddle(tensor):
    import paddle

    return paddle.utils.dlpack.from_dlpack(tensor)


def prefill_paddle(q, k, v, cu_seqlens_q, cu_seqlens_k, *, config,
                   num_heads, kv_num_heads, head_dim, max_num_seqs,
                   causal=True, attn_mask_q=None, seq_lens_encoder=None,
                   seq_lens_decoder=None):
    from .pipeline import prefill_attention
    import paddle
    import torch

    if not causal or attn_mask_q is not None:
        raise ValueError("BTS requires ordinary causal attention without a custom attention mask")
    if q.place != k.place or q.place != v.place:
        raise ValueError("Q/K/V must share a Paddle device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V must share a Paddle dtype")
    with shared_stream(q), torch.inference_mode():
        tq, tk, tv = (to_torch(t) for t in (q, k, v))
        tq = tq.reshape(-1, num_heads, head_dim)
        tk = tk.reshape(-1, kv_num_heads, head_dim)
        tv = tv.reshape(-1, kv_num_heads, head_dim)
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("FastDeploy prefill requires cu_seqlens_q and cu_seqlens_k")
        # FastDeploy normally exposes one-dimensional boundaries, while some
        # metadata construction paths retain the original [N + 1, 1] shape.
        qs = to_torch(cu_seqlens_q).reshape(-1).tolist()
        ks = to_torch(cu_seqlens_k).reshape(-1).tolist()
        if qs != ks:
            raise ValueError("Cached prefixes and chunked prefill are not supported")
        sequence_count = len(qs) - 1
        if sequence_count < 1 or sequence_count > max_num_seqs:
            raise ValueError(
                f"Packed prefill contains {sequence_count} sequence slots, "
                f"but max_num_seqs is {max_num_seqs}"
            )
        if len(qs) < 2 or qs[0] != 0 or qs[-1] != tq.shape[0] or ks[-1] != tk.shape[0]:
            raise ValueError("Invalid cumulative sequence lengths")
        if tv.shape != tk.shape or any(b < a for a, b in zip(qs, qs[1:])):
            raise ValueError("Invalid K/V shapes or nonmonotonic sequence lengths")
        if (seq_lens_encoder is None) != (seq_lens_decoder is None):
            raise ValueError("FastDeploy must provide both encoder and decoder lengths")
        if seq_lens_encoder is None:
            if max_num_seqs != 1 or sequence_count != 1:
                raise ValueError(
                    "Multi-request prefill requires FastDeploy encoder/decoder state"
                )
            encoder_lengths = [b - a for a, b in zip(qs, qs[1:])]
            decoder_lengths = [0] * sequence_count
        else:
            encoder_lengths = to_torch(seq_lens_encoder).reshape(-1).tolist()
            decoder_lengths = to_torch(seq_lens_decoder).reshape(-1).tolist()
            if (
                len(encoder_lengths) < sequence_count
                or len(decoder_lengths) < sequence_count
            ):
                raise ValueError("Sequence state does not match cumulative sequence lengths")
            # FastDeploy keeps max_num_seqs-sized state buffers but trims
            # cu_seqlens to the highest currently occupied slot.
            encoder_lengths = encoder_lengths[:sequence_count]
            decoder_lengths = decoder_lengths[:sequence_count]
        # Paddle owns the output allocation consumed by FastDeploy after this
        # function returns. Torch only borrows its storage while executing BTS.
        # This avoids handing a Torch-caching-allocator allocation to asynchronous
        # Paddle kernels whose eventual consumer stream is outside this bridge.
        paddle_output = paddle.empty_like(q).reshape(tq.shape)
        output = to_torch(paddle_output)
        audited_lengths = []
        for start, end, encoder_length, decoder_length in zip(
            qs, qs[1:], encoder_lengths, decoder_lengths
        ):
            length = end - start
            if encoder_length > 0:
                if decoder_length != 0 or length != encoder_length:
                    raise ValueError("Only complete, uncached prefill sequences are supported")
                audited_lengths.append(length)
            elif decoder_length > 0:
                if length != 1:
                    raise ValueError("Native decode sequence slots must contain one token")
                # FastDeploy's native merge overwrites this slot before the
                # packed output is returned. Avoid a tiny per-layer zero kernel.
                audited_lengths.append(0)
                continue
            else:
                if length != 0:
                    raise ValueError("Inactive sequence slots must be empty")
                audited_lengths.append(0)
                continue
            inputs = [t[start:end].transpose(0, 1).unsqueeze(0) for t in (tq, tk, tv)]
            result = prefill_attention(*inputs, config=config)
            output[start:end].copy_(result.squeeze(0).transpose(0, 1))
        if output.is_cuda:
            # Defensive completion barrier at the framework boundary.  The
            # shared-stream contract should already order the Paddle consumer,
            # but completing the producer here also rules out hidden allocator
            # or interop lifetime effects while validating this adapter.
            torch.cuda.current_stream(output.device).synchronize()
        return paddle_output, tuple(audited_lengths)
