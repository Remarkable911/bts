"""BTS full-prefill attention layered over vLLM 0.17.0 FlashAttention.

vLLM performs the native FlashAttention KV-cache update before calling
``forward`` because this backend inherits
``forward_includes_kv_cache_update = False``.  This implementation therefore
changes only attention computation: complete, uncached prompt segments use
BTS, while decode and prefix-extension segments use the parent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .audit import AUDIT
from .compat import require_supported_vllm


require_supported_vllm()

from vllm.v1.attention.backend import (  # noqa: E402
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec  # noqa: E402


EXPECTED_QUERY_HEADS = 32
EXPECTED_KV_HEADS = 8
EXPECTED_HEAD_SIZE = 128
SUPPORTED_CACHE_DTYPES = ("auto", "bfloat16")


@dataclass
class BTSFlashAttentionMetadata(FlashAttentionMetadata):
    """FlashAttention metadata plus immutable CPU request boundaries."""

    request_query_start_loc: tuple[int, ...] = ()
    request_seq_lens: tuple[int, ...] = ()
    full_prefill_request_indices: tuple[int, ...] = ()


def _cpu_int_tuple(tensor: torch.Tensor, count: int) -> tuple[int, ...]:
    values = tensor[:count]
    if values.device.type != "cpu":
        values = values.detach().cpu()
    return tuple(int(value) for value in values.tolist())


class BTSFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Build metadata that can split full prefill from decode/extend."""

    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        require_supported_vllm()
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        scheduler = vllm_config.scheduler_config
        cache = vllm_config.cache_config
        errors: list[str] = []

        if model is None or model.dtype != torch.bfloat16:
            errors.append("model dtype must be torch.bfloat16")
        if kv_cache_spec.dtype != torch.bfloat16:
            errors.append("KV cache storage dtype must be torch.bfloat16")
        if str(cache.cache_dtype) not in SUPPORTED_CACHE_DTYPES:
            errors.append("FP8/quantized KV cache is not supported")
        if scheduler.enable_chunked_prefill:
            errors.append(
                "chunked prefill must be disabled so a zero-context segment is a "
                "complete prompt"
            )
        if parallel.prefill_context_parallel_size != 1:
            errors.append("prefill context parallelism is not supported")
        if parallel.decode_context_parallel_size != 1:
            errors.append("decode context parallelism is not supported")
        if parallel.enable_dbo or parallel.ubatch_size > 1:
            errors.append("dual-batch overlap/microbatching is not supported")
        if getattr(kv_cache_spec, "sliding_window", None) is not None:
            errors.append("sliding-window KV cache specs are not supported")
        if getattr(kv_cache_spec, "attention_chunk_size", None) is not None:
            errors.append("chunked local-attention specs are not supported")
        value_head_size = getattr(
            kv_cache_spec, "head_size_v", kv_cache_spec.head_size
        )
        if value_head_size != EXPECTED_HEAD_SIZE:
            errors.append("value head_size must be 128")

        if model is not None:
            num_q_heads = model.get_num_attention_heads(parallel)
            num_kv_heads = model.get_num_kv_heads(parallel)
            head_size = model.get_head_size()
            if (num_q_heads, num_kv_heads, head_size) != (
                EXPECTED_QUERY_HEADS,
                EXPECTED_KV_HEADS,
                EXPECTED_HEAD_SIZE,
            ):
                errors.append(
                    "BTS requires the per-worker 32Q/8KV/D128 shape; "
                    f"found {num_q_heads}Q/{num_kv_heads}KV/D{head_size}"
                )
        if errors:
            raise ValueError("unsupported BTS vLLM configuration: " + "; ".join(errors))

        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The custom forward is eager and creates request-dependent sparse layouts.
        self.aot_schedule = False
        self.use_full_cuda_graph = False
        AUDIT.record(metadata_builders=1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BTSFlashAttentionMetadata:
        del common_prefix_len, fast_build
        # Cascade is an optimization, not a semantic requirement.  Keeping it
        # off makes arbitrary dense fallback subsets self-contained.
        base = super().build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )
        num_reqs = common_attn_metadata.num_reqs
        starts = _cpu_int_tuple(common_attn_metadata.query_start_loc_cpu, num_reqs + 1)
        seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = common_attn_metadata.seq_lens
        seq_lens = _cpu_int_tuple(seq_lens_cpu, num_reqs)

        if len(starts) != len(seq_lens) + 1 or starts[0] != 0:
            raise RuntimeError("invalid vLLM request boundaries for BTS metadata")
        query_lens = tuple(end - start for start, end in zip(starts, starts[1:]))
        if any(length < 0 for length in query_lens):
            raise RuntimeError("vLLM query_start_loc must be monotonic")
        contexts = tuple(
            seq_len - query_len
            for seq_len, query_len in zip(seq_lens, query_lens)
        )
        if any(context < 0 for context in contexts):
            raise RuntimeError("vLLM sequence length is smaller than its query length")
        full_prefill = tuple(
            index
            for index, (query_len, context) in enumerate(zip(query_lens, contexts))
            if query_len > 0 and context == 0
        )
        return BTSFlashAttentionMetadata(
            **vars(base),
            request_query_start_loc=starts,
            request_seq_lens=seq_lens,
            full_prefill_request_indices=full_prefill,
        )


def _gather_token_segments(
    tensor: torch.Tensor,
    segments: Sequence[tuple[int, int]],
) -> torch.Tensor:
    if len(segments) == 1:
        start, end = segments[0]
        return tensor[start:end]
    return torch.cat([tensor[start:end] for start, end in segments], dim=0)


def _scatter_token_segments(
    destination: torch.Tensor,
    source: torch.Tensor,
    segments: Sequence[tuple[int, int]],
) -> None:
    source_start = 0
    for start, end in segments:
        count = end - start
        destination[start:end].copy_(source[source_start : source_start + count])
        source_start += count
    if source_start != source.shape[0]:
        raise RuntimeError("dense fallback output did not match request segments")


class BTSFlashAttentionImpl(FlashAttentionImpl):
    """Use BTS for full prompt segments and FlashAttention for everything else."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        errors: list[str] = []
        if (num_heads, num_kv_heads, head_size) != (
            EXPECTED_QUERY_HEADS,
            EXPECTED_KV_HEADS,
            EXPECTED_HEAD_SIZE,
        ):
            errors.append(
                "BTS requires 32 query heads, 8 KV heads, and head_size=128"
            )
        if not math.isclose(
            float(scale), head_size**-0.5, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("non-standard attention scaling is not supported")
        if alibi_slopes is not None:
            errors.append("ALiBi is not supported")
        if sliding_window is not None:
            errors.append("sliding-window attention is not supported")
        if str(kv_cache_dtype) not in SUPPORTED_CACHE_DTYPES:
            errors.append("FP8/quantized KV cache is not supported")
        if logits_soft_cap not in (None, 0, 0.0):
            errors.append("logits soft-cap is not supported")
        if attn_type != AttentionType.DECODER:
            errors.append("only decoder self-attention is supported")
        if kv_sharing_target_layer_name is not None:
            errors.append("cross-layer KV-cache sharing is not supported")
        if sinks is not None:
            errors.append("attention sinks are not supported")
        if errors:
            raise ValueError("unsupported BTS attention layer: " + "; ".join(errors))

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )
        if self.vllm_flash_attn_version != 2:
            raise ValueError(
                "BTS vLLM requires native FlashAttention 2 for dense/decode "
                f"fallback; found version {self.vllm_flash_attn_version!r}"
            )
        AUDIT.record(impl_instances=1)

    def _validate_tensors(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        expected_q_tail = (EXPECTED_QUERY_HEADS, EXPECTED_HEAD_SIZE)
        expected_kv_tail = (EXPECTED_KV_HEADS, EXPECTED_HEAD_SIZE)
        if query.ndim != 3 or tuple(query.shape[1:]) != expected_q_tail:
            raise ValueError(
                "BTS vLLM query must have shape [tokens, 32, 128]; "
                f"got {tuple(query.shape)}"
            )
        if key.ndim != 3 or tuple(key.shape[1:]) != expected_kv_tail:
            raise ValueError(
                f"BTS vLLM key must have shape [tokens, 8, 128]; got {tuple(key.shape)}"
            )
        if value.shape != key.shape:
            raise ValueError("BTS vLLM key/value shapes must match")
        if output.ndim != 3 or tuple(output.shape[1:]) != expected_q_tail:
            raise ValueError(
                "BTS vLLM output must have shape [tokens, 32, 128]; "
                f"got {tuple(output.shape)}"
            )
        if not (query.shape[0] == key.shape[0] == value.shape[0] == output.shape[0]):
            raise ValueError("BTS vLLM Q/K/V/output token capacities must match")
        if any(
            tensor.dtype != torch.bfloat16 for tensor in (query, key, value, output)
        ):
            raise TypeError("BTS vLLM Q/K/V/output must all be BF16")
        if not (query.device == key.device == value.device == output.device):
            raise ValueError("BTS vLLM Q/K/V/output must share one device")

    @staticmethod
    def _segments(
        metadata: BTSFlashAttentionMetadata,
        request_indices: Sequence[int],
    ) -> tuple[tuple[int, int], ...]:
        starts = metadata.request_query_start_loc
        return tuple((starts[index], starts[index + 1]) for index in request_indices)

    @staticmethod
    def _dense_metadata(
        metadata: BTSFlashAttentionMetadata,
        request_indices: Sequence[int],
        segments: Sequence[tuple[int, int]],
    ) -> FlashAttentionMetadata:
        query_lens = [end - start for start, end in segments]
        starts = [0]
        for length in query_lens:
            starts.append(starts[-1] + length)
        device_indices = torch.tensor(
            request_indices,
            dtype=torch.long,
            device=metadata.seq_lens.device,
        )
        seq_lens = metadata.seq_lens.index_select(0, device_indices)
        block_table = metadata.block_table.index_select(0, device_indices)
        slot_mapping = _gather_token_segments(metadata.slot_mapping, segments)
        return FlashAttentionMetadata(
            num_actual_tokens=starts[-1],
            max_query_len=max(query_lens),
            query_start_loc=torch.tensor(
                starts,
                dtype=metadata.query_start_loc.dtype,
                device=metadata.query_start_loc.device,
            ),
            max_seq_len=max(
                metadata.request_seq_lens[index] for index in request_indices
            ),
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            max_dcp_context_kv_len=None,
            dcp_context_kv_lens=None,
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
            max_num_splits=0,
            causal=True,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BTSFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            raise ValueError("BTS vLLM backend requires vLLM's output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("BTS vLLM does not support fused output quantization")
        if attn_metadata is None:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
            )
        if not isinstance(attn_metadata, BTSFlashAttentionMetadata):
            raise TypeError("BTS vLLM backend received foreign attention metadata")
        if attn_metadata.use_cascade or attn_metadata.common_prefix_len:
            raise ValueError("cascade/prefix attention metadata is not supported")
        if attn_metadata.causal is not True:
            raise ValueError("BTS vLLM backend requires causal decoder attention")
        if attn_metadata.dcp_context_kv_lens is not None:
            raise ValueError("decode context parallel metadata is not supported")

        self._validate_tensors(query, key, value, output)
        num_actual_tokens = attn_metadata.num_actual_tokens
        if attn_metadata.request_query_start_loc[-1] != num_actual_tokens:
            raise RuntimeError("BTS metadata token boundary mismatch")

        sparse_indices = attn_metadata.full_prefill_request_indices
        sparse_set = set(sparse_indices)
        request_count = len(attn_metadata.request_seq_lens)
        dense_indices = tuple(
            index for index in range(request_count) if index not in sparse_set
        )

        if not sparse_indices:
            AUDIT.record(
                dense_fallback_calls=1,
                dense_fallback_segments=len(dense_indices),
            )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
            )

        if dense_indices:
            dense_segments = self._segments(attn_metadata, dense_indices)
            dense_query = _gather_token_segments(query, dense_segments)
            dense_key = _gather_token_segments(key, dense_segments)
            dense_value = _gather_token_segments(value, dense_segments)
            dense_output = torch.empty_like(dense_query)
            dense_metadata = self._dense_metadata(
                attn_metadata, dense_indices, dense_segments
            )
            super().forward(
                layer,
                dense_query,
                dense_key,
                dense_value,
                kv_cache,
                dense_metadata,
                output=dense_output,
            )
            _scatter_token_segments(output, dense_output, dense_segments)

        from bts import bts_prefill_attention

        sparse_segments = self._segments(attn_metadata, sparse_indices)
        for start, end in sparse_segments:
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            sparse_output = bts_prefill_attention(q, k, v)
            output[start:end].copy_(sparse_output.squeeze(0).transpose(0, 1))

        AUDIT.record(
            sparse_calls=1,
            sparse_segments=len(sparse_segments),
            dense_fallback_calls=1 if dense_indices else 0,
            dense_fallback_segments=len(dense_indices),
            mixed_calls=1 if dense_indices else 0,
        )
        return output


class BTSFlashAttentionBackend(FlashAttentionBackend):
    """The class registered as ``AttentionBackendEnum.CUSTOM``."""

    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        # Attention.__init__ indexes AttentionBackendEnum with this value.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[BTSFlashAttentionImpl]:
        return BTSFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[BTSFlashAttentionMetadataBuilder]:
        return BTSFlashAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == EXPECTED_HEAD_SIZE

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return kv_cache_dtype is None or str(kv_cache_dtype) in SUPPORTED_CACHE_DTYPES

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER
