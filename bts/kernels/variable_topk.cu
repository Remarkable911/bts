// Project-owned exact per-row variable Top-K selector.
//
// Each row is handled by one cooperative thread block. Four 8-bit radix
// passes find the K-th largest IEEE-754 key without sorting the row. A final
// pass emits every larger key plus the required number of boundary ties.
// This implementation uses only CUDA and PyTorch extension interfaces.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <optional>

// Validated output-width bound; keep in sync with variable_topk.py.
constexpr int64_t kMaxTopK = 65536;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRadixBins = 256;
constexpr int kKeyCacheCapacity = 8192;

__device__ __forceinline__ uint32_t ordered_float_key(float value) {
  // NaNs are outside the selector contract. Treat them as smaller than -inf
  // defensively so they cannot displace a valid score.
  if (value != value) {
    return 0;
  }
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template <bool WriteValues, bool Deterministic>
__global__ void radix_topk_rows(const float* __restrict__ scores,
                                const int* __restrict__ budgets,
                                int* __restrict__ output_indices,
                                float* __restrict__ output_values,
                                int rows,
                                int columns,
                                int output_width) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int k = max(0, min(budgets[row], output_width));
  if (k <= 0) {
    return;
  }

  __shared__ int histogram[kRadixBins];
  __shared__ uint32_t key_cache[kKeyCacheCapacity];
  __shared__ uint32_t prefix;
  __shared__ uint32_t prefix_mask;
  __shared__ uint32_t threshold;
  __shared__ int remaining_rank;
  __shared__ int candidate_count;
  __shared__ int cache_size;
  __shared__ int cache_write_count;
  __shared__ int cache_valid;
  __shared__ int fill_cache;
  __shared__ int greater_write_count;
  __shared__ int tie_write_count;
  // Used only by the Deterministic specialization. The default specialization
  // retains the existing atomic emit path and its throughput characteristics.
  __shared__ int warp_greater_counts[kWarps];
  __shared__ int warp_tie_counts[kWarps];
  __shared__ int warp_greater_bases[kWarps];
  __shared__ int warp_tie_bases[kWarps];

  if (tid == 0) {
    prefix = 0;
    prefix_mask = 0;
    remaining_rank = k;
    candidate_count = columns;
    cache_size = 0;
    cache_valid = 0;
  }
  __syncthreads();

  const float* row_scores = scores + static_cast<int64_t>(row) * columns;
  #pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    histogram[tid] = 0;
    if (tid == 0) {
      fill_cache = pass < 3 && !cache_valid &&
                   candidate_count <= kKeyCacheCapacity;
      cache_write_count = 0;
    }
    __syncthreads();

    const int shift = 24 - pass * 8;
    const uint32_t active_prefix = prefix;
    const uint32_t active_mask = prefix_mask;
    if (cache_valid) {
      for (int index = tid; index < cache_size; index += kThreads) {
        const uint32_t key = key_cache[index];
        if ((key & active_mask) == active_prefix) {
          atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
        }
      }
    } else {
      for (int column = tid; column < columns; column += kThreads) {
        const uint32_t key = ordered_float_key(row_scores[column]);
        if ((key & active_mask) != active_prefix) {
          continue;
        }
        atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
        if (fill_cache) {
          const int slot = atomicAdd(&cache_write_count, 1);
          key_cache[slot] = key;
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      if (fill_cache) {
        cache_size = cache_write_count;
        cache_valid = 1;
      }
      int rank = remaining_rank;
      int selected_digit = 0;
      for (int digit = kRadixBins - 1; digit >= 0; --digit) {
        const int count = histogram[digit];
        if (rank > count) {
          rank -= count;
        } else {
          selected_digit = digit;
          break;
        }
      }
      prefix |= static_cast<uint32_t>(selected_digit) << shift;
      prefix_mask |= 0xffu << shift;
      remaining_rank = rank;
      candidate_count = histogram[selected_digit];
    }
    __syncthreads();
  }

  if (tid == 0) {
    threshold = prefix;
    greater_write_count = 0;
    tie_write_count = 0;
  }
  __syncthreads();

  int* row_indices = output_indices + static_cast<int64_t>(row) * output_width;
  float* row_values = WriteValues
      ? output_values + static_cast<int64_t>(row) * output_width
      : nullptr;
  constexpr unsigned kFullWarpMask = 0xffffffffu;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int tie_target = remaining_rank;
  const int greater_target = k - tie_target;
  if constexpr (Deterministic) {
    // Give each warp one contiguous candidate-index segment. Lanes scan that
    // segment in coalesced 32-item tiles, while a fixed warp-order prefix makes
    // output order and the boundary-tie subset stable. The two-pass emit needs
    // only two block barriers for the entire row.
    const int segment_size = (columns + kWarps - 1) / kWarps;
    const int segment_begin = warp * segment_size;
    const int segment_end = min(segment_begin + segment_size, columns);
    int greater_count = 0;
    int tie_count = 0;
    for (int base = segment_begin; base < segment_end; base += 32) {
      const int column = base + lane;
      uint32_t key = 0;
      if (column < segment_end) {
        key = ordered_float_key(row_scores[column]);
      }
      const unsigned selected_greater = __ballot_sync(
          kFullWarpMask, column < segment_end && key > threshold);
      const unsigned selected_tie = __ballot_sync(
          kFullWarpMask, column < segment_end && key == threshold);
      if (lane == 0) {
        greater_count += __popc(selected_greater);
        tie_count += __popc(selected_tie);
      }
    }
    if (lane == 0) {
      warp_greater_counts[warp] = greater_count;
      warp_tie_counts[warp] = tie_count;
    }
    __syncthreads();

    if (tid == 0) {
      int next_greater = 0;
      int next_tie = 0;
      #pragma unroll
      for (int scan_warp = 0; scan_warp < kWarps; ++scan_warp) {
        warp_greater_bases[scan_warp] = next_greater;
        warp_tie_bases[scan_warp] = next_tie;
        next_greater += warp_greater_counts[scan_warp];
        next_tie += warp_tie_counts[scan_warp];
      }
    }
    __syncthreads();

    int greater_slot = warp_greater_bases[warp];
    int tie_slot = warp_tie_bases[warp];
    for (int base = segment_begin; base < segment_end; base += 32) {
      const int column = base + lane;
      float value = 0;
      uint32_t key = 0;
      if (column < segment_end) {
        value = row_scores[column];
        key = ordered_float_key(value);
      }
      const bool greater = column < segment_end && key > threshold;
      const bool tie = column < segment_end && key == threshold;
      const unsigned selected_greater = __ballot_sync(kFullWarpMask, greater);
      const unsigned selected_tie = __ballot_sync(kFullWarpMask, tie);
      const unsigned lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
      const int lane_greater_slot = greater_slot +
          __popc(selected_greater & lower_lanes);
      if (greater && lane_greater_slot < greater_target) {
        row_indices[lane_greater_slot] = column;
        if constexpr (WriteValues) {
          row_values[lane_greater_slot] = value;
        }
      }
      const int lane_tie_slot = tie_slot + __popc(selected_tie & lower_lanes);
      if (tie && lane_tie_slot < tie_target) {
        const int slot = greater_target + lane_tie_slot;
        row_indices[slot] = column;
        if constexpr (WriteValues) {
          row_values[slot] = value;
        }
      }
      greater_slot += __popc(selected_greater);
      tie_slot += __popc(selected_tie);
    }
  } else {
    for (int base = 0; base < columns; base += kThreads) {
      const int column = base + tid;
      float value = 0;
      uint32_t key = 0;
      if (column < columns) {
        value = row_scores[column];
        key = ordered_float_key(value);
      }
      const bool greater = column < columns && key > threshold;
      const unsigned selected_greater = __ballot_sync(kFullWarpMask, greater);
      const int greater_count = __popc(selected_greater);
      int warp_base = 0;
      if (lane == 0 && greater_count != 0) {
        warp_base = atomicAdd(&greater_write_count, greater_count);
      }
      warp_base = __shfl_sync(kFullWarpMask, warp_base, 0);
      const unsigned lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
      const int greater_slot =
          warp_base + __popc(selected_greater & lower_lanes);
      if (greater && greater_slot < greater_target) {
        row_indices[greater_slot] = column;
        if constexpr (WriteValues) {
          row_values[greater_slot] = value;
        }
      }

      const bool tie = column < columns && key == threshold;
      const unsigned selected_tie = __ballot_sync(kFullWarpMask, tie);
      const int tie_count = __popc(selected_tie);
      warp_base = 0;
      if (lane == 0 && tie_count != 0) {
        warp_base = atomicAdd(&tie_write_count, tie_count);
      }
      warp_base = __shfl_sync(kFullWarpMask, warp_base, 0);
      const int tie_slot = warp_base + __popc(selected_tie & lower_lanes);
      if (tie && tie_slot < tie_target) {
        const int slot = greater_target + tie_slot;
        row_indices[slot] = column;
        if constexpr (WriteValues) {
          row_values[slot] = value;
        }
      }
    }
  }
}

template <bool Deterministic>
void select_cuda_impl(const at::Tensor& scores, const at::Tensor& budgets,
                      at::Tensor& output,
                      const std::optional<at::Tensor>& values) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat &&
              scores.dim() == 2 && scores.is_contiguous(),
              "contiguous FP32 scores required");
  TORCH_CHECK(budgets.device() == scores.device() &&
              budgets.scalar_type() == at::kInt && budgets.is_contiguous() &&
              budgets.numel() == scores.size(0),
              "int32 local budgets required");
  TORCH_CHECK(output.device() == scores.device() &&
              output.scalar_type() == at::kInt && output.is_contiguous() &&
              output.dim() == 2 && output.size(0) == scores.size(0),
              "int32 local output required");
  if (values.has_value()) {
    TORCH_CHECK(values->device() == scores.device() &&
                values->scalar_type() == at::kFloat &&
                values->is_contiguous() && values->sizes() == output.sizes(),
                "FP32 value output required");
  }
  TORCH_CHECK(scores.size(0) > 0 && scores.size(1) > 8192 &&
              scores.size(1) <= 131072 && output.size(1) > 0 &&
              output.size(1) <= kMaxTopK && output.size(1) <= scores.size(1) &&
              output.numel() < INT32_MAX,
              "unsupported variable Top-K dimensions");

  const c10::cuda::CUDAGuard guard(scores.device());
  const auto* properties = at::cuda::getDeviceProperties(scores.get_device());
  TORCH_CHECK(properties->major == 9 && properties->minor == 0,
              "SM90 required");
  const auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
  const int rows = static_cast<int>(scores.size(0));
  const int columns = static_cast<int>(scores.size(1));
  const int output_width = static_cast<int>(output.size(1));
  if (values.has_value()) {
    radix_topk_rows<true, Deterministic><<<rows, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), budgets.data_ptr<int>(),
        output.data_ptr<int>(), values->data_ptr<float>(), rows, columns,
        output_width);
  } else {
    radix_topk_rows<false, Deterministic><<<rows, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), budgets.data_ptr<int>(),
        output.data_ptr<int>(), nullptr, rows, columns, output_width);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void select_cuda(const at::Tensor& scores, const at::Tensor& budgets,
                 at::Tensor& output, const std::optional<at::Tensor>& values) {
  select_cuda_impl<false>(scores, budgets, output, values);
}

void select_deterministic_cuda(
    const at::Tensor& scores, const at::Tensor& budgets, at::Tensor& output,
    const std::optional<at::Tensor>& values) {
  select_cuda_impl<true>(scores, budgets, output, values);
}
