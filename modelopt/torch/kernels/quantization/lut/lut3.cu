/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// 3-bit LUT weight quantization.
//
// Each 8x64 tile of weights shares an 8-entry codebook drawn from the E4M3 grid,
// and each run of 32 weights along a row shares a power-of-two scale, so a weight
// decodes as codebook[index] * 2^scale.
//
// Per tile the codebook is fitted by k-means over the tile's 512 values, snapped
// onto the E4M3 grid by an exhaustive search over all 2^8 up/down roundings, and
// paired with the power-of-two scale that minimises squared error. The result
// cannot be recomputed from the weight by a formula, so the kernel returns the
// codebook and scale alongside the quantized weights.

#include "common.cuh"

#include <cooperative_groups.h>

#define KMEANS_ITER 200

using namespace modelopt::lut;

namespace {

// The shared constants under the plain names this kernel's body reads.
constexpr int LUT_SIZE = kLut3CodebookSize;
constexpr int GRANULARITY_BS32 = kLut3ScaleBlock;
constexpr int TILE_SIZE_BS32_DIM1 = kLut3BlockRows;
constexpr int TILE_SIZE_BS32_DIM2 = kLut3TileCols;
constexpr int FP8_SIZE = kFp8GridSize;

// Launch geometry, used only by this kernel. A block is 32 threads, each owning 32
// contiguous weights of one row; two threads cover a 64-wide row, so a block spans
// the TILE_SIZE_BS32_DIM1 rows and splits into two subtiles that each fit a codebook.
constexpr int THREADS_PER_ROW_BS32 = 2; // two threads cover one 64-wide row
constexpr int THREADS_2D = 32;          // THREADS_PER_ROW_BS32 * TILE_SIZE_BS32_DIM1
constexpr int TRANSACTIONS_BS32 = 4;    // float4 transactions to cover 32 bf16 values
// The E4M3 table plus its negations, the range seeding samples a codebook entry from.
constexpr int SIGNED_FP8_SIZE = 2 * FP8_SIZE;

static_assert(TILE_SIZE_BS32_DIM2 == THREADS_PER_ROW_BS32 * GRANULARITY_BS32,
              "two threads must exactly cover one row of the tile");
static_assert(THREADS_2D == THREADS_PER_ROW_BS32 * TILE_SIZE_BS32_DIM1,
              "the block must hold one thread per row-half");

} // namespace

namespace cg = cooperative_groups;
using Subtile = cg::thread_block_tile<16>;

namespace opportunistic_kmeans_3b_8x64_bs32_k {

__device__ void lcg_advance(unsigned int &seed) { seed = 1664525 * seed + 1013904223; }

__device__ void sample_random_lut(unsigned int &seed, float *current_lut, float *shared_values,
                                  Subtile &tile) {
  unsigned int thread_rank = tile.thread_rank();
  if (thread_rank < LUT_SIZE) {
    lcg_advance(seed);
    unsigned int signed_fp8_index = seed % SIGNED_FP8_SIZE;
    float fp8_value;
    if (signed_fp8_index < FP8_SIZE) {
      fp8_value = shared_values[signed_fp8_index];
    } else {
      fp8_value = -shared_values[signed_fp8_index - FP8_SIZE];
    }
    current_lut[thread_rank] = fp8_value;
  }
  tile.sync();
  if (thread_rank == 0) {
    insertionSort_inplace(current_lut, LUT_SIZE);
  }
  tile.sync();
}

__device__ float sfar_scalar_lut_quantization(float unquantized_value, float scale_factor,
                                              float *lut) {
  for (int i = 0; i < LUT_SIZE - 1; i++) {
    float thr = 0.5 * (lut[i] + lut[i + 1]);
    if (unquantized_value < thr * scale_factor) {
      return lut[i] * scale_factor;
    }
  }
  // if survived the for loop this means it is larger than last threashold
  return lut[LUT_SIZE - 1] * scale_factor;
}

__device__ int retrieve_lut_index(float unquantized_value, float *lut) {
  for (int i = 0; i < LUT_SIZE - 1; i++) {
    float thr = 0.5 * (lut[i] + lut[i + 1]);
    if (unquantized_value < thr) {
      return i;
    }
  }
  // if survived the for loop this means it is larger than last threashold
  return LUT_SIZE - 1;
}

__device__ float get_lut_mse(float *unquant_vector, float scale_factor, float *lut) {
  float delta;
  float result = 0.f;
  float quant_data;

#pragma unroll
  for (int idx = 0; idx < GRANULARITY_BS32; idx++) {
    quant_data = sfar_scalar_lut_quantization(unquant_vector[idx], scale_factor, lut);
    delta = quant_data - unquant_vector[idx];
    result += delta * delta;
  }
  return result;
}

__device__ void get_best_e8_metadata(float *unquant_vector, float *lut, float &best_sf,
                                     float &best_mse, int lower_exponent_bound,
                                     int upper_exponent_bound) {
  int lower_loop_bound = lower_exponent_bound;
  if (lower_exponent_bound < -110) {
    lower_loop_bound = -50;
  }

  int upper_loop_bound = upper_exponent_bound;
  if (upper_exponent_bound > 110) {
    upper_loop_bound = 50;
  }

  // A block whose first element is zero reports an absolute minimum of zero, which
  // can push the lower bound above the upper one. The sweep below would then never
  // run and leave best_sf unwritten, so the block would quantize to whatever the
  // register happened to hold. Collapse the range to one candidate instead.
  if (lower_loop_bound > upper_loop_bound) {
    lower_loop_bound = upper_loop_bound;
  }

  best_mse = INFINITY;
  float current_sf;
  float current_mse;
  for (int exponent = lower_loop_bound; exponent <= upper_loop_bound; exponent++) {
    current_sf = ldexp(1.0f, exponent); // this is the candidate e8 scale factor
    current_mse = get_lut_mse(unquant_vector, current_sf, lut);

    if (current_mse < best_mse) {
      best_sf = current_sf;
      best_mse = current_mse;
    }
  }
}

__device__ void cooperative_kmeans(float *data, float *this_current_lut, float *this_even_lut,
                                   float *this_odd_lut, float *this_masses, int *this_counters,
                                   Subtile &tile) {
#pragma unroll
  for (int iter = 0; iter <= KMEANS_ITER; iter++) {
    if (tile.thread_rank() == 0) {
      for (int idx = 0; idx < LUT_SIZE; idx++) {
        this_masses[idx] = 0.f;
        this_counters[idx] = 0;
      }
    }
    tile.sync();

    for (int idx = 0; idx < GRANULARITY_BS32; idx++) {
      int lut_idx = retrieve_lut_index(data[idx], this_current_lut);
      atomicAdd(this_masses + lut_idx, data[idx]);
      atomicAdd(this_counters + lut_idx, 1);
    }
    tile.sync();

    if (tile.thread_rank() == 0) {
      for (int idx = 0; idx < LUT_SIZE; idx++) {
        if (this_counters[idx] != 0) {
          this_current_lut[idx] =
              fminf(448.f, fmaxf(-448.f, this_masses[idx] / this_counters[idx]));
        } else {
          if (idx == 0) {
            this_current_lut[idx] = 0.5 * (-448.f + this_current_lut[idx + 1]);
          } else if (idx == LUT_SIZE - 1) {
            this_current_lut[idx] = 0.5 * (this_current_lut[idx - 1] + 448.f);
          } else {
            this_current_lut[idx] = 0.5 * (this_current_lut[idx - 1] + this_current_lut[idx + 1]);
          }
        }
        if (isnan(this_current_lut[idx]) || isinf(this_current_lut[idx])) {
          unsigned int special_seed = idx;
          lcg_advance(special_seed);
          this_current_lut[idx] = special_seed % 512 - 256; // resampled as random int
        }
      }
      // we sort just in case empty regions lead to reordering
      insertionSort_inplace(this_current_lut, LUT_SIZE);
    }
  }
  if (tile.thread_rank() == 0) {
    for (int idx = 0; idx < LUT_SIZE; idx++) {
      float sign = 1.f;
      if (this_current_lut[idx] < 0) {
        sign = -1.f;
      }
      float2 even_odd_entries = get_even_and_odd_sfs(fabs(this_current_lut[idx]));
      this_even_lut[idx] = sign * even_odd_entries.x;
      this_odd_lut[idx] = sign * even_odd_entries.y;
    }
  }
  tile.sync();
}

__device__ void opportunistic_search_for_luts_and_sf(
    float *unquant_vector, float *shared_values, float &best_sf, unsigned int &seed,
    float *this_global_mse, float *this_best_lut, float *this_current_lut, float *this_even_lut,
    float *this_odd_lut, float *this_masses, int *this_counters, int lower_exponent_bound,
    int upper_exponent_bound, int max_scaling_normalizer, Subtile &tile) {
  float best_mse = INFINITY;
  float current_mse;

  float normalized_data[GRANULARITY_BS32];
  float current_sf;

  sample_random_lut(seed, this_current_lut, shared_values, tile); // this function has a sync
  copy_array(unquant_vector, normalized_data, GRANULARITY_BS32);
  scale_elemwise_inplace(normalized_data, ldexp(1.f, max_scaling_normalizer), GRANULARITY_BS32);

  cooperative_kmeans(normalized_data, this_current_lut, this_even_lut, this_odd_lut, this_masses,
                     this_counters, tile);

#pragma unroll
  for (int combination = 0; combination < (1 << LUT_SIZE); combination++) {
    if (tile.thread_rank() == 0) {
      for (int lut_idx = 0; lut_idx < LUT_SIZE; lut_idx++) {
        unsigned int b = (combination >> lut_idx) & 1; // gives 0s and 1s
        if (b == 0) {
          this_current_lut[lut_idx] = this_even_lut[lut_idx];
        } else {
          this_current_lut[lut_idx] = this_odd_lut[lut_idx];
        }
      }
      *this_global_mse = 0.f;
    }
    tile.sync();

    get_best_e8_metadata(unquant_vector, this_current_lut, current_sf, current_mse,
                         lower_exponent_bound, upper_exponent_bound);
    atomicAdd(this_global_mse, current_mse);
    tile.sync();

    if (*this_global_mse < best_mse) {
      best_mse = *this_global_mse;
      best_sf = current_sf;
      if (tile.thread_rank() == 0) {
        copy_array(this_current_lut, this_best_lut, LUT_SIZE);
      }
    }
    tile.sync(); // because at start of loop thread 0 will zero out global mse
  }
}

__global__ void quantKernel(const float4 *operand, float4 *out, int dim1, int dim2,
                            __nv_fp8_e4m3 *lut_out, signed char *sf_out) {
  int f4_dim2 = dim2 / 8;
  // each thread in block consumes 4 f4s along dim2
  int tile_offset =
      blockIdx.x * f4_dim2 * TILE_SIZE_BS32_DIM1 + blockIdx.y * THREADS_PER_ROW_BS32 * 4;
  // every 2 consecutive threads consume one row of 64 elements; 16 rows overall
  int thread_xcoord = threadIdx.x / THREADS_PER_ROW_BS32;
  int thread_ycoord = threadIdx.x % THREADS_PER_ROW_BS32;
  // xcoord puts us at the right row, ycoord skips a quartet of f4s if odd
  int thread_offset = tile_offset + thread_xcoord * f4_dim2 + thread_ycoord * 4;

  // each thread has 32 values; cooperative groups split them into two subtiles of 16
  Subtile tile = cg::tiled_partition<16>(cg::this_thread_block());

  const float4 *f4_data_ptr = operand + thread_offset;
  float4 *f4_out_ptr = out + thread_offset;

  float fdata[GRANULARITY_BS32];
  prefetch_and_unpack(f4_data_ptr, fdata); // loads 2 f4s i.e. 16 values
  prefetch_and_unpack(f4_data_ptr + 2, fdata + 16);

  float abs_max = get_abs_max(fdata, GRANULARITY_BS32);
  if (abs_max == 0.f) {
    abs_max = 1.f; // avoid potential zero/zero
  }
  float abs_min = get_abs_min_nz(fdata, GRANULARITY_BS32);

  int max_useful_exponent;
  int min_useful_exponent;
  int max_scaling_normalizer;
  bool tiny_block = false;

  // upper bounding exponent on the max because frexpf returns significand in [0.5,1)
  float significand = frexpf(abs_max, &max_useful_exponent);
  if (max_useful_exponent < -64) {
    tiny_block = true;
  }

  // absmax/exp<1 -> absmax/(exp-9) < 2^9 -> abs*(9-exp)<512
  max_scaling_normalizer = 9 - max_useful_exponent;
  // absmax<exp -> absmax/exp<1 -> absmax/(exp+9) < 2^-9, the lowest denorm in e4m3,
  // so normalizing by something bigger than this zeroes out the whole block
  max_useful_exponent += 9;

  significand = frexpf(abs_min, &min_useful_exponent);
  // absmin>exp -> absmin/exp>1 -> absmin/(exp-9) > 512 which is greater than 448,
  // so normalizing by something bigger than this saturates the whole block
  min_useful_exponent -= 10;

  if (tiny_block) {
    // dealing with tiny data: set to zero and force a tiny exponent
    max_scaling_normalizer = 0;
    min_useful_exponent = 0;
    max_useful_exponent = 0;
    scale_elemwise_inplace(fdata, 0.f, GRANULARITY_BS32);
  }

  float best_sf;

  __shared__ float shared_values[FP8_SIZE];
  fp8_values_32_threads(shared_values);

  __shared__ float best_lut_subtile[2][LUT_SIZE];
  __shared__ float current_lut_subtile[2][LUT_SIZE];
  __shared__ float even_lut_subtile[2][LUT_SIZE];
  __shared__ float odd_lut_subtile[2][LUT_SIZE];
  __shared__ float global_mse_subtile[2];
  __shared__ float masses[2][LUT_SIZE];
  __shared__ int counters[2][LUT_SIZE];

  const int sub = tile.meta_group_rank();
  float *this_best_lut = best_lut_subtile[sub];
  float *this_current_lut = current_lut_subtile[sub];
  float *this_even_lut = even_lut_subtile[sub];
  float *this_odd_lut = odd_lut_subtile[sub];
  float *this_global_mse = &global_mse_subtile[sub];
  float *this_masses = masses[sub];
  int *this_counters = counters[sub];

  unsigned int seed = blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;

  opportunistic_search_for_luts_and_sf(fdata, shared_values, best_sf, seed, this_global_mse,
                                       this_best_lut, this_current_lut, this_even_lut, this_odd_lut,
                                       this_masses, this_counters, min_useful_exponent,
                                       max_useful_exponent, max_scaling_normalizer, tile);

  if (tiny_block) {
    best_sf = ldexp(1.f, -115);
  }

#pragma unroll
  for (int tr = 0; tr < TRANSACTIONS_BS32; tr++) {
    // prepare a packer to hold 8 values
    union {
      float4 f4;
      __nv_bfloat16 bf16[8];
    } packer;

#pragma unroll
    for (int i = 0; i < 8; i++) {
      packer.bf16[i] =
          __float2bfloat16(sfar_scalar_lut_quantization(fdata[tr * 8 + i], best_sf, this_best_lut));
    }

    f4_out_ptr[tr] = packer.f4; // 8x BF16 values in 1 transaction
  }

  // --- added: return the fitted codebook and scale instead of discarding them ---
  if (lut_out != nullptr && tile.thread_rank() == 0) {
    int tile_id = (blockIdx.x * 2 + sub) * gridDim.y + blockIdx.y;
    for (int idx = 0; idx < LUT_SIZE; idx++) {
      lut_out[tile_id * LUT_SIZE + idx] = __nv_fp8_e4m3(this_best_lut[idx]);
    }
  }
  if (sf_out != nullptr) {
    int row = blockIdx.x * TILE_SIZE_BS32_DIM1 + thread_xcoord;
    int scale_group = blockIdx.y * THREADS_PER_ROW_BS32 + thread_ycoord;
    sf_out[row * (dim2 / GRANULARITY_BS32) + scale_group] = (signed char)ilogbf(best_sf);
  }
}

void launchKernel(const float4 *operand, float4 *out, int dim1, int dim2, __nv_fp8_e4m3 *lut_out,
                  signed char *sf_out, cudaStream_t stream) {
  // dim1 and dim2 are the number of elements in the incoming tensor; f4_dim2 = dim2/8.
  // Data arriving here has already been checked for size modulos.
  dim3 blocks(dim1 / TILE_SIZE_BS32_DIM1, dim2 / TILE_SIZE_BS32_DIM2);
  dim3 threads(THREADS_2D);
  quantKernel<<<blocks, threads, 0, stream>>>(operand, out, dim1, dim2, lut_out, sf_out);
}

} // namespace opportunistic_kmeans_3b_8x64_bs32_k

void modelopt::lut::upload_fp8_grid(const float *host_grid) {
  cudaMemcpyToSymbol(c_fp8_grid, host_grid, FP8_SIZE * sizeof(float));
}
