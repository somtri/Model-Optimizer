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

// Format geometry, array helpers and E4M3 grid lookups shared by the LUT kernels
// here and by the host binding.
//
// The constants below describe the encoded format, not any one kernel's thread
// mapping: one codebook covers an 8x64 tile, one scale covers 32 weights along a
// row, and a 16-row block carries two codebooks. lut3.cu holds the launch geometry
// that only it uses, and binds the kLut3* names to the plain ones its body reads.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>

namespace modelopt::lut {

// The 3-bit format's geometry. Prefixed because a second LUT format would declare
// its own alongside these; each kernel aliases the ones it needs to the plain names
// its body uses, so the prefix stays out of the hot code.
constexpr int kLut3CodebookSize = 8;              // 3 bits per weight
constexpr int kLut3ScaleBlock = 32;               // weights sharing one scale
constexpr int kLut3BlockRows = 16;                // rows per block; two subtiles
constexpr int kLut3TileCols = 64;                 // columns per block
constexpr int kLut3TileRows = kLut3BlockRows / 2; // rows sharing one codebook

// E4M3 magnitudes indexed by bit pattern, 0x00..0x7F. The table spans all 128
// non-negative patterns including 0x7F, which is NaN and carries 0.0 here.
// Seeding draws an index over the table and its negations, so the table length is
// part of the format: changing it changes where k-means starts and therefore the
// codebook it converges to.
constexpr int kFp8GridSize = 128;
constexpr int kFp8FiniteCount = 127;

#ifdef __CUDACC__

// Filled once from the host by upload_fp8_grid, whose definition lives in lut3.cu
// because it owns this symbol. A __constant__ defined in a header gives each
// translation unit its own copy, so a second kernel including this file would read
// an empty one -- that needs an extern declaration here and a single definition.
__constant__ float c_fp8_grid[kFp8GridSize];

__device__ inline void insertionSort_inplace(float *a, int n) {
  for (int i = 1; i < n; i++) {
    float key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) {
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = key;
  }
}

__device__ inline void copy_array(const float *src, float *dst, int n) {
  for (int i = 0; i < n; i++)
    dst[i] = src[i];
}

__device__ inline void scale_elemwise_inplace(float *a, float s, int n) {
  for (int i = 0; i < n; i++)
    a[i] *= s;
}

__device__ inline float get_abs_max(const float *a, int n) {
  float m = 0.f;
  for (int i = 0; i < n; i++)
    m = fmaxf(m, fabsf(a[i]));
  return m;
}

// Smallest magnitude in the block. Element 0 seeds the running minimum whether or
// not it is zero, and only the remaining elements skip zeros -- so a block whose
// first value is zero reports zero, which widens the scale sweep's lower bound.
__device__ inline float get_abs_min_nz(const float *a, int n) {
  float current_min = fabsf(a[0]);
  for (int i = 1; i < n; i++) {
    float candidate = fabsf(a[i]);
    if (candidate != 0.f && candidate < current_min)
      current_min = candidate;
  }
  return current_min;
}

// Copy the E4M3 table into shared memory, strided across the block's threads. Seeding
// reads it at eight independent indices at once, which constant memory would
// serialize. All kFp8GridSize entries are copied, index 127 included.
__device__ inline void fp8_values_32_threads(float *shared_values) {
  for (int i = threadIdx.x; i < kFp8GridSize; i += blockDim.x)
    shared_values[i] = c_fp8_grid[i];
  __syncthreads();
}

// The two E4M3 grid points bracketing x, by binary search over the finite entries.
// The caller reaches every combination of the pair across its 2^8 sweep, so which
// entry is which only decides exact ties -- see the return statement below.
__device__ inline float2 get_even_and_odd_sfs(float x) {
  // First index strictly greater than x, so the pair is (floor, strict ceil).
  // When x already sits on the grid this gives (x, next up) rather than (x, x),
  // which keeps that entry's degree of freedom in the 2^8 search.
  int lo = 0, hi = kFp8FiniteCount - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (c_fp8_grid[mid] <= x)
      lo = mid + 1;
    else
      hi = mid;
  }
  // Grid index 0 is exact zero and is deliberately excluded, so an all-zero block
  // quantizes to the smallest denormal times its scale rather than to zero. Only
  // magnitudes below the smallest denormal are affected; ordinary weights are not.
  int below = lo > 1 ? lo - 1 : 1;
  // Order matters: ".x" is the entry that combination bit 0 selects, so
  // combination 0 is all-".x". The search keeps the first strict minimum, which
  // makes this ordering the tie-break rule whenever two codebooks score equally.
  return make_float2(c_fp8_grid[lo], c_fp8_grid[below]);
}

// Unpack two float4s worth of bf16 (16 values) into floats.
__device__ inline void prefetch_and_unpack(const float4 *src, float *dst) {
  for (int t = 0; t < 2; t++) {
    union {
      float4 f4;
      __nv_bfloat16 bf[8];
    } u;
    u.f4 = src[t];
    for (int i = 0; i < 8; i++)
      dst[t * 8 + i] = __bfloat162float(u.bf[i]);
  }
}

#endif // __CUDACC__

// Defined in lut3.cu, which owns c_fp8_grid. Declared outside the guard so the host
// binding can call it.
void upload_fp8_grid(const float *host_grid);

} // namespace modelopt::lut
