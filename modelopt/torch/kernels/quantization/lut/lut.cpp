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

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "common.cuh"

#include <cstdint>
#include <mutex>
#include <vector>

namespace opportunistic_kmeans_3b_8x64_bs32_k {
void launchKernel(const float4 *operand, float4 *out, int dim1, int dim2, __nv_fp8_e4m3 *lut_out,
                  signed char *sf_out, cudaStream_t stream);
} // namespace opportunistic_kmeans_3b_8x64_bs32_k

namespace modelopt::lut {

// The E4M3 magnitudes indexed by bit pattern. For a sign-magnitude format the
// non-negative patterns are already in ascending order, so no sort is needed --
// and sorting would be undefined behaviour once 0x7F (NaN) is in the table.
const std::vector<float> &fp8_grid() {
  static const std::vector<float> grid = [] {
    std::vector<float> g;
    g.reserve(kFp8GridSize);
    for (unsigned int bits = 0; bits < kFp8GridSize; bits++) {
      __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(bits), __NV_E4M3);
      g.push_back(__half2float(*reinterpret_cast<__half *>(&h)));
    }
    // 0x7F is the NaN pattern. The table carries 0.0 there so a draw landing on
    // it seeds a zero rather than a NaN the k-means repair would have to fix.
    g[127] = 0.0f;
    return g;
  }();
  return grid;
}

// Quantize `weight` and return (quantized, codebook, scale). `weight` is not modified.
std::tuple<at::Tensor, at::Tensor, at::Tensor> lut3_quantize(at::Tensor weight) {
  TORCH_CHECK(weight.is_cuda(), "lut3 requires a CUDA tensor");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "lut3 requires bfloat16, got ",
              weight.scalar_type());
  TORCH_CHECK(weight.dim() == 2, "lut3 requires a 2-D weight, got ", weight.dim(), "-D");
  TORCH_CHECK(weight.is_contiguous(), "lut3 requires a contiguous weight");

  const int rows = weight.size(0);
  const int cols = weight.size(1);
  // The kernel has no bounds checks; a ragged shape would silently leave the
  // trailing rows or columns unquantized.
  TORCH_CHECK(rows % kLut3BlockRows == 0, "lut3 requires rows divisible by ", kLut3BlockRows,
              ", got ", rows);
  TORCH_CHECK(cols % kLut3TileCols == 0, "lut3 requires columns divisible by ", kLut3TileCols,
              ", got ", cols);
  // A single non-finite value poisons every squared-error comparison in the search,
  // which is a strict <, so the block never selects a scale and the whole 8x64 tile
  // falls out as arbitrary bits. One reduction and sync per weight, against a search
  // that runs 200 k-means iterations.
  TORCH_CHECK(weight.isfinite().all().item<bool>(), "lut3 requires a finite weight");
  // Zero satisfies both modulos above but gives the launcher an empty grid, which
  // the driver rejects with a bare "invalid argument".
  TORCH_CHECK(rows > 0 && cols > 0, "lut3 requires a non-empty weight, got ", rows, "x", cols);
  // The kernel reads through float4, so the first element must be 16-byte aligned.
  // A contiguous tensor can still fail this: a view such as base[1:] starts two
  // bytes into its storage, and the launch then dies with a misaligned address.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "lut3 requires 16-byte aligned storage; clone a sliced view first");

  // The kernel launches on whichever device is current, while the output tensors
  // follow the input's device. Without this guard a weight on a non-current GPU
  // quantizes to garbage rather than raising.
  const at::cuda::OptionalCUDAGuard guard(weight.device());

  // c_fp8_grid is __constant__, so every device holds its own copy and the upload
  // has to happen once per device rather than once per process. The mutex also
  // makes concurrent first calls safe; it is taken once per tensor, which is
  // nothing next to the search kernel.
  {
    static std::mutex upload_mutex;
    static std::vector<bool> uploaded(at::cuda::device_count(), false);
    const int device = weight.get_device();
    const std::lock_guard<std::mutex> lock(upload_mutex);
    if (!uploaded[device]) {
      // Only recorded once the copy has actually landed, so a transient failure is
      // not cached for the life of the process.
      C10_CUDA_CHECK(upload_fp8_grid(fp8_grid().data(), at::cuda::getCurrentCUDAStream()));
      uploaded[device] = true;
    }
  }

  auto quantized = at::empty_like(weight);
  auto codebook = at::empty({rows / kLut3TileRows, cols / kLut3TileCols, kLut3CodebookSize},
                            weight.options().dtype(at::kFloat8_e4m3fn));
  auto scale = at::empty({rows, cols / kLut3ScaleBlock}, weight.options().dtype(at::kChar));

  ::opportunistic_kmeans_3b_8x64_bs32_k::launchKernel(
      reinterpret_cast<const float4 *>(weight.data_ptr()),
      reinterpret_cast<float4 *>(quantized.data_ptr()), rows, cols,
      reinterpret_cast<__nv_fp8_e4m3 *>(codebook.data_ptr()),
      reinterpret_cast<signed char *>(scale.data_ptr()), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {quantized, codebook, scale};
}

} // namespace modelopt::lut

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lut3_quantize", &modelopt::lut::lut3_quantize,
        "Fake-quantize a bf16 weight; returns (quantized, codebook, scale)");
  m.def(
      "geometry",
      [] {
        py::dict d;
        d["codebook_size"] = modelopt::lut::kLut3CodebookSize;
        d["scale_block"] = modelopt::lut::kLut3ScaleBlock;
        d["tile_rows"] = modelopt::lut::kLut3TileRows;
        d["tile_cols"] = modelopt::lut::kLut3TileCols;
        return d;
      },
      "Compile-time geometry, as lut3.cu was actually compiled with it");
}
