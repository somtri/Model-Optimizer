# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the 3-bit LUT quantization kernel."""

import pytest
import torch

import modelopt.torch.quantization.extensions as ext

# JIT-compiling the extension is slow
pytestmark = pytest.mark.timeout(300)

# Must agree with common.cuh. The kernel reports its own values so a rebuild
# with different constants is caught rather than silently producing wrong shapes.
CODEBOOK_SIZE = 8
SCALE_BLOCK = 32
TILE_ROWS = 8
TILE_COLS = 64


@pytest.fixture(scope="module")
def lut_ext():
    extension = ext.get_cuda_ext_lut()
    if extension is None:
        pytest.skip("LUT CUDA extension is unavailable")
    return extension


def test_geometry(lut_ext):
    """The kernel reports its compile-time constants so a rebuild with different
    ones is caught here rather than by silently wrong buffer shapes."""
    assert lut_ext.geometry() == {
        "codebook_size": CODEBOOK_SIZE,
        "scale_block": SCALE_BLOCK,
        "tile_rows": TILE_ROWS,
        "tile_cols": TILE_COLS,
    }


def test_output_shapes_and_dtypes(lut_ext):
    rows, cols = 64, 256
    w = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    _, codebook, scale = lut_ext.lut3_quantize(w)

    # One codebook per tile, one scale byte per 32 weights.
    assert codebook.shape == (rows // TILE_ROWS, cols // TILE_COLS, CODEBOOK_SIZE)
    assert codebook.dtype == torch.float8_e4m3fn
    assert scale.shape == (rows, cols // SCALE_BLOCK)
    assert scale.dtype == torch.int8


def test_decode_round_trip(lut_ext):
    """The returned codebook and scale must fully describe the quantized weight."""
    rows, cols = 64, 256
    w = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    quantized, codebook, scale = lut_ext.lut3_quantize(w)

    r = torch.arange(rows, device="cuda")
    c = torch.arange(cols, device="cuda")
    candidates = (
        codebook.float()[r[:, None] // TILE_ROWS, c[None, :] // TILE_COLS, :]
        * torch.exp2(scale.float())[r[:, None], c[None, :] // SCALE_BLOCK][..., None]
    )
    nearest = (quantized.float()[..., None] - candidates).abs().argmin(-1, keepdim=True)
    decoded = candidates.gather(-1, nearest).squeeze(-1)

    assert torch.equal(decoded, quantized.float())


# Golden values. The quantized weights below are what the reference kernel produces
# for these inputs, so they are what holds us to it. The codebook and scale are our
# own -- the reference does not return them -- recorded so they cannot drift silently.


def ramp_input():
    """A 16x64 weight whose every row ramps from -4.0 to 3.875 in steps of 0.125."""
    index = torch.arange(16 * 64, device="cuda")
    return (((index % 64).float() - 32) / 8).reshape(16, 64).to(torch.bfloat16)


# fmt: off
RAMP_FIRST_ROW = [
    -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5,
    -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5,
    -2.5, -1.375, -1.375, -1.375, -1.375, -1.375, -1.375, -1.375,
    -1.375, -0.40625, -0.40625, -0.40625, -0.40625, -0.40625, -0.40625, -0.40625,
    -0.203125, 0.40625, 0.40625, 0.40625, 0.40625, 0.40625, 0.40625, 1.25,
    1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 2.25, 2.25,
    2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 3.5,
    3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,
]
# fmt: on
RAMP_CODEBOOK = [
    [-224.0, -160.0, -88.0, -26.0, 52.0, 160.0, 288.0, 448.0],
    [-224.0, -160.0, -88.0, -26.0, 52.0, 160.0, 288.0, 448.0],
]
RAMP_SCALE_PER_ROW = [-6, -7]


def unstructured_input():
    """A 16x64 weight of values spread across [-4, 4), with no repeating pattern.

    Closer to a real weight tensor than ramp_input(). Generated with integer
    arithmetic rather than torch.randn so it is bit-identical on any device.
    """
    index = torch.arange(16 * 64, device="cuda", dtype=torch.int64)
    values = (((index * 1103515245 + 12345) % 2048).float() - 1024) / 256
    return values.reshape(16, 64).to(torch.bfloat16)


# fmt: off
UNSTRUCTURED_FIRST_ROW = [
    -3.5, 2.5, 1.5, -0.40625, -2.5, -3.5, 2.5, 1.5,
    -0.40625, -2.5, -3.5, 2.5, 1.5, -0.40625, -1.375, -3.5,
    3.5, 1.5, -0.40625, -1.375, -3.5, 3.5, 1.5, -0.40625,
    -1.375, -3.5, 3.5, 1.5, 0.5, -1.375, -2.5, 3.5,
    1.5, 0.5, -1.375, -2.5, 3.5, 1.5, 0.5, -1.375,
    -2.5, 3.5, 2.5, 0.5, -1.375, -2.5, 3.5, 2.5,
    0.5, -1.375, -2.5, 3.5, 2.5, 0.5, -0.40625, -2.5,
    -3.5, 2.5, 0.5, -0.40625, -2.5, -3.5, 2.5, 1.5,
]
# fmt: on
# The two tiles settle on different codebooks here, which is the format doing what it
# is for: fitting each tile to its own values.
UNSTRUCTURED_CODEBOOK = [
    [-448.0, -320.0, -176.0, -52.0, 64.0, 192.0, 320.0, 448.0],
    [-448.0, -320.0, -192.0, -56.0, 64.0, 192.0, 320.0, 448.0],
]
UNSTRUCTURED_SCALE_PER_ROW = [-7, -7]


@pytest.mark.parametrize(
    ("make_input", "expected_first_row"),
    [
        pytest.param(ramp_input, RAMP_FIRST_ROW, id="ramp"),
        pytest.param(unstructured_input, UNSTRUCTURED_FIRST_ROW, id="unstructured"),
    ],
)
def test_produces_the_expected_output(lut_ext, make_input, expected_first_row):
    """Pins the numerics: a known input must quantize to the reference's weights."""
    quantized, _, _ = lut_ext.lut3_quantize(make_input())

    assert quantized[0].float().tolist() == expected_first_row


@pytest.mark.parametrize(
    ("make_input", "expected_codebook", "expected_scale"),
    [
        pytest.param(ramp_input, RAMP_CODEBOOK, RAMP_SCALE_PER_ROW, id="ramp"),
        pytest.param(
            unstructured_input, UNSTRUCTURED_CODEBOOK, UNSTRUCTURED_SCALE_PER_ROW, id="unstructured"
        ),
    ],
)
def test_produces_the_expected_codebook_and_scale(
    lut_ext, make_input, expected_codebook, expected_scale
):
    _, codebook, scale = lut_ext.lut3_quantize(make_input())

    assert [codebook[tile, 0].float().tolist() for tile in range(2)] == expected_codebook
    # One distinct row means every row carries these scales.
    assert torch.unique(scale, dim=0).tolist() == [expected_scale]


def test_ramp_rows_quantize_identically(lut_ext):
    """Every row of ramp_input() is the same ramp, so every quantized row must match
    too -- which is what lets one expected row above stand for the whole output."""
    quantized, _, _ = lut_ext.lut3_quantize(ramp_input())

    assert torch.equal(quantized, quantized[0].expand(16, 64))


def test_all_zero_weight_quantizes_to_near_zero(lut_ext):
    """Exact zero is deliberately not reachable from the codebook, so an all-zero
    weight lands on the smallest value the format can express rather than on zero
    itself."""
    w, _, _ = lut_ext.lut3_quantize(torch.zeros(64, 256, dtype=torch.bfloat16, device="cuda"))

    assert torch.isfinite(w).all()
    assert w.abs().max() < 1e-5


def test_subnormal_weight_does_not_produce_nan(lut_ext):
    """Values this small drive the kernel's tiny-block path."""
    w, _, _ = lut_ext.lut3_quantize((torch.randn(64, 256, device="cuda") * 1e-25).bfloat16())
    assert torch.isfinite(w).all()


def test_block_starting_with_zero_does_not_produce_nan(lut_ext):
    """A block whose first value is exactly zero reports an absolute minimum of zero,
    which inverts the bounds of the kernel's scale search. It must still settle on a
    scale rather than leaving one unset. torch.randn never returns an exact zero, so
    the subnormal test above cannot reach this path."""
    row = [0.0 if col % SCALE_BLOCK == 0 else 2.0**-24 * (1 + col % 7) for col in range(64)]
    w = torch.tensor([row] * 16, dtype=torch.bfloat16, device="cuda")

    quantized, _, _ = lut_ext.lut3_quantize(w)

    assert torch.isfinite(quantized).all()


def test_deterministic(lut_ext):
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    first, _, _ = lut_ext.lut3_quantize(w)
    for _ in range(3):
        again, _, _ = lut_ext.lut3_quantize(w)
        assert torch.equal(first, again)


def test_weight_on_a_non_current_device(lut_ext):
    """The kernel launches on whichever device is current and its E4M3 table lives in
    per-device constant memory, so a weight on another GPU has to be handled
    explicitly. Getting this wrong returns an all-zero codebook rather than raising."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    torch.manual_seed(0)
    base = torch.randn(64, 256, dtype=torch.bfloat16)

    torch.cuda.set_device(0)
    expected, _, _ = lut_ext.lut3_quantize(base.cuda(0))
    on_other_device, _, _ = lut_ext.lut3_quantize(base.cuda(1))

    assert torch.equal(on_other_device.cpu(), expected.cpu())


def test_leaves_the_input_weight_unchanged(lut_ext):
    """The kernel writes to its own output buffer, so callers keep their weight."""
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    original = w.clone()

    quantized, _, _ = lut_ext.lut3_quantize(w)

    assert torch.equal(w, original)
    assert not torch.equal(quantized, original)


def _misaligned_view():
    """Contiguous and correctly shaped, but starting two bytes into its storage."""
    base = torch.randn(16 * 64 + 1, dtype=torch.bfloat16, device="cuda")
    return base[1:].view(16, 64)


@pytest.mark.parametrize(
    ("make_bad_input", "message"),
    [
        pytest.param(
            lambda: torch.randn(63, 256, dtype=torch.bfloat16, device="cuda"),
            "rows divisible by 16",
            id="ragged-rows",
        ),
        pytest.param(
            lambda: torch.randn(64, 250, dtype=torch.bfloat16, device="cuda"),
            "columns divisible by 64",
            id="ragged-columns",
        ),
        pytest.param(
            lambda: torch.randn(64, 256, dtype=torch.float32, device="cuda"),
            "bfloat16",
            id="wrong-dtype",
        ),
        pytest.param(
            lambda: torch.randn(2, 64, 256, dtype=torch.bfloat16, device="cuda"),
            "2-D",
            id="three-dimensional",
        ),
        pytest.param(
            lambda: torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")[:, ::2],
            "contiguous",
            id="strided",
        ),
        pytest.param(
            lambda: torch.empty(0, 64, dtype=torch.bfloat16, device="cuda"),
            "non-empty",
            id="zero-rows",
        ),
        pytest.param(
            lambda: torch.empty(16, 0, dtype=torch.bfloat16, device="cuda"),
            "non-empty",
            id="zero-columns",
        ),
        pytest.param(_misaligned_view, "16-byte aligned", id="misaligned"),
    ],
)
def test_rejects_unsupported_input(lut_ext, make_bad_input, message):
    """The kernel has no bounds checks and reads through float4, so a bad shape or a
    misaligned pointer must raise here rather than corrupting output or killing the
    CUDA context."""
    with pytest.raises(RuntimeError, match=message):
        lut_ext.lut3_quantize(make_bad_input())
