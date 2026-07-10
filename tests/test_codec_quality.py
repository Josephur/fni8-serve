# SPDX-License-Identifier: MIT
"""Wire-codec accuracy gate: verify_boundary_fidelity (issue #175).

Tests that the codec-quality function correctly round-trips activations through
each wire-codec scheme and returns plausible SQNR / cosine values on synthetic
(Gaussian) activations.  Acceptance: real model activations pass SQNR >= 40 dB
and cosine >= 0.98 on the int4 path.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.dist.codec_quality import verify_boundary_fidelity

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="codec quality needs CUDA + fni8")


def _activation(shape, device="cuda", outlier=False):
    """Realistic-ish activation: Gaussian (optional channel outliers)."""
    x = torch.randn(*shape, device=device, dtype=torch.float16)
    if outlier:
        d = shape[-1]
        chan = torch.zeros(d, device=device, dtype=torch.float16)
        chan[:: max(1, d // 8)] = 12.0
        x = x + chan
    return x


class TestVerifyBoundaryFidelity:
    @cuda_only
    def test_int8_returns_high_sqnr(self):
        x = _activation((4, 64, 256))
        sqnr, cos = verify_boundary_fidelity(x, "int8")
        assert sqnr > 35.0, f"int8 SQNR {sqnr:.1f} dB too low"
        assert cos > 0.999, f"int8 cos {cos:.6f} too low"

    @cuda_only
    def test_int4_returns_reasonable_quality(self):
        x = _activation((4, 64, 256))
        sqnr, cos = verify_boundary_fidelity(x, "int4")
        assert sqnr > 14.0, f"int4 SQNR {sqnr:.1f} dB too low"
        assert cos > 0.98, f"int4 cos {cos:.6f} too low"

    @cuda_only
    def test_int4_had_returns_reasonable_quality(self):
        x = _activation((4, 64, 256))
        sqnr, cos = verify_boundary_fidelity(x, "int4-had")
        assert sqnr > 14.0, f"int4-had SQNR {sqnr:.1f} dB too low"
        assert cos > 0.98, f"int4-had cos {cos:.6f} too low"

    @cuda_only
    def test_nf4_returns_reasonable_quality(self):
        x = _activation((4, 64, 256))
        sqnr, cos = verify_boundary_fidelity(x, "nf4")
        assert sqnr > 16.0, f"nf4 SQNR {sqnr:.1f} dB too low"
        assert cos > 0.985, f"nf4 cos {cos:.6f} too low"

    @cuda_only
    def test_fp16_is_lossless(self):
        x = _activation((4, 64, 256))
        sqnr, cos = verify_boundary_fidelity(x, "fp16")
        assert sqnr == float("inf"), f"fp16 SQNR {sqnr} != inf"
        assert cos == pytest.approx(1.0, abs=1e-6)

    @cuda_only
    def test_outlier_quality_int4_had_beats_int4(self):
        """int4-had should outperform int4 on outlier activations."""
        x = _activation((4, 64, 128), outlier=True)
        sqnr_plain, cos_plain = verify_boundary_fidelity(x, "int4")
        sqnr_had, cos_had = verify_boundary_fidelity(x, "int4-had")
        assert sqnr_had > sqnr_plain, f"hadamard SQNR {sqnr_had:.1f} <= plain {sqnr_plain:.1f}"

    @cuda_only
    def test_custom_group_size(self):
        x = _activation((4, 64, 256))
        sqnr, cos = verify_boundary_fidelity(x, "int4", group_size=64)
        assert sqnr > 14.0, f"int4 group_size=64 SQNR {sqnr:.1f} dB too low"
        assert cos > 0.98, f"int4 group_size=64 cos {cos:.6f} too low"

    @cuda_only
    def test_explicit_none_group_size_for_int8(self):
        x = _activation((2, 128, 256))
        sqnr, cos = verify_boundary_fidelity(x, "int8", group_size=None)
        assert sqnr > 35.0
        assert cos > 0.999

    @cuda_only
    def test_different_shapes(self):
        for shape in [(1, 256), (2, 128, 256), (1, 4, 64, 256)]:
            x = _activation(shape)
            sqnr, cos = verify_boundary_fidelity(x, "int4")
            assert sqnr > 14.0, f"shape {shape} int4 SQNR {sqnr:.1f} dB too low"
            assert cos > 0.98, f"shape {shape} int4 cos {cos:.6f} too low"

    @cuda_only
    def test_return_type_is_float_tuple(self):
        x = _activation((4, 64, 256))
        result = verify_boundary_fidelity(x, "int4")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
