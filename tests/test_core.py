from __future__ import annotations

import torch

from agm_yolo.core import AIC, GCE, MSA


def test_aic_shape_and_gain_bounds():
    m = AIC(32, reduction=8)
    x = torch.randn(2, 32, 20, 20)
    y = m(x)
    gain = m.calibration(x)
    assert y.shape == x.shape
    assert torch.all(gain >= 1.0)
    assert torch.all(gain <= 2.0)


def test_gce_shape():
    m = GCE(32, reduction=8)
    x = torch.randn(2, 32, 20, 24)
    y = m(x)
    assert y.shape == x.shape


def test_msa_aligns_adjacent_scales():
    m = MSA([32, 64, 128])
    current = torch.randn(2, 32, 40, 40)
    neighbor1 = torch.randn(2, 64, 20, 20)
    neighbor2 = torch.randn(2, 128, 10, 10)
    y = m([current, neighbor1, neighbor2])
    assert y.shape == current.shape
