"""Core AGM-YOLO feature modules.

The implementations follow the AGM-YOLO module definitions.
They are framework-agnostic PyTorch modules and can be tested independently.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class AIC(nn.Module):
    """Adaptive Illumination Calibration.

    y = x * (1 + sigmoid(phi2(ReLU(phi1(GAP(x))))))
    The residual gain is bounded to [1, 2] in finite precision.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.gate(x.mean(dim=(2, 3), keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.calibration(x)


class GCE(nn.Module):
    """Global Context Enhancement with channel, width and height branches."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.width = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.height = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def attention(self, x: torch.Tensor):
        return (
            self.channel(x.mean(dim=(2, 3), keepdim=True)),
            self.width(x.mean(dim=2, keepdim=True)),
            self.height(x.mean(dim=3, keepdim=True)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc, mw, mh = self.attention(x)
        return x * mc * mw * mh


class MSA(nn.Module):
    """Multi-Scale Selective Aggregation.

    The first tensor is the current-scale feature. Remaining tensors are adjacent
    scales. Each adjacent feature is channel-aligned, bilinearly resized to the
    current spatial size, spatially gated, and added residually.
    """

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        if len(channels) < 2 or any(int(c) < 1 for c in channels):
            raise ValueError("MSA requires current and at least one positive adjacent channel count")
        current = int(channels[0])
        self.projections = nn.ModuleList(
            nn.Conv2d(int(c), current, 1, bias=False) for c in channels[1:]
        )
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Conv2d(2 * current, 1, 1), nn.Sigmoid())
            for _ in channels[1:]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.projections) + 1:
            raise ValueError("Unexpected number of MSA input features")
        current = features[0]
        output = current
        for neighbor, projection, gate in zip(features[1:], self.projections, self.gates):
            aligned = F.interpolate(
                projection(neighbor),
                size=current.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            weight = gate(torch.cat((current, aligned), dim=1))
            output = output + weight * aligned
        return output
