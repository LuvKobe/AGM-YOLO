# SPDX-License-Identifier: MIT
"""Adaptive Illumination Calibration module used in AGM-YOLO."""

import torch
from torch import nn


class AIC(nn.Module):
    """Bounded, spatially constant channel gain: y = x * (1 + sigmoid(MLP(GAP(x)))).

    Args:
        channels: Input/output channels.
        reduction: Bottleneck ratio (default: 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def calibration(self, x: torch.Tensor) -> torch.Tensor:
        """Return B,C,1,1 gain, bounded by [1,2] in finite precision."""
        return 1.0 + self.gate(x.mean(dim=(2, 3), keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.calibration(x)
