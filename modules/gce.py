# SPDX-License-Identifier: MIT
"""Global Context Enhancement module used in AGM-YOLO."""

import torch
from torch import nn


class GCE(nn.Module):
    """Apply multiplicative channel, width, and height attention.

    Width attention pools over height, while height attention pools over width.
    The 1x1 projections mix channels at each retained coordinate.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
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
