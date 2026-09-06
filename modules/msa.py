# SPDX-License-Identifier: MIT
"""Multi-Scale Selective Aggregation module used in AGM-YOLO."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MSA(nn.Module):
    """Fuse [current, adjacent, ...] into a tensor shaped like current.

    Each neighbor is projected to current channels with a 1x1 convolution,
    then bilinearly resized (align_corners=False). A learned 1x1 convolution
    on [current, aligned_neighbor] generates one spatial sigmoid gate.
    Adjacent-scale weights are independent and are not softmax-normalized.
    """

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        if len(channels) < 2 or any(c < 1 for c in channels):
            raise ValueError("MSA requires current and at least one adjacent channel count")
        current = channels[0]
        self.projections = nn.ModuleList(nn.Conv2d(c, current, 1, bias=False) for c in channels[1:])
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Conv2d(2 * current, 1, 1), nn.Sigmoid()) for _ in channels[1:]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.projections) + 1:
            raise ValueError("Unexpected number of MSA input features")
        current = features[0]
        output = current
        for neighbor, projection, gate in zip(features[1:], self.projections, self.gates):
            aligned = F.interpolate(
                projection(neighbor), size=current.shape[-2:], mode="bilinear", align_corners=False
            )
            weight = gate(torch.cat((current, aligned), dim=1))
            output = output + weight * aligned
        return output
