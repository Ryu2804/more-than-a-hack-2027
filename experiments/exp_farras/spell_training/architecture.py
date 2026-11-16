from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import bridge as b


class SharedScaleConv(nn.Module):
    """Average native and unit dilation with one registered Conv5 kernel."""

    def __init__(self, conv: nn.Conv1d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilated = self.conv(x)
        if self.conv.dilation == (1,):
            return dilated
        local = F.conv1d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=2,
            dilation=1,
            groups=self.conv.groups,
        )
        return (dilated + local) * 0.5


def make_model(config: Mapping[str, Any], family: str) -> nn.Module:
    model = b.original._make_model(config)
    if family == "dual_scale_tcn":
        for block in model.blocks:
            conv = block[0]
            if conv.dilation != (1,):
                block[0] = SharedScaleConv(conv)
    return model
