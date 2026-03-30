from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class BaselineSurface:
    """Simple callable base surface used by residual height fields."""

    def __call__(self, xy: Tensor) -> Tensor:
        raise NotImplementedError


@dataclass
class ConstantBaseline(BaselineSurface):
    value: float = 0.0

    def __call__(self, xy: Tensor) -> Tensor:
        return torch.full((xy.shape[0], 1), self.value, dtype=xy.dtype, device=xy.device)


@dataclass
class PlaneBaseline(BaselineSurface):
    ax: float = 0.0
    ay: float = 0.0
    bias: float = 0.0

    def __call__(self, xy: Tensor) -> Tensor:
        z = self.ax * xy[:, :1] + self.ay * xy[:, 1:2] + self.bias
        return z.to(dtype=xy.dtype, device=xy.device)
