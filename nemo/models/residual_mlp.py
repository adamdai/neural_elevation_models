from __future__ import annotations

import torch
from torch import Tensor, nn

from nemo.baselines import BaselineSurface, ConstantBaseline
from nemo.height_field import Bounds, HeightField


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class ResidualMLPHeightField(HeightField):
    """Residual MLP over normalized coordinates in [-1, 1]^2."""

    def __init__(
        self,
        bounds: Bounds,
        *,
        baseline: BaselineSurface | None = None,
        hidden_dim: int = 128,
        depth: int = 4,
    ) -> None:
        super().__init__(bounds)
        self.baseline = baseline or ConstantBaseline()
        stem = [nn.Linear(2, hidden_dim), nn.SiLU()]
        trunk = [ResidualBlock(hidden_dim) for _ in range(max(depth - 1, 0))]
        head = [nn.Linear(hidden_dim, 1)]
        self.residual = nn.Sequential(*stem, *trunk, *head)

    def h(self, xy: Tensor) -> Tensor:
        xy_norm = self.normalizer.normalize_minus_one_to_one(xy)
        return self.baseline(xy) + self.residual(xy_norm)
