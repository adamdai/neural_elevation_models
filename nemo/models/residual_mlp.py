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
        super().__init__(
            bounds,
            input_normalization="minus_one_to_one",
            output_normalization="standardize",
        )
        self.baseline = baseline or ConstantBaseline()
        stem = [nn.Linear(2, hidden_dim), nn.SiLU()]
        trunk = [ResidualBlock(hidden_dim) for _ in range(max(depth - 1, 0))]
        head = [nn.Linear(hidden_dim, 1)]
        self.residual = nn.Sequential(*stem, *trunk, *head)

    def _baseline(self, xy: Tensor) -> Tensor:
        return self.baseline(xy)

    def _output_statistics(self, xy: Tensor, z: Tensor) -> tuple[float, float]:
        residual = z - self._baseline(xy)
        valid = torch.isfinite(residual.squeeze(-1))
        if not torch.any(valid):
            return 0.0, 1.0
        residual_valid = residual[valid]
        offset = float(residual_valid.mean().item())
        scale = float(residual_valid.std(unbiased=False).item())
        return offset, max(scale, 1e-8)

    def training_targets(self, xy: Tensor, z: Tensor) -> Tensor:
        residual = z - self._baseline(xy)
        return self.normalize_outputs(residual)

    def training_predictions(self, xy: Tensor) -> Tensor:
        xy_norm = self.normalize_inputs(xy)
        return self.residual(xy_norm)

    def h(self, xy: Tensor) -> Tensor:
        residual = self.denormalize_outputs(self.training_predictions(xy))
        return self._baseline(xy) + residual
