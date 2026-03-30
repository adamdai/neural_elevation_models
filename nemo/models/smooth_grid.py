from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nemo.height_field import Bounds, HeightField


class CoarseMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64, depth: int = 4) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("CoarseMLP depth must be at least 2.")

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Softplus()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Softplus()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: Tensor) -> Tensor:
        return self.net(xy)


class SmoothGridHeightField(HeightField):
    """Smooth MLP plus differentiable residual grid sampled with grid_sample."""

    def __init__(
        self,
        bounds: Bounds,
        *,
        hidden_dim: int = 64,
        depth: int = 4,
        grid_resolution_x: int = 128,
        grid_resolution_y: int = 128,
        interpolation: Literal["bilinear", "bicubic"] = "bilinear",
    ) -> None:
        super().__init__(
            bounds,
            input_normalization="minus_one_to_one",
            output_normalization="standardize",
        )
        if grid_resolution_x < 2 or grid_resolution_y < 2:
            raise ValueError("Residual grid resolution must be at least 2x2.")
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.grid_resolution_x = grid_resolution_x
        self.grid_resolution_y = grid_resolution_y
        self.interpolation = interpolation
        self.mlp = CoarseMLP(hidden_dim=hidden_dim, depth=depth)
        self.residual_grid = nn.Parameter(
            torch.zeros(1, 1, grid_resolution_y, grid_resolution_x, dtype=torch.float32)
        )

    def _sample_residual(self, xy_norm: Tensor) -> Tensor:
        original_shape = xy_norm.shape[:-1]
        xy_flat = xy_norm.reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            self.residual_grid.expand(1, -1, -1, -1),
            xy_flat,
            mode=self.interpolation,
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(*original_shape, 1)

    def training_predictions(self, xy: Tensor) -> Tensor:
        xy_norm = self.normalize_inputs(xy)
        coarse = self.mlp(xy_norm.reshape(-1, 2)).reshape(*xy.shape[:-1], 1)
        residual = self._sample_residual(xy_norm)
        return coarse + residual

    def training_gradients(self, xy: Tensor) -> Tensor:
        xy = xy.clone().detach().requires_grad_(True)
        xy_norm = self.normalize_inputs(xy)
        coarse = self.mlp(xy_norm.reshape(-1, 2)).reshape(*xy.shape[:-1], 1)
        return torch.autograd.grad(
            outputs=coarse.sum(),
            inputs=xy,
            create_graph=True,
        )[0]

    def h(self, xy: Tensor) -> Tensor:
        return self.denormalize_outputs(self.training_predictions(xy))
