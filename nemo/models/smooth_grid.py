from __future__ import annotations

import math
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


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, xy: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(xy))


class SirenBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 64, depth: int = 4, *, omega_0: float = 30.0) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("SIREN depth must be at least 2.")

        layers: list[nn.Module] = [
            SineLayer(2, hidden_dim, is_first=True, omega_0=omega_0),
        ]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        final_linear = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            final_linear.weight.uniform_(-bound, bound)
            final_linear.bias.uniform_(-bound, bound)
        layers.append(final_linear)
        self.net = nn.Sequential(*layers)

    def forward(self, xy: Tensor) -> Tensor:
        return self.net(xy)


class ZeroResidual(nn.Module):
    def forward(self, xy_norm: Tensor) -> Tensor:
        return torch.zeros(*xy_norm.shape[:-1], 1, dtype=xy_norm.dtype, device=xy_norm.device)


class ResidualGrid(nn.Module):
    def __init__(
        self,
        grid_resolution_x: int = 128,
        grid_resolution_y: int = 128,
        *,
        interpolation: Literal["bilinear", "bicubic"] = "bilinear",
    ) -> None:
        super().__init__()
        if grid_resolution_x < 2 or grid_resolution_y < 2:
            raise ValueError("Residual grid resolution must be at least 2x2.")
        self.grid_resolution_x = grid_resolution_x
        self.grid_resolution_y = grid_resolution_y
        self.interpolation = interpolation
        self.grid = nn.Parameter(
            torch.zeros(1, 1, grid_resolution_y, grid_resolution_x, dtype=torch.float32)
        )

    def forward(self, xy_norm: Tensor) -> Tensor:
        original_shape = xy_norm.shape[:-1]
        xy_flat = xy_norm.reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            self.grid.expand(1, -1, -1, -1),
            xy_flat,
            mode=self.interpolation,
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(*original_shape, 1)


class SmoothGridHeightField(HeightField):
    """Configurable smooth backbone plus optional differentiable residual."""

    def __init__(
        self,
        bounds: Bounds,
        *,
        hidden_dim: int = 64,
        depth: int = 4,
        backbone_type: Literal["mlp", "siren"] = "mlp",
        residual_type: Literal["grid", "none"] = "grid",
        grid_resolution_x: int = 128,
        grid_resolution_y: int = 128,
        interpolation: Literal["bilinear", "bicubic"] = "bilinear",
        siren_omega_0: float = 30.0,
    ) -> None:
        super().__init__(
            bounds,
            input_normalization="minus_one_to_one",
            output_normalization="standardize",
        )
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.backbone_type = backbone_type
        self.residual_type = residual_type
        self.grid_resolution_x = grid_resolution_x
        self.grid_resolution_y = grid_resolution_y
        self.interpolation = interpolation
        self.siren_omega_0 = siren_omega_0

        if backbone_type == "mlp":
            self.backbone = CoarseMLP(hidden_dim=hidden_dim, depth=depth)
        elif backbone_type == "siren":
            self.backbone = SirenBackbone(hidden_dim=hidden_dim, depth=depth, omega_0=siren_omega_0)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        if residual_type == "grid":
            self.residual = ResidualGrid(
                grid_resolution_x=grid_resolution_x,
                grid_resolution_y=grid_resolution_y,
                interpolation=interpolation,
            )
        elif residual_type == "none":
            self.residual = ZeroResidual()
        else:
            raise ValueError(f"Unsupported residual_type: {residual_type}")

    def training_predictions(self, xy: Tensor) -> Tensor:
        xy_norm = self.normalize_inputs(xy)
        coarse = self.backbone(xy_norm.reshape(-1, 2)).reshape(*xy.shape[:-1], 1)
        residual = self.residual(xy_norm)
        return coarse + residual

    def training_gradients(self, xy: Tensor) -> Tensor:
        xy = xy.clone().detach().requires_grad_(True)
        xy_norm = self.normalize_inputs(xy)
        coarse = self.backbone(xy_norm.reshape(-1, 2)).reshape(*xy.shape[:-1], 1)
        return torch.autograd.grad(
            outputs=coarse.sum(),
            inputs=xy,
            create_graph=True,
        )[0]

    def h(self, xy: Tensor) -> Tensor:
        return self.denormalize_outputs(self.training_predictions(xy))
