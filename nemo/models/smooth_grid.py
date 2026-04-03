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

    def forward_with_grad(self, xy: Tensor) -> tuple[Tensor, Tensor]:
        grad = None
        activations = xy
        for module in self.net:
            if isinstance(module, nn.Linear):
                activations = module(activations)
                weight = module.weight.to(dtype=xy.dtype, device=xy.device)
                if grad is None:
                    grad = weight.unsqueeze(0).expand(xy.shape[0], -1, -1)
                else:
                    grad = torch.einsum("oi,nij->noj", weight, grad)
            elif isinstance(module, nn.Softplus):
                slope = torch.sigmoid(activations)
                activations = module(activations)
                assert grad is not None
                grad = slope[..., None] * grad
            else:
                raise TypeError(f"Unsupported CoarseMLP module: {type(module).__name__}")
        assert grad is not None
        return activations, grad.squeeze(1)


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

    def forward_with_grad(self, xy: Tensor) -> tuple[Tensor, Tensor]:
        grad = None
        activations = xy
        for module in self.net:
            if isinstance(module, SineLayer):
                linear_out = module.linear(activations)
                weight = module.linear.weight.to(dtype=xy.dtype, device=xy.device)
                if grad is None:
                    linear_grad = weight.unsqueeze(0).expand(xy.shape[0], -1, -1)
                else:
                    linear_grad = torch.einsum("oi,nij->noj", weight, grad)
                slope = module.omega_0 * torch.cos(module.omega_0 * linear_out)
                activations = torch.sin(module.omega_0 * linear_out)
                grad = slope[..., None] * linear_grad
            elif isinstance(module, nn.Linear):
                activations = module(activations)
                weight = module.weight.to(dtype=xy.dtype, device=xy.device)
                assert grad is not None
                grad = torch.einsum("oi,nij->noj", weight, grad)
            else:
                raise TypeError(f"Unsupported SirenBackbone module: {type(module).__name__}")
        assert grad is not None
        return activations, grad.squeeze(1)


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

    def forward_with_grad(self, xy_norm: Tensor) -> tuple[Tensor, Tensor]:
        values = self.forward(xy_norm)
        x = xy_norm[..., 0].clamp(-1.0, 1.0)
        y = xy_norm[..., 1].clamp(-1.0, 1.0)

        grid = self.grid[0, 0]
        width = self.grid_resolution_x
        height = self.grid_resolution_y

        x_idx = 0.5 * (x + 1.0) * (width - 1)
        y_idx = 0.5 * (y + 1.0) * (height - 1)

        x0 = torch.floor(x_idx).to(torch.long).clamp(0, width - 2)
        y0 = torch.floor(y_idx).to(torch.long).clamp(0, height - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = x_idx - x0.to(dtype=x_idx.dtype)
        wy = y_idx - y0.to(dtype=y_idx.dtype)

        v00 = grid[y0, x0]
        v01 = grid[y0, x1]
        v10 = grid[y1, x0]
        v11 = grid[y1, x1]

        dv_dx_idx = (1.0 - wy) * (v01 - v00) + wy * (v11 - v10)
        dv_dy_idx = (1.0 - wx) * (v10 - v00) + wx * (v11 - v01)

        scale_x = 0.5 * float(width - 1)
        scale_y = 0.5 * float(height - 1)
        grad = torch.stack([dv_dx_idx * scale_x, dv_dy_idx * scale_y], dim=-1)

        boundary_x = (xy_norm[..., 0] <= -1.0) | (xy_norm[..., 0] >= 1.0)
        boundary_y = (xy_norm[..., 1] <= -1.0) | (xy_norm[..., 1] >= 1.0)
        grad[..., 0] = torch.where(boundary_x, torch.zeros_like(grad[..., 0]), grad[..., 0])
        grad[..., 1] = torch.where(boundary_y, torch.zeros_like(grad[..., 1]), grad[..., 1])
        return values, grad


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

    def h_and_grad(self, xy: Tensor, create_graph: bool = False) -> tuple[Tensor, Tensor]:
        if self.interpolation == "bicubic":
            return super().h_and_grad(xy, create_graph=create_graph)

        xy_local = xy.clone().detach()
        xy_norm = self.normalize_inputs(xy_local)
        norm_scale = self._input_gradient_scale(xy_norm)

        coarse_norm, coarse_grad_norm = self._backbone_with_grad(xy_norm.reshape(-1, 2))
        coarse_norm = coarse_norm.reshape(*xy.shape[:-1], 1)
        coarse_grad_norm = coarse_grad_norm.reshape(*xy.shape[:-1], 2)

        residual_norm, residual_grad_norm = self._residual_with_grad(xy_norm)
        z_norm = coarse_norm + residual_norm
        grad_norm = coarse_grad_norm + residual_grad_norm

        output_scale = self.output_scale.to(dtype=z_norm.dtype, device=z_norm.device)
        z = self.denormalize_outputs(z_norm)
        grad = output_scale * grad_norm * norm_scale
        return z, grad

    def h(self, xy: Tensor) -> Tensor:
        return self.denormalize_outputs(self.training_predictions(xy))

    def _backbone_with_grad(self, xy_norm_flat: Tensor) -> tuple[Tensor, Tensor]:
        if self.backbone_type == "mlp":
            return self.backbone.forward_with_grad(xy_norm_flat)
        if self.backbone_type == "siren":
            return self.backbone.forward_with_grad(xy_norm_flat)
        raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")

    def _residual_with_grad(self, xy_norm: Tensor) -> tuple[Tensor, Tensor]:
        if self.residual_type == "none":
            values = self.residual(xy_norm)
            grad = torch.zeros(*xy_norm.shape[:-1], 2, dtype=xy_norm.dtype, device=xy_norm.device)
            return values, grad
        if self.residual_type == "grid" and self.interpolation == "bilinear":
            return self.residual.forward_with_grad(xy_norm)
        return self.residual(xy_norm), torch.zeros(
            *xy_norm.shape[:-1],
            2,
            dtype=xy_norm.dtype,
            device=xy_norm.device,
        )

    def _input_gradient_scale(self, xy_norm: Tensor) -> Tensor:
        if self.input_normalization == "minus_one_to_one":
            scale = xy_norm.new_tensor(
                [
                    2.0 / max(float(self.bounds[0][1] - self.bounds[0][0]), 1e-8),
                    2.0 / max(float(self.bounds[1][1] - self.bounds[1][0]), 1e-8),
                ]
            )
            return scale
        if self.input_normalization == "zero_to_one":
            scale = xy_norm.new_tensor(
                [
                    1.0 / max(float(self.bounds[0][1] - self.bounds[0][0]), 1e-8),
                    1.0 / max(float(self.bounds[1][1] - self.bounds[1][0]), 1e-8),
                ]
            )
            return scale
        return xy_norm.new_tensor([1.0, 1.0])
