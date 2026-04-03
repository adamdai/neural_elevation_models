from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from typing import Sequence

import torch
from torch import Tensor, nn

Bounds = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class CoordinateNormalizer:
    bounds: Bounds

    def normalize_minus_one_to_one(self, xy: Tensor) -> Tensor:
        xy_min = xy.new_tensor([self.bounds[0][0], self.bounds[1][0]])
        xy_max = xy.new_tensor([self.bounds[0][1], self.bounds[1][1]])
        scale = torch.clamp(xy_max - xy_min, min=1e-8)
        return 2.0 * (xy - xy_min) / scale - 1.0

    def normalize_zero_to_one(self, xy: Tensor) -> Tensor:
        xy_min = xy.new_tensor([self.bounds[0][0], self.bounds[1][0]])
        xy_max = xy.new_tensor([self.bounds[0][1], self.bounds[1][1]])
        scale = torch.clamp(xy_max - xy_min, min=1e-8)
        return (xy - xy_min) / scale

    def contains(self, xy: Tensor) -> Tensor:
        x_ok = (xy[:, 0] >= self.bounds[0][0]) & (xy[:, 0] <= self.bounds[0][1])
        y_ok = (xy[:, 1] >= self.bounds[1][0]) & (xy[:, 1] <= self.bounds[1][1])
        return x_ok & y_ok


class HeightField(nn.Module, ABC):
    """Abstract neural height field exposing height and gradient queries."""

    def __init__(
        self,
        bounds: Bounds,
        *,
        input_normalization: Literal["none", "zero_to_one", "minus_one_to_one"] = "none",
        output_normalization: Literal["none", "standardize"] = "none",
    ) -> None:
        super().__init__()
        self.bounds = bounds
        self.normalizer = CoordinateNormalizer(bounds)
        self.input_normalization = input_normalization
        self.output_normalization = output_normalization
        self.register_buffer("output_offset", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("output_scale", torch.ones(1, dtype=torch.float32))
        self.normalization_metadata = {
            "input_normalization": input_normalization,
            "output_normalization": output_normalization,
            "bounds": {
                "x": [float(bounds[0][0]), float(bounds[0][1])],
                "y": [float(bounds[1][0]), float(bounds[1][1])],
            },
            "output_offset": float(self.output_offset.item()),
            "output_scale": float(self.output_scale.item()),
        }

    @abstractmethod
    def h(self, xy: Tensor) -> Tensor:
        raise NotImplementedError

    def normalize_inputs(self, xy: Tensor) -> Tensor:
        if self.input_normalization == "none":
            return xy
        if self.input_normalization == "zero_to_one":
            return self.normalizer.normalize_zero_to_one(xy)
        if self.input_normalization == "minus_one_to_one":
            return self.normalizer.normalize_minus_one_to_one(xy)
        raise ValueError(f"Unsupported input normalization mode: {self.input_normalization}")

    def configure_output_normalization(self, xy: Tensor, z: Tensor) -> None:
        if self.output_normalization == "none":
            self.output_offset.fill_(0.0)
            self.output_scale.fill_(1.0)
        elif self.output_normalization == "standardize":
            offset, scale = self._output_statistics(xy, z)
            self.output_offset.fill_(float(offset))
            self.output_scale.fill_(max(float(scale), 1e-8))
        else:
            raise ValueError(f"Unsupported output normalization mode: {self.output_normalization}")
        self.normalization_metadata["output_offset"] = float(self.output_offset.item())
        self.normalization_metadata["output_scale"] = float(self.output_scale.item())

    def _output_statistics(self, xy: Tensor, z: Tensor) -> tuple[float, float]:
        valid = torch.isfinite(z.squeeze(-1))
        if not torch.any(valid):
            return 0.0, 1.0
        z_valid = z[valid]
        offset = float(z_valid.mean().item())
        scale = float(z_valid.std(unbiased=False).item())
        return offset, max(scale, 1e-8)

    def normalize_outputs(self, z: Tensor) -> Tensor:
        if self.output_normalization == "none":
            return z
        return (z - self.output_offset.to(dtype=z.dtype, device=z.device)) / self.output_scale.to(
            dtype=z.dtype, device=z.device
        )

    def denormalize_outputs(self, z: Tensor) -> Tensor:
        if self.output_normalization == "none":
            return z
        return self.output_offset.to(dtype=z.dtype, device=z.device) + self.output_scale.to(
            dtype=z.dtype, device=z.device
        ) * z

    def training_targets(self, xy: Tensor, z: Tensor) -> Tensor:
        del xy
        return self.normalize_outputs(z)

    def training_predictions(self, xy: Tensor) -> Tensor:
        return self.normalize_outputs(self.h(xy))

    def forward(self, xy: Tensor) -> Tensor:
        return self.h(xy)

    def training_gradients(self, xy: Tensor) -> Tensor:
        return self.grad(xy, create_graph=True)

    def h_and_grad(self, xy: Tensor, create_graph: bool = False) -> tuple[Tensor, Tensor]:
        xy = xy.clone().detach().requires_grad_(True)
        z = self.h(xy)
        grad = torch.autograd.grad(
            outputs=z.sum(),
            inputs=xy,
            create_graph=create_graph,
        )[0]
        return z, grad

    def grad(self, xy: Tensor, create_graph: bool = False) -> Tensor:
        _, grad = self.h_and_grad(xy, create_graph=create_graph)
        return grad


def as_tensor(xy: Tensor | Sequence[Sequence[float]], *, device: torch.device | None = None) -> Tensor:
    tensor = torch.as_tensor(xy, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[-1] != 2:
        raise ValueError("Expected xy to have shape (N, 2).")
    return tensor


def as_targets(z: Tensor | Sequence[float] | Sequence[Sequence[float]], *, device: torch.device | None = None) -> Tensor:
    tensor = torch.as_tensor(z, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2 or tensor.shape[-1] != 1:
        raise ValueError("Expected z to have shape (N, 1) or (N,).")
    return tensor
