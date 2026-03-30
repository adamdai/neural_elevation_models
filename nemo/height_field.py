from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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

    def __init__(self, bounds: Bounds) -> None:
        super().__init__()
        self.bounds = bounds
        self.normalizer = CoordinateNormalizer(bounds)

    @abstractmethod
    def h(self, xy: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, xy: Tensor) -> Tensor:
        return self.h(xy)

    def grad(self, xy: Tensor, create_graph: bool = False) -> Tensor:
        xy = xy.clone().detach().requires_grad_(True)
        z = self.h(xy)
        grad = torch.autograd.grad(
            outputs=z.sum(),
            inputs=xy,
            create_graph=create_graph,
        )[0]
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
