from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from nemo.fit import TorchFitConfig, TorchHeightFieldFitter
from nemo.height_field import Bounds, HeightField, as_targets, as_tensor


@dataclass(frozen=True)
class TileConfig:
    bounds: Bounds
    tile_size: tuple[float, float]
    overlap: tuple[float, float]


def _axis_starts(start: float, stop: float, size: float, overlap: float) -> list[float]:
    stride = max(size - overlap, 1e-8)
    points: list[float] = []
    current = start
    while current < stop:
        points.append(current)
        if current + size >= stop:
            break
        current += stride
    return points


def build_tile_bounds(config: TileConfig) -> list[Bounds]:
    x_starts = _axis_starts(config.bounds[0][0], config.bounds[0][1], config.tile_size[0], config.overlap[0])
    y_starts = _axis_starts(config.bounds[1][0], config.bounds[1][1], config.tile_size[1], config.overlap[1])
    tiles: list[Bounds] = []
    for x0 in x_starts:
        x1 = min(x0 + config.tile_size[0], config.bounds[0][1])
        for y0 in y_starts:
            y1 = min(y0 + config.tile_size[1], config.bounds[1][1])
            tiles.append(((x0, x1), (y0, y1)))
    return tiles


class TiledHeightField(HeightField):
    """Blend overlapping local fields for larger DEMs."""

    def __init__(
        self,
        config: TileConfig,
        field_factory: Callable[[Bounds], HeightField],
        *,
        fitter: TorchHeightFieldFitter | None = None,
        fit_config: TorchFitConfig | None = None,
    ) -> None:
        super().__init__(config.bounds)
        self.config = config
        self.tile_bounds = build_tile_bounds(config)
        self.fields = torch.nn.ModuleList([field_factory(tile_bounds) for tile_bounds in self.tile_bounds])
        self.fitter = fitter or TorchHeightFieldFitter()
        self.fit_config = fit_config or TorchFitConfig()

    def _tile_mask(self, tile_bounds: Bounds, xy: Tensor) -> Tensor:
        x0 = tile_bounds[0][0] - self.config.overlap[0] * 0.5
        x1 = tile_bounds[0][1] + self.config.overlap[0] * 0.5
        y0 = tile_bounds[1][0] - self.config.overlap[1] * 0.5
        y1 = tile_bounds[1][1] + self.config.overlap[1] * 0.5
        return (
            (xy[:, 0] >= x0)
            & (xy[:, 0] <= x1)
            & (xy[:, 1] >= y0)
            & (xy[:, 1] <= y1)
        )

    def _tile_weight(self, tile_bounds: Bounds, xy: Tensor) -> Tensor:
        cx = 0.5 * (tile_bounds[0][0] + tile_bounds[0][1])
        cy = 0.5 * (tile_bounds[1][0] + tile_bounds[1][1])
        hx = max(0.5 * (tile_bounds[0][1] - tile_bounds[0][0]), 1e-8)
        hy = max(0.5 * (tile_bounds[1][1] - tile_bounds[1][0]), 1e-8)
        dx = (xy[:, 0] - cx).abs() / hx
        dy = (xy[:, 1] - cy).abs() / hy
        wx = torch.clamp(1.0 - dx, min=0.0)
        wy = torch.clamp(1.0 - dy, min=0.0)
        return (wx * wy).unsqueeze(-1)

    def fit(self, xy: Tensor, z: Tensor) -> "TiledHeightField":
        xy = as_tensor(xy)
        z = as_targets(z)
        for tile_bounds, field in zip(self.tile_bounds, self.fields):
            mask = self._tile_mask(tile_bounds, xy)
            if not torch.any(mask):
                continue
            self.fitter.fit(field, xy[mask], z[mask], config=self.fit_config)
        return self

    def h(self, xy: Tensor) -> Tensor:
        xy = as_tensor(xy, device=next(self.fields.parameters()).device if len(self.fields) else None)
        total = torch.zeros((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)
        weight_sum = torch.zeros_like(total)
        for tile_bounds, field in zip(self.tile_bounds, self.fields):
            weights = self._tile_weight(tile_bounds, xy)
            if torch.all(weights == 0):
                continue
            total = total + weights * field.h(xy)
            weight_sum = weight_sum + weights
        return total / torch.clamp(weight_sum, min=1e-8)
