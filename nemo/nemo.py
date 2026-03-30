from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from nemo.baselines import BaselineSurface
from nemo.fit import TorchFitConfig, TorchHeightFieldFitter
from nemo.height_field import Bounds, HeightField, as_targets, as_tensor
from nemo.models.hashgrid import TCNNHashGridHeightField
from nemo.models.residual_mlp import ResidualMLPHeightField
from nemo.tiling import TileConfig, TiledHeightField


class Nemo:
    """User-facing wrapper around a parameterizable neural height field."""

    def __init__(
        self,
        field: HeightField,
        *,
        fitter: TorchHeightFieldFitter | None = None,
    ) -> None:
        self.field = field
        self.fitter = fitter or TorchHeightFieldFitter()

    @classmethod
    def residual_mlp(
        cls,
        *,
        bounds: Bounds,
        baseline: BaselineSurface | None = None,
        hidden_dim: int = 128,
        depth: int = 4,
        fitter: TorchHeightFieldFitter | None = None,
    ) -> "Nemo":
        field = ResidualMLPHeightField(
            bounds=bounds,
            baseline=baseline,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        return cls(field, fitter=fitter)

    @classmethod
    def hashgrid(
        cls,
        *,
        bounds: Bounds,
        encoding_config: dict[str, Any] | None = None,
        network_config: dict[str, Any] | None = None,
        fitter: TorchHeightFieldFitter | None = None,
    ) -> "Nemo":
        field = TCNNHashGridHeightField(
            bounds=bounds,
            encoding_config=encoding_config,
            network_config=network_config,
        )
        return cls(field, fitter=fitter)

    @classmethod
    def tiled(
        cls,
        *,
        tile_config: TileConfig,
        field_factory: Callable[[Bounds], HeightField],
        fitter: TorchHeightFieldFitter | None = None,
        fit_config: TorchFitConfig | None = None,
    ) -> "Nemo":
        field = TiledHeightField(
            config=tile_config,
            field_factory=field_factory,
            fitter=fitter,
            fit_config=fit_config,
        )
        return cls(field, fitter=fitter)

    def fit(
        self,
        xy: Tensor,
        z: Tensor,
        *,
        fit_config: TorchFitConfig | None = None,
    ) -> "Nemo":
        if hasattr(self.field, "fit") and isinstance(self.field, TiledHeightField):
            self.field.fit(xy, z)
        else:
            self.fitter.fit(self.field, xy, z, config=fit_config)
        return self

    def h(self, xy: Tensor) -> Tensor:
        xy = as_tensor(xy, device=self.device)
        with torch.no_grad():
            return self.field.h(xy)

    def grad(self, xy: Tensor) -> Tensor:
        xy = as_tensor(xy, device=self.device)
        return self.field.grad(xy)

    def evaluate(self, xy: Tensor) -> Tensor:
        return self.h(xy)

    @property
    def device(self) -> torch.device:
        return next(self.field.parameters(), torch.empty(0, device=torch.device("cpu"))).device

    def to(self, device: torch.device | str) -> "Nemo":
        self.field.to(device)
        return self
