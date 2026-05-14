from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from nemo.baselines import BaselineSurface, ConstantBaseline, PlaneBaseline
from nemo.fit import TorchFitConfig, TorchHeightFieldFitter
from nemo.height_field import Bounds, HeightField, as_targets, as_tensor
from nemo.image_training import HorizonRenderResult, render_horizon_samples
from nemo.models.hashgrid import TCNNHashGridHeightField
from nemo.models.residual_mlp import ResidualMLPHeightField
from nemo.models.smooth_grid import SmoothGridHeightField
from nemo.rendering import RenderResult, render_height_field
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
    def smooth_grid(
        cls,
        *,
        bounds: Bounds,
        hidden_dim: int = 64,
        depth: int = 4,
        backbone_type: str = "mlp",
        residual_type: str = "grid",
        color_type: str = "none",
        grid_resolution_x: int = 128,
        grid_resolution_y: int = 128,
        interpolation: str = "bilinear",
        siren_omega_0: float = 30.0,
        color_encoding_config: dict[str, Any] | None = None,
        color_network_config: dict[str, Any] | None = None,
        fitter: TorchHeightFieldFitter | None = None,
    ) -> "Nemo":
        field = SmoothGridHeightField(
            bounds=bounds,
            hidden_dim=hidden_dim,
            depth=depth,
            backbone_type=backbone_type,
            residual_type=residual_type,
            color_type=color_type,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
            interpolation=interpolation,
            siren_omega_0=siren_omega_0,
            color_encoding_config=color_encoding_config,
            color_network_config=color_network_config,
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
        grad_targets: Tensor | None = None,
        *,
        fit_config: TorchFitConfig | None = None,
    ) -> "Nemo":
        if hasattr(self.field, "fit") and isinstance(self.field, TiledHeightField):
            self.field.fit(xy, z, grad_targets=grad_targets)
        else:
            self.fitter.fit(self.field, xy, z, grad_targets=grad_targets, config=fit_config)
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

    def render_view(self, intrinsics: Any, world_T_camera: Any, **kwargs: Any) -> RenderResult:
        return render_height_field(self.field, intrinsics, world_T_camera, **kwargs)

    def render_horizon(self, intrinsics: Any, world_T_camera: Any, **kwargs: Any) -> HorizonRenderResult:
        return render_horizon_samples(self.field, intrinsics, world_T_camera, **kwargs)

    @property
    def normalization_metadata(self) -> dict[str, Any]:
        return _field_normalization_metadata(self.field)

    @property
    def device(self) -> torch.device:
        return next(self.field.parameters(), torch.empty(0, device=torch.device("cpu"))).device

    def to(self, device: torch.device | str) -> "Nemo":
        self.field.to(device)
        return self

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.field.state_dict(),
            "field_spec": _serialize_field(self.field),
            "normalization": self.normalization_metadata,
        }
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        fitter: TorchHeightFieldFitter | None = None,
    ) -> "Nemo":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        field_spec = checkpoint["field_spec"]
        field = _deserialize_field(field_spec)
        state_dict = _compat_state_dict_for_field(field, checkpoint["state_dict"])
        field.load_state_dict(state_dict)
        return cls(field, fitter=fitter)


def _baseline_to_spec(baseline: BaselineSurface) -> dict[str, Any]:
    if isinstance(baseline, ConstantBaseline):
        return {"type": "constant", "value": float(baseline.value)}
    if isinstance(baseline, PlaneBaseline):
        return {
            "type": "plane",
            "ax": float(baseline.ax),
            "ay": float(baseline.ay),
            "bias": float(baseline.bias),
        }
    raise TypeError(f"Unsupported baseline type: {type(baseline).__name__}")


def _baseline_from_spec(spec: dict[str, Any]) -> BaselineSurface:
    baseline_type = spec["type"]
    if baseline_type == "constant":
        return ConstantBaseline(value=float(spec["value"]))
    if baseline_type == "plane":
        return PlaneBaseline(
            ax=float(spec["ax"]),
            ay=float(spec["ay"]),
            bias=float(spec["bias"]),
        )
    raise ValueError(f"Unsupported baseline type: {baseline_type}")


def _serialize_bounds(bounds: Bounds) -> dict[str, list[float]]:
    return {
        "x": [float(bounds[0][0]), float(bounds[0][1])],
        "y": [float(bounds[1][0]), float(bounds[1][1])],
    }


def _deserialize_bounds(spec: dict[str, Any]) -> Bounds:
    return (
        (float(spec["x"][0]), float(spec["x"][1])),
        (float(spec["y"][0]), float(spec["y"][1])),
    )


def _serialize_field(field: HeightField) -> dict[str, Any]:
    if isinstance(field, ResidualMLPHeightField):
        hidden_dim = field.residual[0].out_features
        depth = 1 + sum(1 for module in field.residual if module.__class__.__name__ == "ResidualBlock")
        return {
            "type": "residual_mlp",
            "bounds": _serialize_bounds(field.bounds),
            "hidden_dim": hidden_dim,
            "depth": depth,
            "baseline": _baseline_to_spec(field.baseline),
        }
    if isinstance(field, TCNNHashGridHeightField):
        return {
            "type": "hashgrid",
            "bounds": _serialize_bounds(field.bounds),
            "encoding_config": field.encoding_config,
            "network_config": field.network_config,
        }
    if isinstance(field, SmoothGridHeightField):
        return {
            "type": "smooth_grid",
            "bounds": _serialize_bounds(field.bounds),
            "hidden_dim": field.hidden_dim,
            "depth": field.depth,
            "backbone_type": field.backbone_type,
            "residual_type": field.residual_type,
            "color_type": field.color_type,
            "grid_resolution_x": field.grid_resolution_x,
            "grid_resolution_y": field.grid_resolution_y,
            "interpolation": field.interpolation,
            "siren_omega_0": field.siren_omega_0,
            "color_encoding_config": field.color_encoding_config,
            "color_network_config": field.color_network_config,
        }
    if isinstance(field, TiledHeightField):
        return {
            "type": "tiled",
            "config": {
                "bounds": _serialize_bounds(field.config.bounds),
                "tile_size": [float(field.config.tile_size[0]), float(field.config.tile_size[1])],
                "overlap": [float(field.config.overlap[0]), float(field.config.overlap[1])],
            },
            "fields": [_serialize_field(child) for child in field.fields],
        }
    raise TypeError(f"Unsupported height field type: {type(field).__name__}")


def _field_normalization_metadata(field: HeightField) -> dict[str, Any]:
    metadata = {
        "input_normalization": field.input_normalization,
        "output_normalization": field.output_normalization,
        "bounds": _serialize_bounds(field.bounds),
        "output_offset": float(field.output_offset.detach().cpu().item()),
        "output_scale": float(field.output_scale.detach().cpu().item()),
    }
    if isinstance(field, TiledHeightField):
        metadata["tile_bounds"] = [
            _serialize_bounds(tile_bounds)
            for tile_bounds in field.tile_bounds
        ]
        metadata["tile_field_normalization"] = [
            _field_normalization_metadata(child)
            for child in field.fields
        ]
    return metadata


def _deserialize_field(spec: dict[str, Any]) -> HeightField:
    field_type = spec["type"]
    if field_type == "residual_mlp":
        return ResidualMLPHeightField(
            bounds=_deserialize_bounds(spec["bounds"]),
            baseline=_baseline_from_spec(spec["baseline"]),
            hidden_dim=int(spec["hidden_dim"]),
            depth=int(spec["depth"]),
        )
    if field_type == "hashgrid":
        return TCNNHashGridHeightField(
            bounds=_deserialize_bounds(spec["bounds"]),
            encoding_config=dict(spec["encoding_config"]),
            network_config=dict(spec["network_config"]),
        )
    if field_type == "smooth_grid":
        return SmoothGridHeightField(
            bounds=_deserialize_bounds(spec["bounds"]),
            hidden_dim=int(spec["hidden_dim"]),
            depth=int(spec["depth"]),
            backbone_type=str(spec.get("backbone_type", "mlp")),
            residual_type=str(spec.get("residual_type", "grid")),
            color_type=str(spec.get("color_type", "none")),
            grid_resolution_x=int(spec.get("grid_resolution_x", 128)),
            grid_resolution_y=int(spec.get("grid_resolution_y", 128)),
            interpolation=str(spec.get("interpolation", "bilinear")),
            siren_omega_0=float(spec.get("siren_omega_0", 30.0)),
            color_encoding_config=(
                dict(spec["color_encoding_config"])
                if spec.get("color_encoding_config") is not None
                else None
            ),
            color_network_config=(
                dict(spec["color_network_config"])
                if spec.get("color_network_config") is not None
                else None
            ),
        )
    if field_type == "tiled":
        config_spec = spec["config"]
        tile_config = TileConfig(
            bounds=_deserialize_bounds(config_spec["bounds"]),
            tile_size=(float(config_spec["tile_size"][0]), float(config_spec["tile_size"][1])),
            overlap=(float(config_spec["overlap"][0]), float(config_spec["overlap"][1])),
        )
        child_specs = iter(spec["fields"])

        def field_factory(_: Bounds) -> HeightField:
            return _deserialize_field(next(child_specs))

        return TiledHeightField(config=tile_config, field_factory=field_factory)
    raise ValueError(f"Unsupported height field type: {field_type}")


def _compat_state_dict_for_field(field: HeightField, state_dict: dict[str, Any]) -> dict[str, Any]:
    if isinstance(field, SmoothGridHeightField):
        remapped: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("mlp."):
                remapped["backbone." + key[len("mlp."):]] = value
            elif key == "residual_grid":
                remapped["residual.grid"] = value
            else:
                remapped[key] = value
        return remapped
    return state_dict
