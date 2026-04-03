from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
import tyro

from nemo import (
    ConstantBaseline,
    DEM,
    Nemo,
    PlaneBaseline,
    TileConfig,
    TorchFitConfig,
    fit_plane_baseline,
)
from nemo.models.residual_mlp import ResidualMLPHeightField
from nemo.plotting import create_dem_fit_figure, evaluate_dem_fit


@dataclass
class ResidualMLPArgs:
    architecture: Literal["residual_mlp"] = "residual_mlp"
    hidden_dim: int = 128
    depth: int = 4
    baseline: Literal["constant", "plane"] = "plane"


@dataclass
class HashGridArgs:
    architecture: Literal["hashgrid"] = "hashgrid"
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 1.5
    n_neurons: int = 64
    n_hidden_layers: int = 2


@dataclass
class TiledResidualMLPArgs:
    architecture: Literal["tiled_residual_mlp"] = "tiled_residual_mlp"
    hidden_dim: int = 64
    depth: int = 3
    baseline: Literal["constant", "plane"] = "plane"
    tile_size_x: float = 200.0
    tile_size_y: float = 200.0
    overlap_x: float = 20.0
    overlap_y: float = 20.0


@dataclass
class SmoothGridArgs:
    architecture: Literal["smooth_grid"] = "smooth_grid"
    hidden_dim: int = 64
    depth: int = 4
    backbone_type: Literal["mlp", "siren"] = "mlp"
    residual_type: Literal["grid", "none"] = "grid"
    grid_resolution_x: int = 128
    grid_resolution_y: int = 128
    interpolation: Literal["bilinear", "bicubic"] = "bilinear"
    siren_omega_0: float = 30.0


ModelArgs = (
    Annotated[ResidualMLPArgs, tyro.conf.subcommand(name="residual-mlp")]
    | Annotated[HashGridArgs, tyro.conf.subcommand(name="hashgrid")]
    | Annotated[TiledResidualMLPArgs, tyro.conf.subcommand(name="tiled-residual-mlp")]
    | Annotated[SmoothGridArgs, tyro.conf.subcommand(name="smooth-grid")]
)


@dataclass
class FitDemArgs:
    dem_path: str
    output_root: str = "output"
    output_html: str | None = None
    patch_name: str | None = None
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    crop_center_x: float | None = None
    crop_center_y: float | None = None
    crop_width: float | None = None
    crop_height: float | None = None
    auto_open_plot: bool = True
    checkpoint_out: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_fit_points: int | None = 200000
    plot_stride: int = 1
    plot_batch_size: int = 65536
    iterations: int = 1000
    lr: float = 1e-3
    batch_size: int | None = 65536
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    gradient_supervision: bool = False
    gradient_loss_weight: float = 0.1
    eval_every: int = 25
    early_stop: bool = False
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5
    restore_best: bool = True
    seed: int = 0
    model: ModelArgs = field(default_factory=ResidualMLPArgs)


def _sample_points(
    xy: np.ndarray, z: np.ndarray, max_points: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or len(xy) <= max_points:
        return xy, z
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xy), size=max_points, replace=False)
    return xy[idx], z[idx]


def _make_baseline(
    kind: Literal["constant", "plane"], xy: torch.Tensor, z: torch.Tensor
) -> ConstantBaseline | PlaneBaseline:
    if kind == "plane":
        return fit_plane_baseline(xy, z)
    return ConstantBaseline(value=float(z.mean().item()))


def _build_nemo(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    args: FitDemArgs,
    xy: torch.Tensor,
    z: torch.Tensor,
) -> Nemo:
    model = args.model
    if isinstance(model, ResidualMLPArgs):
        return Nemo.residual_mlp(
            bounds=bounds,
            baseline=_make_baseline(model.baseline, xy, z),
            hidden_dim=model.hidden_dim,
            depth=model.depth,
        )
    if isinstance(model, HashGridArgs):
        return Nemo.hashgrid(
            bounds=bounds,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": model.n_levels,
                "n_features_per_level": model.n_features_per_level,
                "log2_hashmap_size": model.log2_hashmap_size,
                "base_resolution": model.base_resolution,
                "per_level_scale": model.per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": model.n_neurons,
                "n_hidden_layers": model.n_hidden_layers,
            },
        )
    if isinstance(model, SmoothGridArgs):
        return Nemo.smooth_grid(
            bounds=bounds,
            hidden_dim=model.hidden_dim,
            depth=model.depth,
            backbone_type=model.backbone_type,
            residual_type=model.residual_type,
            grid_resolution_x=model.grid_resolution_x,
            grid_resolution_y=model.grid_resolution_y,
            interpolation=model.interpolation,
            siren_omega_0=model.siren_omega_0,
        )
    tile_config = TileConfig(
        bounds=bounds,
        tile_size=(model.tile_size_x, model.tile_size_y),
        overlap=(model.overlap_x, model.overlap_y),
    )
    return Nemo.tiled(
        tile_config=tile_config,
        field_factory=lambda tile_bounds: ResidualMLPHeightField(
            bounds=tile_bounds,
            baseline=_make_baseline(model.baseline, xy, z),
            hidden_dim=model.hidden_dim,
            depth=model.depth,
        ),
        fit_config=TorchFitConfig(
            iterations=args.iterations,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            gradient_loss_weight=args.gradient_loss_weight if args.gradient_supervision else 0.0,
            eval_every=args.eval_every,
            early_stopping_patience=args.early_stopping_patience if args.early_stop else None,
            early_stopping_min_delta=args.early_stopping_min_delta,
            restore_best=args.restore_best,
        ),
    )


def _is_ssh_session() -> bool:
    return any(name in os.environ for name in ("SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY"))


def _has_gui_session() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _pick_local_port(preferred: int = 8765) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])


def _serve_plot_for_ssh(path: Path) -> None:
    resolved = path.resolve()
    port = _pick_local_port()
    directory = resolved.parent
    cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(port),
        "--bind",
        "127.0.0.1",
        "--directory",
        str(directory),
    ]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        print(f"Skipped auto-opening plot over SSH: could not start local HTTP server ({exc}).")
        print(f"Open {resolved} manually.")
        return

    url = f"http://127.0.0.1:{port}/{resolved.name}"
    print("SSH session detected without X11/Wayland forwarding.")
    print(f"Serving plot at {url}")
    print(f"If needed, reconnect with: ssh -L {port}:127.0.0.1:{port} <host>")


def _open_plot(path: Path) -> None:
    resolved = path.resolve()
    is_ssh = _is_ssh_session()
    has_gui = _has_gui_session()

    if is_ssh and not has_gui:
        _serve_plot_for_ssh(resolved)
        return

    if sys.platform == "darwin":
        cmd = ["open", str(resolved)]
    elif os.name == "nt":
        os.startfile(str(resolved))  # type: ignore[attr-defined]
        print(f"Opened plot at {resolved}")
        return
    else:
        cmd = ["xdg-open", str(resolved)]

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        print(f"Skipped auto-opening plot: {exc}.")
        return

    if is_ssh and has_gui:
        print(f"Opened plot via forwarded display at {resolved}")
    else:
        print(f"Opened plot at {resolved}")


def _rmse(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean(values[valid] ** 2)))


def _max_abs(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.max(np.abs(values[valid])))


def _print_fit_metrics(fields: dict[str, np.ndarray]) -> None:
    z_error = fields["z_error"]
    gx_error = fields["gx_error"]
    gy_error = fields["gy_error"]
    grad_mag_error = fields["grad_mag_error"]

    print("Fit metrics:")
    print(f"  Height error  RMSE={_rmse(z_error):.6f}  max_abs={_max_abs(z_error):.6f}")
    print(f"  Grad error    RMSE={_rmse(grad_mag_error):.6f}  max={_max_abs(grad_mag_error):.6f}")
    print(f"  X grad error  RMSE={_rmse(gx_error):.6f}  max_abs={_max_abs(gx_error):.6f}")
    print(f"  Y grad error  RMSE={_rmse(gy_error):.6f}  max_abs={_max_abs(gy_error):.6f}")


def _model_name(model: ModelArgs) -> str:
    return model.architecture.replace("_", "-")


def _default_run_dir(args: FitDemArgs) -> Path:
    dem_name = Path(args.dem_path).stem
    return Path(args.output_root) / f"{dem_name}__{_model_name(args.model)}"


def _resolve_output_html(args: FitDemArgs, run_dir: Path) -> Path:
    if args.output_html is not None:
        return Path(args.output_html)
    return run_dir / "fit.html"


def _resolve_checkpoint_path(args: FitDemArgs, run_dir: Path) -> Path:
    if args.checkpoint_out is not None:
        return Path(args.checkpoint_out)
    return run_dir / "model.pt"


def _metrics_dict(fields: dict[str, np.ndarray]) -> dict[str, float]:
    z_error = fields["z_error"]
    gx_error = fields["gx_error"]
    gy_error = fields["gy_error"]
    grad_mag_error = fields["grad_mag_error"]
    return {
        "height_rmse": _rmse(z_error),
        "height_max_abs": _max_abs(z_error),
        "grad_rmse": _rmse(grad_mag_error),
        "grad_max": _max_abs(grad_mag_error),
        "grad_x_rmse": _rmse(gx_error),
        "grad_x_max_abs": _max_abs(gx_error),
        "grad_y_rmse": _rmse(gy_error),
        "grad_y_max_abs": _max_abs(gy_error),
    }


def _load_patch_registry() -> dict[str, dict[str, dict[str, object]]]:
    registry_path = Path(__file__).resolve().parent.parent / "data" / "dem_patches.json"
    if not registry_path.exists():
        return {}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _default_patch_name(dem_path: str | Path) -> str | None:
    dem_name = Path(dem_path).name
    if dem_name == "Mt_Etna-DSM.tif":
        return "s3li_crater_default_dem_buffer"
    return None


def _resolve_crop_bounds(
    args: FitDemArgs,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, str | None]:
    explicit_xlims = args.xlims
    explicit_ylims = args.ylims
    if (
        args.crop_center_x is not None
        or args.crop_center_y is not None
        or args.crop_width is not None
        or args.crop_height is not None
    ):
        if None in (args.crop_center_x, args.crop_center_y, args.crop_width, args.crop_height):
            raise ValueError(
                "Provide all of `crop_center_x`, `crop_center_y`, `crop_width`, and `crop_height`."
            )
        half_width = float(args.crop_width) / 2.0
        half_height = float(args.crop_height) / 2.0
        explicit_xlims = (float(args.crop_center_x) - half_width, float(args.crop_center_x) + half_width)
        explicit_ylims = (float(args.crop_center_y) - half_height, float(args.crop_center_y) + half_height)

    if explicit_xlims is not None or explicit_ylims is not None:
        if explicit_xlims is None or explicit_ylims is None:
            raise ValueError("Provide both `xlims` and `ylims` when cropping explicitly.")
        return explicit_xlims, explicit_ylims, None

    registry = _load_patch_registry()
    dem_name = Path(args.dem_path).name
    patch_name = args.patch_name or _default_patch_name(args.dem_path)
    if patch_name is None:
        return None, None, None

    dem_patches = registry.get(dem_name, {})
    patch = dem_patches.get(patch_name)
    if patch is None:
        available = ", ".join(sorted(dem_patches)) or "none"
        raise ValueError(f"Unknown patch '{patch_name}' for {dem_name}. Available patches: {available}.")

    xlims = tuple(float(v) for v in patch["xlims"])
    ylims = tuple(float(v) for v in patch["ylims"])
    return xlims, ylims, patch_name


def main(args: FitDemArgs) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    xlims, ylims, patch_name = _resolve_crop_bounds(args)
    dem = DEM.from_path(args.dem_path, xlims=xlims, ylims=ylims)
    xyz = dem.to_xyz()
    valid = np.isfinite(xyz[:, 2])
    xy_np = xyz[valid, :2].astype(np.float32)
    z_np = xyz[valid, 2:3].astype(np.float32)
    xy_np, z_np = _sample_points(xy_np, z_np, args.max_fit_points, args.seed)

    xy = torch.from_numpy(xy_np)
    z = torch.from_numpy(z_np)
    grad_targets = None
    if args.gradient_supervision:
        gx_np, gy_np = dem.grad(xy_np[:, 0], xy_np[:, 1])
        grad_targets = torch.from_numpy(
            np.stack([gx_np, gy_np], axis=-1).astype(np.float32)
        )

    bounds = (dem.bounds.x, dem.bounds.y)
    nemo = _build_nemo(bounds, args, xy, z).to(args.device)

    fit_config = TorchFitConfig(
        iterations=args.iterations,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        gradient_loss_weight=args.gradient_loss_weight if args.gradient_supervision else 0.0,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping_patience if args.early_stop else None,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best=args.restore_best,
        verbose=True,
    )
    nemo.fit(xy, z, grad_targets=grad_targets, fit_config=fit_config)

    fit_fields = evaluate_dem_fit(
        dem,
        nemo,
        stride=args.plot_stride,
        batch_size=args.plot_batch_size,
    )
    _print_fit_metrics(fit_fields)
    metrics = _metrics_dict(fit_fields)

    run_dir = _default_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    figure = create_dem_fit_figure(
        dem,
        nemo,
        stride=args.plot_stride,
        batch_size=args.plot_batch_size,
    )
    output_html = _resolve_output_html(args, run_dir)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_html))
    if args.auto_open_plot:
        _open_plot(output_html)

    checkpoint_path = _resolve_checkpoint_path(args, run_dir)
    nemo.save_checkpoint(
        checkpoint_path,
        extra={
            "dem_path": str(Path(args.dem_path).resolve()),
            "model": _model_name(args.model),
            "patch_name": patch_name,
        },
    )

    results_path = run_dir / "results.json"
    results = {
        "dem_path": str(Path(args.dem_path).resolve()),
        "model": _model_name(args.model),
        "patch_name": patch_name,
        "output_dir": str(run_dir.resolve()),
        "plot_html": str(output_html.resolve()),
        "model_checkpoint": str(checkpoint_path.resolve()),
        "normalization": nemo.normalization_metadata,
        "metrics": metrics,
        "config": {
            "device": args.device,
            "xlims": list(xlims) if xlims is not None else None,
            "ylims": list(ylims) if ylims is not None else None,
            "max_fit_points": args.max_fit_points,
            "plot_stride": args.plot_stride,
            "plot_batch_size": args.plot_batch_size,
            "iterations": args.iterations,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "gradient_supervision": args.gradient_supervision,
            "gradient_loss_weight": args.gradient_loss_weight if args.gradient_supervision else 0.0,
            "eval_every": args.eval_every,
            "early_stop": args.early_stop,
            "early_stopping_patience": args.early_stopping_patience if args.early_stop else None,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "restore_best": args.restore_best,
            "seed": args.seed,
        },
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if patch_name is not None:
        print(f"Loaded DEM patch '{patch_name}'")
    print(f"Saved outputs to {run_dir}")
    print(f"Saved plot to {output_html}")
    print(f"Saved results to {results_path}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main(tyro.cli(FitDemArgs))
