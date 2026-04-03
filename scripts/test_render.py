from __future__ import annotations

import json
import time
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from nemo import CameraIntrinsics, DEM, Nemo, TorchFitConfig, look_at_pose
from nemo.plotting import evaluate_dem_fit
from nemo.viewer import shade_render


@dataclass(frozen=True)
class ViewSpec:
    name: str
    eye_x: float
    eye_y: float
    eye_z: float
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model: str
    iterations: int
    hidden_dim: int
    depth: int
    grid_resolution_x: int | None = None
    grid_resolution_y: int | None = None
    gradient_loss_weight: float = 0.0
    max_fit_points: int | None = None


@dataclass
class TestRenderArgs:
    dem_path: str = "data/dems/Mt_Etna-DSM.tif"
    patch_name: str = "s3li_crater_dem_buffer_5"
    output_dir: str = "output/test_render"
    checkpoint_path: str | None = "output/Mt_Etna-DSM__smooth-grid/model.pt"
    width: int = 640
    height: int = 480
    fx: float = 500.0
    fy: float = 500.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fit_nemo: bool = False
    iterations: int = 200
    max_fit_points: int = 50000
    model: str = "smooth-grid"
    fit_eval_stride: int = 4
    compare_presets: bool = True
    benchmark_repeats: int = 3
    num_bracket_samples: int = 96
    num_bisection_steps: int = 14
    num_newton_steps: int = 2
    ray_batch_size: int = 8192
    sweep: bool = False
    sweep_ray_batch_sizes: str = "4096,8192,16384,32768"
    sweep_num_bracket_samples: str = "48,64,96"
    sweep_num_bisection_steps: str = "8,12,14"
    sweep_num_newton_steps: str = "1,2"
    sweep_view_name: str = "crater_close"


def _load_patch_bounds(dem_path: str, patch_name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    registry = json.loads((Path(__file__).resolve().parent.parent / "data" / "dem_patches.json").read_text())
    patch = registry[Path(dem_path).name][patch_name]
    return tuple(patch["xlims"]), tuple(patch["ylims"])


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value in comma-separated list.")
    return [int(item) for item in values]


def _build_camera(dem: DEM, view: ViewSpec) -> np.ndarray:
    cx = 0.5 * (dem.bounds.x[0] + dem.bounds.x[1])
    cy = 0.5 * (dem.bounds.y[0] + dem.bounds.y[1])
    width = dem.bounds.x[1] - dem.bounds.x[0]
    height = dem.bounds.y[1] - dem.bounds.y[0]
    z_ref = float(np.nanpercentile(dem.z, 75))
    eye = np.array(
        [
            cx + view.eye_x * width,
            cy + view.eye_y * height,
            z_ref + view.eye_z,
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            cx + view.target_x * width,
            cy + view.target_y * height,
            z_ref + view.target_z,
        ],
        dtype=np.float32,
    )
    return look_at_pose(eye, target)


def _default_experiments(args: TestRenderArgs) -> list[ExperimentSpec]:
    if not args.compare_presets:
        if args.model == "smooth-grid":
            return [
                ExperimentSpec(
                    name="custom",
                    model="smooth-grid",
                    iterations=args.iterations,
                    hidden_dim=64,
                    depth=4,
                    grid_resolution_x=160,
                    grid_resolution_y=160,
                    max_fit_points=args.max_fit_points,
                )
            ]
        return [
            ExperimentSpec(
                name="custom",
                model="residual-mlp",
                iterations=args.iterations,
                hidden_dim=128,
                depth=4,
                max_fit_points=args.max_fit_points,
            )
        ]

    return [
        ExperimentSpec(
            name="baseline",
            model="smooth-grid",
            iterations=args.iterations,
            hidden_dim=64,
            depth=4,
            grid_resolution_x=160,
            grid_resolution_y=160,
            max_fit_points=args.max_fit_points,
        ),
        ExperimentSpec(
            name="hires_grad",
            model="smooth-grid",
            iterations=max(args.iterations * 3, 600),
            hidden_dim=96,
            depth=5,
            grid_resolution_x=256,
            grid_resolution_y=256,
            gradient_loss_weight=0.05,
            max_fit_points=max(args.max_fit_points, 120000) if args.max_fit_points is not None else None,
        ),
    ]


def _fit_nemo(dem: DEM, spec: ExperimentSpec, args: TestRenderArgs) -> tuple[Nemo, dict[str, float | int | str | None]]:
    xyz = dem.to_xyz()
    valid = np.isfinite(xyz[:, 2])
    xyz = xyz[valid]
    if spec.max_fit_points is not None and len(xyz) > spec.max_fit_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=spec.max_fit_points, replace=False)
        xyz = xyz[idx]
    xy = torch.from_numpy(xyz[:, :2].astype(np.float32))
    z = torch.from_numpy(xyz[:, 2:3].astype(np.float32))
    bounds = (dem.bounds.x, dem.bounds.y)
    grad_targets = None
    if spec.gradient_loss_weight > 0.0:
        gx, gy = dem.grad(xyz[:, 0], xyz[:, 1])
        grad_targets = torch.from_numpy(np.column_stack([gx, gy]).astype(np.float32))

    if spec.model == "smooth-grid":
        nemo = Nemo.smooth_grid(
            bounds=bounds,
            hidden_dim=spec.hidden_dim,
            depth=spec.depth,
            grid_resolution_x=int(spec.grid_resolution_x or 160),
            grid_resolution_y=int(spec.grid_resolution_y or 160),
        )
    else:
        nemo = Nemo.residual_mlp(bounds=bounds, hidden_dim=spec.hidden_dim, depth=spec.depth)
    nemo = nemo.to(args.device)
    fit_config = TorchFitConfig(
        iterations=spec.iterations,
        lr=1e-3,
        batch_size=32768,
        gradient_loss_weight=spec.gradient_loss_weight,
        eval_every=25,
        early_stopping_patience=None,
        restore_best=True,
        verbose=True,
    )
    fit_start = time.perf_counter()
    nemo.fit(xy, z, grad_targets=grad_targets, fit_config=fit_config)
    fit_seconds = time.perf_counter() - fit_start
    metrics: dict[str, float | int | str | None] = {
        "fit_seconds": float(fit_seconds),
        "fit_points": int(len(xyz)),
        "gradient_loss_weight": float(spec.gradient_loss_weight),
        "iterations": int(spec.iterations),
        "model": spec.model,
    }
    return nemo, metrics


def _save_depth(path: Path, depth: np.ndarray, *, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(depth, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    plt.imsave(path, mask.astype(np.float32), cmap="gray")


def _save_fit_diagnostics(output_dir: Path, fields: dict[str, np.ndarray]) -> None:
    z_error = fields["z_error"]
    grad_mag_error = fields["grad_mag_error"]
    summary = {
        "height_rmse": float(np.sqrt(np.nanmean(z_error**2))),
        "height_mae": float(np.nanmean(np.abs(z_error))),
        "height_max_abs": float(np.nanmax(np.abs(z_error))),
        "grad_rmse": float(np.sqrt(np.nanmean(grad_mag_error**2))),
        "grad_mae": float(np.nanmean(np.abs(grad_mag_error))),
        "grad_max": float(np.nanmax(grad_mag_error)),
    }
    (output_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    print("[fit_eval]", json.dumps(summary))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        (fields["z_dem"], "DEM Height", "terrain"),
        (fields["z_pred"], "NEMo Height", "terrain"),
        (fields["z_error"], "Height Error", "RdBu_r"),
        (fields["grad_mag_error"], "Gradient Error Magnitude", "magma"),
    ]
    for ax, (image, title, cmap) in zip(axes.flat, panels, strict=True):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "fit_diagnostics.png", dpi=160)
    plt.close(fig)


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _print_render_diagnostics(depth: np.ndarray, *, t_near: float, t_far: float) -> None:
    finite = np.isfinite(depth)
    if not np.any(finite):
        print("[render] no finite depth values")
        return

    values = depth[finite]
    near_hits = int(np.sum(values <= t_near + 1e-3))
    far_hits = int(np.sum(values >= t_far - 1e-3))
    print(
        "[render] finite_pixels="
        f"{int(finite.sum())} "
        f"near_hits={near_hits} "
        f"far_hits={far_hits} "
        f"min={float(values.min()):.3f} "
        f"median={float(np.median(values)):.3f} "
        f"max={float(values.max()):.3f}"
    )


def _default_views() -> list[ViewSpec]:
    return [
        ViewSpec(name="overview", eye_x=-0.7, eye_y=-0.9, eye_z=1200.0, target_z=80.0),
        ViewSpec(name="close_low_west", eye_x=-0.45, eye_y=-0.55, eye_z=450.0, target_z=20.0),
        ViewSpec(name="close_low_east", eye_x=0.55, eye_y=0.10, eye_z=500.0, target_x=-0.1, target_y=-0.1, target_z=20.0),
        ViewSpec(name="grazing_south", eye_x=0.10, eye_y=0.95, eye_z=320.0, target_y=-0.2, target_z=0.0),
        ViewSpec(name="crater_close", eye_x=-0.05, eye_y=-0.20, eye_z=260.0, target_x=0.1, target_y=0.1, target_z=-40.0),
    ]


def _render_single_view(
    output_dir: Path,
    dem: DEM,
    nemo: Nemo,
    intrinsics: CameraIntrinsics,
    view: ViewSpec,
    args: TestRenderArgs,
) -> dict[str, float | int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    world_T_camera = _build_camera(dem, view)

    try:
        dem_depth, dem_rgb = dem.render_view(world_T_camera, intrinsics, return_rgb=True)
        _save_depth(output_dir / "dem_depth.png", dem_depth)
        plt.imsave(output_dir / "dem_rgb.png", dem_rgb)
        np.save(output_dir / "dem_depth.npy", dem_depth)
        print(f"Saved DEM render to {output_dir}")
    except ImportError as exc:
        print(f"Skipped PyVista DEM render for {view.name}: {exc}")

    t_near = 1.0
    t_far = 5000.0
    render_times: list[float] = []
    render = None
    for _ in range(max(int(args.benchmark_repeats), 1)):
        render_start = time.perf_counter()
        render = nemo.render_view(
            intrinsics,
            world_T_camera,
            t_near=t_near,
            t_far=t_far,
            num_bracket_samples=int(args.num_bracket_samples),
            num_bisection_steps=int(args.num_bisection_steps),
            num_newton_steps=int(args.num_newton_steps),
            ray_batch_size=int(args.ray_batch_size),
        )
        render_times.append(time.perf_counter() - render_start)
    assert render is not None
    render_seconds = float(np.mean(render_times))
    print(f"[view={view.name}]", end=" ")
    _print_render_diagnostics(render.depth, t_near=t_near, t_far=t_far)

    _save_depth(output_dir / "nemo_depth.png", render.depth)
    _save_mask(output_dir / "nemo_hit_mask.png", render.hit_mask)
    plt.imsave(output_dir / "nemo_rgb.png", shade_render(render))
    np.save(output_dir / "nemo_depth.npy", render.depth)
    np.save(output_dir / "nemo_points.npy", render.points)
    np.save(output_dir / "nemo_normals.npy", render.normals)
    if render.rgb is not None:
        np.save(output_dir / "nemo_rgb.npy", render.rgb)
    _save_json(output_dir / "view_spec.json", asdict(view))
    print(f"Saved NEMo render to {output_dir}")
    finite = np.isfinite(render.depth)
    values = render.depth[finite]
    stats: dict[str, float | int | str] = {
        "view": view.name,
        "render_seconds": render_seconds,
        "render_seconds_min": float(np.min(render_times)),
        "render_seconds_max": float(np.max(render_times)),
        "benchmark_repeats": int(len(render_times)),
        "finite_pixels": int(finite.sum()),
        "min_depth": float(values.min()) if values.size else float("nan"),
        "median_depth": float(np.median(values)) if values.size else float("nan"),
        "max_depth": float(values.max()) if values.size else float("nan"),
    }
    _save_json(output_dir / "render_summary.json", stats)
    return stats


def _benchmark_single_view(
    dem: DEM,
    nemo: Nemo,
    intrinsics: CameraIntrinsics,
    view: ViewSpec,
    *,
    num_bracket_samples: int,
    num_bisection_steps: int,
    num_newton_steps: int,
    ray_batch_size: int,
    benchmark_repeats: int,
) -> dict[str, float | int | str]:
    world_T_camera = _build_camera(dem, view)
    t_near = 1.0
    t_far = 5000.0
    render_times: list[float] = []
    render = None
    for _ in range(max(int(benchmark_repeats), 1)):
        render_start = time.perf_counter()
        render = nemo.render_view(
            intrinsics,
            world_T_camera,
            t_near=t_near,
            t_far=t_far,
            num_bracket_samples=int(num_bracket_samples),
            num_bisection_steps=int(num_bisection_steps),
            num_newton_steps=int(num_newton_steps),
            ray_batch_size=int(ray_batch_size),
        )
        render_times.append(time.perf_counter() - render_start)
    assert render is not None
    finite = np.isfinite(render.depth)
    return {
        "view": view.name,
        "render_seconds": float(np.mean(render_times)),
        "render_seconds_min": float(np.min(render_times)),
        "render_seconds_max": float(np.max(render_times)),
        "benchmark_repeats": int(len(render_times)),
        "ray_batch_size": int(ray_batch_size),
        "num_bracket_samples": int(num_bracket_samples),
        "num_bisection_steps": int(num_bisection_steps),
        "num_newton_steps": int(num_newton_steps),
        "finite_pixels": int(finite.sum()),
    }


def _run_sweep(
    output_dir: Path,
    dem: DEM,
    nemo: Nemo,
    intrinsics: CameraIntrinsics,
    args: TestRenderArgs,
) -> None:
    views = {view.name: view for view in _default_views()}
    if args.sweep_view_name not in views:
        raise ValueError(f"Unknown sweep_view_name: {args.sweep_view_name}")
    view = views[args.sweep_view_name]

    ray_batch_sizes = _parse_int_list(args.sweep_ray_batch_sizes)
    bracket_samples = _parse_int_list(args.sweep_num_bracket_samples)
    bisection_steps = _parse_int_list(args.sweep_num_bisection_steps)
    newton_steps = _parse_int_list(args.sweep_num_newton_steps)

    records: list[dict[str, float | int | str]] = []
    total = len(ray_batch_sizes) * len(bracket_samples) * len(bisection_steps) * len(newton_steps)
    index = 0
    for ray_batch_size in ray_batch_sizes:
        for num_bracket_samples in bracket_samples:
            for num_bisection_steps in bisection_steps:
                for num_newton_steps in newton_steps:
                    index += 1
                    result = _benchmark_single_view(
                        dem,
                        nemo,
                        intrinsics,
                        view,
                        num_bracket_samples=num_bracket_samples,
                        num_bisection_steps=num_bisection_steps,
                        num_newton_steps=num_newton_steps,
                        ray_batch_size=ray_batch_size,
                        benchmark_repeats=args.benchmark_repeats,
                    )
                    records.append(result)
                    print(
                        f"[sweep {index}/{total}] view={view.name} "
                        f"batch={ray_batch_size} bracket={num_bracket_samples} "
                        f"bisect={num_bisection_steps} newton={num_newton_steps} "
                        f"time={result['render_seconds']:.4f}s "
                        f"finite={result['finite_pixels']}"
                    )

    records.sort(key=lambda item: float(item["render_seconds"]))
    summary = {
        "mode": "render_sweep",
        "checkpoint_path": args.checkpoint_path,
        "view": view.name,
        "results": records,
        "best": records[0] if records else None,
    }
    _save_json(output_dir / "sweep_summary.json", summary)
    if records:
        best = records[0]
        print(
            f"[sweep_best] view={view.name} "
            f"batch={best['ray_batch_size']} bracket={best['num_bracket_samples']} "
            f"bisect={best['num_bisection_steps']} newton={best['num_newton_steps']} "
            f"time={best['render_seconds']:.4f}s"
        )


def main(args: TestRenderArgs) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xlims, ylims = _load_patch_bounds(args.dem_path, args.patch_name)
    dem = DEM.from_path(args.dem_path, xlims=xlims, ylims=ylims)
    intrinsics = CameraIntrinsics(
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
        cx=0.5 * args.width,
        cy=0.5 * args.height,
    )

    if not args.fit_nemo:
        if args.checkpoint_path is None:
            raise ValueError("checkpoint_path is required when fit_nemo is False.")
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        nemo = Nemo.load_checkpoint(checkpoint_path, map_location=args.device).to(args.device)
        run_dir = output_dir / checkpoint_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.sweep:
            _run_sweep(run_dir, dem, nemo, intrinsics, args)
            return
        render_summaries = []
        for view in _default_views():
            render_summaries.append(_render_single_view(run_dir / view.name, dem, nemo, intrinsics, view, args))
        avg_render_seconds = float(np.mean([item["render_seconds"] for item in render_summaries]))
        summary = {
            "mode": "checkpoint_render",
            "checkpoint_path": str(checkpoint_path),
            "avg_render_seconds": avg_render_seconds,
            "views": render_summaries,
            "render_config": {
                "width": int(args.width),
                "height": int(args.height),
                "fx": float(args.fx),
                "fy": float(args.fy),
                "benchmark_repeats": int(args.benchmark_repeats),
                "num_bracket_samples": int(args.num_bracket_samples),
                "num_bisection_steps": int(args.num_bisection_steps),
                "num_newton_steps": int(args.num_newton_steps),
                "ray_batch_size": int(args.ray_batch_size),
            },
        }
        _save_json(run_dir / "benchmark_summary.json", summary)
        print(
            f"[checkpoint={checkpoint_path.name}] "
            f"avg_render_seconds={avg_render_seconds:.3f}"
        )
        return

    aggregate: list[dict[str, object]] = []
    for spec in _default_experiments(args):
        variant_dir = output_dir / spec.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        _save_json(variant_dir / "experiment_spec.json", asdict(spec))
        nemo, fit_metrics = _fit_nemo(dem, spec, args)
        fit_fields = evaluate_dem_fit(dem, nemo, stride=args.fit_eval_stride)
        _save_fit_diagnostics(variant_dir, fit_fields)

        fit_summary = json.loads((variant_dir / "fit_summary.json").read_text())
        fit_summary.update(fit_metrics)
        _save_json(variant_dir / "fit_summary.json", fit_summary)

        render_summaries = []
        for view in _default_views():
            render_summaries.append(_render_single_view(variant_dir / view.name, dem, nemo, intrinsics, view, args))

        avg_render_seconds = float(np.mean([item["render_seconds"] for item in render_summaries]))
        aggregate.append(
            {
                "name": spec.name,
                "fit_summary": fit_summary,
                "avg_render_seconds": avg_render_seconds,
                "views": render_summaries,
            }
        )
        print(
            f"[experiment={spec.name}] fit_seconds={fit_summary['fit_seconds']:.2f} "
            f"height_rmse={fit_summary['height_rmse']:.2f} "
            f"avg_render_seconds={avg_render_seconds:.3f}"
        )

    _save_json(output_dir / "experiment_summary.json", {"experiments": aggregate})


if __name__ == "__main__":
    main(tyro.cli(TestRenderArgs))
