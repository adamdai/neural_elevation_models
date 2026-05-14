from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from nemo import CameraIntrinsics, DEM, Nemo, look_at_pose


@dataclass
class RenderHorizonArgs:
    dem_path: str = "data/dems/Mt_Etna-DSM.tif"
    checkpoint_path: str = "output/Mt_Etna-DSM__smooth-grid/model.pt"
    output_dir: str = "output/render_horizon"
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    zero_origin: bool = False
    width: int = 1024
    height: int = 384
    fx: float = 900.0
    fy: float = 900.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eye_height_m: float = 1.7
    lookahead_m: float = 250.0
    heading_deg: float = 35.0
    eye_x: float | None = None
    eye_y: float | None = None
    use_dem_gradient_heading: bool = False
    level_gaze: bool = True
    t_near: float = 0.05
    t_far: float | None = None
    num_bracket_samples: int = 96
    num_bisection_steps: int = 12
    num_newton_steps: int = 2
    row_step: float = 8.0
    row_bisection_steps: int = 8
    horizon_refine_steps: int = 2
    horizon_softmin_beta: float = 32.0
    horizon_clearance_samples: int = 64


def main(args: RenderHorizonArgs) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nemo = Nemo.load_checkpoint(args.checkpoint_path, map_location=args.device).to(args.device)
    dem = DEM.from_path(
        args.dem_path,
        xlims=args.xlims,
        ylims=args.ylims,
        zero_origin=bool(args.zero_origin),
    )
    dem, crop_summary = _align_dem_to_checkpoint_bounds(dem, nemo, args)
    intrinsics = CameraIntrinsics(
        width=int(args.width),
        height=int(args.height),
        fx=float(args.fx),
        fy=float(args.fy),
        cx=0.5 * float(args.width),
        cy=0.5 * float(args.height),
    )
    pose, pose_summary = _build_low_surface_pose(dem, nemo, args)

    _, dem_horizon = dem.render_view(pose, intrinsics, return_horizon=True)
    horizon = nemo.render_horizon(
        intrinsics,
        pose,
        t_near=float(args.t_near),
        t_far=args.t_far,
        num_bracket_samples=int(args.num_bracket_samples),
        num_bisection_steps=int(args.num_bisection_steps),
        num_newton_steps=int(args.num_newton_steps),
        row_step=float(args.row_step),
        row_bisection_steps=int(args.row_bisection_steps),
        horizon_refine_steps=int(args.horizon_refine_steps),
        horizon_softmin_beta=float(args.horizon_softmin_beta),
        horizon_clearance_samples=int(args.horizon_clearance_samples),
        differentiable=False,
    )
    nemo_horizon = horizon.rows.detach().cpu().numpy()
    nemo_horizon[~horizon.hit_mask.detach().cpu().numpy()] = -1.0

    comparison = _compose_comparison(
        dem_horizon=dem_horizon.astype(np.float32),
        nemo_horizon=nemo_horizon.astype(np.float32),
        height=int(args.height),
        width=int(args.width),
    )
    image_path = output_dir / "horizon_comparison.png"
    plt.imsave(image_path, comparison)

    np.save(output_dir / "dem_horizon.npy", dem_horizon.astype(np.int32))
    np.save(output_dir / "nemo_horizon.npy", nemo_horizon.astype(np.float32))

    valid = (dem_horizon >= 0) & (nemo_horizon >= 0)
    metrics = {
        "mean_abs_row_error": float(np.mean(np.abs(nemo_horizon[valid] - dem_horizon[valid]))) if np.any(valid) else None,
        "max_abs_row_error": float(np.max(np.abs(nemo_horizon[valid] - dem_horizon[valid]))) if np.any(valid) else None,
        "valid_columns": int(np.count_nonzero(valid)),
        "dem_hit_columns": int(np.count_nonzero(dem_horizon >= 0)),
        "nemo_hit_columns": int(np.count_nonzero(nemo_horizon >= 0)),
    }
    summary = {
        "args": asdict(args),
        "crop": crop_summary,
        "pose": pose_summary,
        "metrics": metrics,
        "image_path": str(image_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _build_low_surface_pose(
    dem: DEM,
    nemo: Nemo,
    args: RenderHorizonArgs,
) -> tuple[np.ndarray, dict[str, float | list[float] | bool]]:
    x_min, x_max = dem.bounds.x
    y_min, y_max = dem.bounds.y
    x = float(args.eye_x) if args.eye_x is not None else 0.5 * (x_min + x_max)
    y = float(args.eye_y) if args.eye_y is not None else 0.5 * (y_min + y_max)
    x = float(np.clip(x, x_min, x_max))
    y = float(np.clip(y, y_min, y_max))
    x, y, z_surface = _snap_to_finite_surface(dem, x, y)

    heading = np.deg2rad(float(args.heading_deg))
    direction_xy = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    if bool(args.use_dem_gradient_heading):
        gx, gy = dem.grad(x, y)
        downhill = -np.array([float(gx), float(gy)], dtype=np.float32)
        norm = float(np.linalg.norm(downhill))
        if norm > 1e-6:
            direction_xy = downhill / norm

    model_eye_surface = _query_field_height(nemo, x, y)
    eye_surface = max(z_surface, model_eye_surface)
    eye = np.array([x, y, eye_surface + float(args.eye_height_m)], dtype=np.float32)
    target_xy = eye[:2] + float(args.lookahead_m) * direction_xy
    target_xy[0] = np.clip(target_xy[0], x_min, x_max)
    target_xy[1] = np.clip(target_xy[1], y_min, y_max)
    target_x, target_y, target_surface = _snap_to_finite_surface(
        dem,
        float(target_xy[0]),
        float(target_xy[1]),
    )
    model_target_surface = _query_field_height(nemo, target_x, target_y)
    target_surface_for_pose = max(target_surface, model_target_surface)
    target_z = eye[2] if bool(args.level_gaze) else target_surface_for_pose + float(args.eye_height_m)
    target = np.array([target_x, target_y, target_z], dtype=np.float32)
    pose = look_at_pose(eye, target)
    return pose, {
        "eye": eye.tolist(),
        "target": target.tolist(),
        "eye_surface_z": z_surface,
        "eye_surface_z_model": model_eye_surface,
        "eye_surface_z_used": eye_surface,
        "target_surface_z": target_surface,
        "target_surface_z_model": model_target_surface,
        "target_surface_z_used": target_surface_for_pose,
        "heading_from_gradient": bool(args.use_dem_gradient_heading),
        "level_gaze": bool(args.level_gaze),
    }


def _snap_to_finite_surface(dem: DEM, x: float, y: float) -> tuple[float, float, float]:
    z = float(dem.query(x, y))
    if np.isfinite(z):
        return x, y, z

    finite = np.isfinite(dem.z)
    if not np.any(finite):
        raise ValueError("DEM has no finite elevation samples.")
    x_candidates = dem.x[finite]
    y_candidates = dem.y[finite]
    z_candidates = dem.z[finite]
    dist2 = (x_candidates - float(x)) ** 2 + (y_candidates - float(y)) ** 2
    idx = int(np.argmin(dist2))
    return (
        float(x_candidates[idx]),
        float(y_candidates[idx]),
        float(z_candidates[idx]),
    )


def _query_field_height(nemo: Nemo, x: float, y: float) -> float:
    device = next(nemo.field.parameters(), torch.empty(0, device=torch.device("cpu"))).device
    xy = torch.tensor([[x, y]], dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(nemo.field.h(xy).squeeze().detach().cpu().item())


def _align_dem_to_checkpoint_bounds(
    dem: DEM,
    nemo: Nemo,
    args: RenderHorizonArgs,
) -> tuple[DEM, dict[str, object]]:
    summary: dict[str, object] = {
        "mode": "user-specified" if args.xlims is not None or args.ylims is not None else "checkpoint-bounds",
        "applied": False,
        "dem_bounds_before": {"x": list(dem.bounds.x), "y": list(dem.bounds.y)},
        "field_bounds": {
            "x": [float(nemo.field.bounds[0][0]), float(nemo.field.bounds[0][1])],
            "y": [float(nemo.field.bounds[1][0]), float(nemo.field.bounds[1][1])],
        },
    }
    if args.xlims is not None or args.ylims is not None:
        summary["reason"] = "explicit crop provided"
        return dem, summary

    field_x = tuple(float(v) for v in nemo.field.bounds[0])
    field_y = tuple(float(v) for v in nemo.field.bounds[1])
    overlaps = not (
        field_x[1] < dem.bounds.x[0]
        or field_x[0] > dem.bounds.x[1]
        or field_y[1] < dem.bounds.y[0]
        or field_y[0] > dem.bounds.y[1]
    )
    if not overlaps:
        raise ValueError(
            "Checkpoint bounds do not overlap the loaded DEM. "
            f"DEM x={dem.bounds.x}, y={dem.bounds.y}; field x={field_x}, y={field_y}."
        )

    if dem.bounds.x == field_x and dem.bounds.y == field_y:
        summary["reason"] = "dem already matches checkpoint bounds"
        return dem, summary

    aligned = dem.crop(field_x, field_y)
    summary["applied"] = True
    summary["reason"] = "cropped DEM to checkpoint bounds"
    summary["dem_bounds_after"] = {"x": list(aligned.bounds.x), "y": list(aligned.bounds.y)}
    return aligned, summary


def _compose_comparison(
    *,
    dem_horizon: np.ndarray,
    nemo_horizon: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    dem_panel = _draw_horizon_panel(
        dem_horizon,
        height=height,
        width=width,
        line_color=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        title="PyVista DEM Horizon",
    )
    nemo_panel = _draw_horizon_panel(
        nemo_horizon,
        height=height,
        width=width,
        line_color=np.array([0.78, 0.2, 0.16], dtype=np.float32),
        title="NeMo Horizon",
    )
    return np.concatenate([dem_panel, nemo_panel], axis=1)


def _draw_horizon_panel(
    horizon_rows: np.ndarray,
    *,
    height: int,
    width: int,
    line_color: np.ndarray,
    title: str,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.float32)
    sky = np.array([0.82, 0.9, 0.98], dtype=np.float32)
    land = np.array([0.31, 0.28, 0.22], dtype=np.float32)
    image[:] = sky
    cols = np.arange(width, dtype=np.int32)
    rows = np.round(horizon_rows).astype(np.int32)
    valid = (rows >= 0) & (rows < height)
    for col in cols[valid]:
        row = rows[col]
        image[row:, col] = land
        r0 = max(row - 1, 0)
        r1 = min(row + 2, height)
        image[r0:r1, col] = line_color
    image[:28, :, :] = 0.96 * image[:28, :, :] + 0.04
    _draw_text_banner(image, title)
    return np.clip(image, 0.0, 1.0)


def _draw_text_banner(image: np.ndarray, text: str) -> None:
    fig_w = max(image.shape[1] / 160.0, 4.0)
    fig_h = max(image.shape[0] / 160.0, 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    ax.imshow(np.zeros_like(image))
    ax.axis("off")
    ax.text(
        8,
        18,
        text,
        fontsize=12,
        color="black",
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "pad": 4, "edgecolor": "none"},
    )
    fig.canvas.draw()
    banner = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].astype(np.float32) / 255.0
    plt.close(fig)
    banner = banner[: image.shape[0], : image.shape[1]]
    mask = np.any(banner < 0.995, axis=-1)
    image[: banner.shape[0], : banner.shape[1]][mask] = banner[mask]


if __name__ == "__main__":
    main(tyro.cli(RenderHorizonArgs))
