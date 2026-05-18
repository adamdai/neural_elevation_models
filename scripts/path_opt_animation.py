"""Animate terrain-aware crater path optimization."""

from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import mediapy as media
import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from nemo import Nemo
from nemo.airsim_trajectory import generate_airsim_reference_trajectory
from nemo.physical_objective_planner import VehicleModelConfig
from nemo.terrain_aware_planner import (
    TerrainAwarePlannerConfig,
    _cost_normalizers,
    _trajectory_to_numpy,
    _weighted_total,
    bspline_basis,
    compute_raw_costs,
    initialize_control_points,
    make_diagnostics,
    path_xyz,
    sample_path,
    trajectory_from_path,
    validate_path,
)
from scripts.compare_physical_objectives import dataset_to_airsim_path, sample_airsim_grid
from scripts.crater_test import airsim_to_dataset, start_preview_server
from scripts.crater_test_terrain_aware import (
    PATH_PLOT_Z_OFFSET_M,
    build_astar_seed,
    trajectory_metrics,
)


DEFAULT_CHECKPOINT = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
)


def build_planner_config() -> TerrainAwarePlannerConfig:
    return TerrainAwarePlannerConfig(
        num_control_points=100,
        num_samples=1000,
        num_iters=1800,
        lr=0.25,
        nominal_speed=6.0,
        wheelbase_m=2.8,
        max_steering_rad=0.1,
        max_throttle_accel=2.0,
        gravity=9.81,
        uphill_limit_rad=float(np.deg2rad(10.0)),
        downhill_limit_rad=float(np.deg2rad(25.0)),
        crosstrack_limit_rad=float(np.deg2rad(15.0)),
        w_length=0.05,
        w_uphill=1.0,
        w_downhill=0.01,
        w_crosstrack=0.3,
        w_throttle=0.36,
        w_steering=0.01,
        w_smoothness=0.0,
        bounds_margin=2.0,
        normalize_costs=True,
        verbose=True,
    )


def ground_airsim(nemo: Nemo, xy_airsim: np.ndarray) -> np.ndarray:
    ds_xy = airsim_to_dataset(np.append(xy_airsim, 0.0))[:2]
    xyz_ds = path_xyz(nemo, ds_xy[None])[0]
    return dataset_to_airsim_path(xyz_ds[None])[0]


def build_animation_figure(
    *,
    terrain: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    astar_airsim: np.ndarray,
    current_airsim: np.ndarray | None,
    start_airsim: np.ndarray,
    goal_airsim: np.ndarray,
    show_markers: bool,
    eye_distance: float,
) -> go.Figure:
    as_x, as_y, as_z, _slope = terrain
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=as_x,
            y=as_y,
            z=as_z,
            surfacecolor=as_z,
            colorscale="Viridis",
            cmin=float(np.nanpercentile(as_z, 1.0)),
            cmax=float(np.nanpercentile(as_z, 99.0)),
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            colorbar=dict(title="Elevation<br>up (m)", thickness=16),
            name="NEMo surface",
        ),
    )

    # astar_color = "#ff3333"
    astar_color = "red"
    # opt_color = "#ff9900"
    opt_color = "orange"
    paths: list[tuple[np.ndarray, str, str, str, int, str | None]] = [
        (astar_airsim, "A* seed", astar_color, "lines+markers", 5, "dash")
    ]
    if current_airsim is not None:
        paths.append((current_airsim, "optimized path", opt_color, "lines", 8, None))

    for path, name, color, mode, width, dash in paths:
        line = dict(color=color, width=width)
        if dash is not None:
            line["dash"] = dash
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2] + PATH_PLOT_Z_OFFSET_M,
                mode=mode,
                line=line,
                marker=dict(color=color, size=3),
                name=name,
            ),
        )

    if show_markers:
        fig.add_trace(
            go.Scatter3d(
                x=[start_airsim[0]],
                y=[start_airsim[1]],
                z=[start_airsim[2] + PATH_PLOT_Z_OFFSET_M],
                mode="markers",
                marker=dict(color="#00ffff", size=10, symbol="circle"),
                name="Start",
            ),
        )
        fig.add_trace(
            go.Scatter3d(
                x=[goal_airsim[0]],
                y=[goal_airsim[1]],
                z=[goal_airsim[2] + PATH_PLOT_Z_OFFSET_M],
                mode="markers",
                marker=dict(color="#ff00ff", size=12, symbol="diamond"),
                name="Goal",
            ),
        )

    clean_axis = dict(
        showgrid=False, zeroline=False, showline=False, showticklabels=False, title=""
    )
    camera_direction = np.asarray([1.3, -1.5, 0.9], dtype=np.float64)
    camera_direction /= np.linalg.norm(camera_direction)
    camera_eye = camera_direction * float(eye_distance)
    fig.update_layout(
        height=1300,
        width=1500,
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.35)",
        ),
        scene=dict(
            aspectmode="data",
            xaxis=clean_axis,
            yaxis=dict(autorange="reversed", **clean_axis),
            zaxis=clean_axis,
            camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])),
        ),
    )
    return fig


def save_frame(fig: go.Figure, path: Path, *, width: int, height: int, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(path, width=width, height=height, scale=scale)


def run_animated_optimization(
    *,
    nemo: Nemo,
    astar_xy: np.ndarray,
    astar_airsim: np.ndarray,
    start_airsim: np.ndarray,
    goal_airsim: np.ndarray,
    cfg: TerrainAwarePlannerConfig,
    terrain: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    frames_dir: Path,
    frame_every: int,
    width: int,
    height: int,
    scale: float,
    show_markers: bool,
    eye_distance: float,
) -> dict[str, object]:
    seed = validate_path(astar_xy)
    controls_np = initialize_control_points(seed, cfg.num_control_points)
    basis_np = bspline_basis(
        num_control_points=controls_np.shape[0],
        num_samples=max(int(cfg.num_samples), 2),
        degree=min(3, controls_np.shape[0] - 1),
    )
    device = nemo.device
    basis = torch.as_tensor(basis_np, dtype=torch.float32, device=device)
    controls_initial = torch.as_tensor(controls_np, dtype=torch.float32, device=device)
    start = controls_initial[0].detach().clone()
    goal = controls_initial[-1].detach().clone()
    interior = torch.nn.Parameter(controls_initial[1:-1].detach().clone())
    optimizer = torch.optim.Adam([interior], lr=float(cfg.lr))

    initial_path = sample_path(controls_initial, basis)
    initial_trajectory = trajectory_from_path(nemo, initial_path, cfg)
    initial_raw_costs = compute_raw_costs(initial_trajectory, controls_initial, cfg)
    normalizers = _cost_normalizers(initial_raw_costs, enabled=cfg.normalize_costs)
    initial_total = _weighted_total(initial_raw_costs, normalizers, cfg)
    initial_terms = {**initial_raw_costs, "total": initial_total}
    initial_diagnostics = make_diagnostics(initial_trajectory, initial_terms)

    best_total = float(initial_total.detach().cpu().item())
    best_interior = interior.detach().clone()
    best_terms = {name: value.detach().clone() for name, value in initial_terms.items()}
    history: list[float] = []

    x_min = float(nemo.field.bounds[0][0]) + float(cfg.bounds_margin)
    x_max = float(nemo.field.bounds[0][1]) - float(cfg.bounds_margin)
    y_min = float(nemo.field.bounds[1][0]) + float(cfg.bounds_margin)
    y_max = float(nemo.field.bounds[1][1]) - float(cfg.bounds_margin)

    frame_index = 0
    save_frame(
        build_animation_figure(
            terrain=terrain,
            astar_airsim=astar_airsim,
            current_airsim=None,
            start_airsim=start_airsim,
            goal_airsim=goal_airsim,
            show_markers=show_markers,
            eye_distance=eye_distance,
        ),
        frames_dir / f"frame-{frame_index:05d}.png",
        width=width,
        height=height,
        scale=scale,
    )
    frame_index += 1

    initial_airsim = dataset_to_airsim_path(path_xyz(nemo, initial_path.detach().cpu().numpy()))
    save_frame(
        build_animation_figure(
            terrain=terrain,
            astar_airsim=astar_airsim,
            current_airsim=initial_airsim,
            start_airsim=start_airsim,
            goal_airsim=goal_airsim,
            show_markers=show_markers,
            eye_distance=eye_distance,
        ),
        frames_dir / f"frame-{frame_index:05d}.png",
        width=width,
        height=height,
        scale=scale,
    )
    frame_index += 1

    frame_every = max(int(frame_every), 1)
    pbar = tqdm(range(1, int(cfg.num_iters) + 1), desc="Optimizing and rendering")
    for iteration in pbar:
        optimizer.zero_grad(set_to_none=True)
        controls = torch.cat([start[None], interior, goal[None]], dim=0)
        path = sample_path(controls, basis)
        traj = trajectory_from_path(nemo, path, cfg)
        raw_costs = compute_raw_costs(traj, controls, cfg)
        total = _weighted_total(raw_costs, normalizers, cfg)
        total.backward()
        optimizer.step()

        with torch.no_grad():
            interior[:, 0].clamp_(x_min, x_max)
            interior[:, 1].clamp_(y_min, y_max)
            candidate_controls = torch.cat([start[None], interior, goal[None]], dim=0)
            candidate_path = sample_path(candidate_controls, basis)
            candidate_traj = trajectory_from_path(nemo, candidate_path, cfg)
            candidate_raw = compute_raw_costs(candidate_traj, candidate_controls, cfg)
            candidate_total = _weighted_total(candidate_raw, normalizers, cfg)
            candidate_value = float(candidate_total.detach().cpu().item())
            if candidate_value < best_total:
                best_total = candidate_value
                best_interior = interior.detach().clone()
                best_terms = {name: value.detach().clone() for name, value in candidate_raw.items()}
                best_terms["total"] = candidate_total.detach().clone()
        history.append(candidate_value)
        pbar.set_postfix(cost=f"{candidate_value:.4f}", best=f"{best_total:.4f}")

        if iteration % frame_every == 0 or iteration == int(cfg.num_iters):
            current_controls = torch.cat([start[None], interior, goal[None]], dim=0)
            current_path = sample_path(current_controls, basis)
            current_airsim = dataset_to_airsim_path(
                path_xyz(nemo, current_path.detach().cpu().numpy())
            )
            save_frame(
                build_animation_figure(
                    terrain=terrain,
                    astar_airsim=astar_airsim,
                    current_airsim=current_airsim,
                    start_airsim=start_airsim,
                    goal_airsim=goal_airsim,
                    show_markers=show_markers,
                    eye_distance=eye_distance,
                ),
                frames_dir / f"frame-{frame_index:05d}.png",
                width=width,
                height=height,
                scale=scale,
            )
            frame_index += 1

    best_controls = torch.cat([start[None], best_interior, goal[None]], dim=0)
    optimized_path = sample_path(best_controls, basis)
    optimized_trajectory = trajectory_from_path(nemo, optimized_path, cfg)
    optimized_terms = {
        **compute_raw_costs(optimized_trajectory, best_controls, cfg),
        "total": _weighted_total(
            compute_raw_costs(optimized_trajectory, best_controls, cfg), normalizers, cfg
        ),
    }

    return {
        "initial_path_xy": initial_path.detach().cpu().numpy().astype(np.float32),
        "optimized_path_xy": optimized_path.detach().cpu().numpy().astype(np.float32),
        "control_points_initial": controls_np.astype(np.float32),
        "control_points_optimized": best_controls.detach().cpu().numpy().astype(np.float32),
        "initial_diagnostics": initial_diagnostics,
        "optimized_diagnostics": make_diagnostics(optimized_trajectory, optimized_terms),
        "cost_history": history,
        "trajectory": _trajectory_to_numpy(optimized_trajectory),
        "num_frames": frame_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/path_opt_animation"))
    parser.add_argument("--frame-every", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=1300)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--eye-distance", type=float, default=0.4)
    parser.add_argument("--terrain-res", type=int, default=240)
    parser.add_argument("--video-filename", type=str, default="path_opt_animation.mp4")
    parser.add_argument("--show-markers", action="store_true", help="Show start and goal markers.")
    parser.add_argument(
        "--no-serve", action="store_true", help="Do not start an HTTP preview server."
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame-*.png"):
        old_frame.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(args.checkpoint.expanduser(), map_location=device).to(device)

    start_xy = np.array([0.0, 0.0], dtype=np.float64)
    goal_xy = np.array([430.0, 330.0], dtype=np.float64)
    start_airsim = ground_airsim(nemo, start_xy)
    goal_airsim = ground_airsim(nemo, goal_xy)
    start_ds = airsim_to_dataset(start_airsim)
    goal_ds = airsim_to_dataset(goal_airsim)
    astar_xy = build_astar_seed(
        nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
    )
    astar_airsim = dataset_to_airsim_path(path_xyz(nemo, astar_xy))
    cfg = build_planner_config()
    if args.num_iters is not None:
        cfg = TerrainAwarePlannerConfig(**{**asdict(cfg), "num_iters": int(args.num_iters)})

    print(f"Sampling terrain grid at resolution {args.terrain_res}...")
    terrain = sample_airsim_grid(nemo, res=int(args.terrain_res))
    print(f"Rendering animation frames to {frames_dir}...")
    result = run_animated_optimization(
        nemo=nemo,
        astar_xy=astar_xy,
        astar_airsim=astar_airsim,
        start_airsim=start_airsim,
        goal_airsim=goal_airsim,
        cfg=cfg,
        terrain=terrain,
        frames_dir=frames_dir,
        frame_every=int(args.frame_every),
        width=int(args.width),
        height=int(args.height),
        scale=float(args.scale),
        show_markers=bool(args.show_markers),
        eye_distance=float(args.eye_distance),
    )

    vehicle = VehicleModelConfig(
        nominal_speed_mps=cfg.nominal_speed,
        wheelbase_m=cfg.wheelbase_m,
        max_yaw_rate_radps=1.2,
        max_lateral_accel_mps2=2.0,
        max_longitudinal_accel_mps2=cfg.max_throttle_accel,
    )
    initial_airsim = dataset_to_airsim_path(path_xyz(nemo, result["initial_path_xy"]))
    optimized_control_airsim = dataset_to_airsim_path(path_xyz(nemo, result["optimized_path_xy"]))
    reference = generate_airsim_reference_trajectory(
        optimized_control_airsim, vehicle=vehicle, sample_spacing_m=2.0
    )
    optimized_airsim = np.asarray(reference["positions"], dtype=np.float32)
    initial_reference = generate_airsim_reference_trajectory(
        initial_airsim, vehicle=vehicle, sample_spacing_m=2.0
    )
    metrics = {
        "initial": {
            **asdict(result["initial_diagnostics"]),
            **trajectory_metrics(initial_reference),
        },
        "optimized": {**asdict(result["optimized_diagnostics"]), **trajectory_metrics(reference)},
    }
    payload = {
        "config": asdict(cfg),
        "checkpoint": str(args.checkpoint.expanduser()),
        "start_airsim": start_airsim.tolist(),
        "goal_airsim": goal_airsim.tolist(),
        "frame_every": int(args.frame_every),
        "eye_distance": float(args.eye_distance),
        "num_frames": int(result["num_frames"]),
        "metrics": metrics,
        "cost_history": result["cost_history"],
    }

    np.save(output_dir / "airsim_path_opt_astar_path.npy", astar_airsim)
    np.save(output_dir / "airsim_path_opt_initial_path.npy", initial_airsim)
    np.save(output_dir / "airsim_path_opt_control_path.npy", optimized_control_airsim)
    np.save(output_dir / "airsim_path_opt_path.npy", optimized_airsim)
    np.savez(output_dir / "airsim_path_opt_trajectory.npz", **reference)
    np.save(
        output_dir / "dataset_path_opt_control_points_initial.npy", result["control_points_initial"]
    )
    np.save(
        output_dir / "dataset_path_opt_control_points_optimized.npy",
        result["control_points_optimized"],
    )
    np.savez(output_dir / "dataset_path_opt_trajectory.npz", **result["trajectory"])
    (output_dir / "path_opt_animation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    frame_paths = sorted(frames_dir.glob("frame-*.png"))
    if not frame_paths:
        raise RuntimeError("No frames were generated.")
    video_path = output_dir / args.video_filename
    first_frame = np.asarray(media.read_image(frame_paths[0]))[..., :3]
    with media.VideoWriter(
        path=video_path, shape=first_frame.shape[:2], fps=int(args.fps), codec="h264"
    ) as writer:
        writer.add_image(first_frame)
        for frame_path in frame_paths[1:]:
            writer.add_image(np.asarray(media.read_image(frame_path))[..., :3])

    print(
        "initial: "
        f"total={result['initial_diagnostics'].total_cost:.3f}, "
        f"distance={metrics['initial']['distance_3d_m']:.1f} m, "
        f"time={metrics['initial']['time_s']:.1f} s"
    )
    print(
        "optimized: "
        f"total={result['optimized_diagnostics'].total_cost:.3f}, "
        f"distance={metrics['optimized']['distance_3d_m']:.1f} m, "
        f"time={metrics['optimized']['time_s']:.1f} s"
    )
    print(f"Saved {len(frame_paths)} frames to {frames_dir}")
    print(f"Saved video to {video_path}")

    if not args.no_serve:
        server_process, base_url = start_preview_server(output_dir)
        video_url = f"{base_url}/{quote(video_path.name)}"
        print(f"Video preview: {video_url}")
        print(f"Preview server PID: {server_process.pid}")
        webbrowser.open(video_url, new=2)


if __name__ == "__main__":
    main()
