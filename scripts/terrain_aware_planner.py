"""Crater-entry test using the terrain-aware steering/throttle planner."""

from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from nemo import Nemo
from nemo.airsim_trajectory import generate_airsim_reference_trajectory
from nemo.physical_objective_planner import (
    PhysicalObjectivePlannerConfig,
    VehicleModelConfig,
    _build_astar_seed,
)
from nemo.terrain_aware_planner import (
    TerrainAwarePlannerConfig,
    cumulative_lengths,
    initialize_control_points,
    optimize_terrain_aware_path,
    path_xyz,
)
from scripts.compare_physical_objectives import (
    dataset_to_airsim_path,
    sample_airsim_grid,
)
from scripts.crater_test import (
    CRATER_CENTER_AIRSIM_XY,
    airsim_to_dataset,
    start_preview_server,
)


PATH_PLOT_Z_OFFSET_M = 2.0
DEFAULT_CHECKPOINT = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
)


def build_astar_seed(
    nemo: Nemo, start_xy: tuple[float, float], goal_xy: tuple[float, float]
) -> np.ndarray:
    cfg = PhysicalObjectivePlannerConfig(
        astar_max_slope_deg=89.0,
        astar_slope_weight=8.0,
        astar_grid_resolution_x=320,
        astar_grid_resolution_y=320,
        astar_slope_reference_deg=10.0,
        astar_slope_exponent=3.0,
        batch_size=65536,
    )
    return _build_astar_seed(nemo, start_xy=start_xy, goal_xy=goal_xy, cfg=cfg)


def build_planner_config(
    num_control_points: int, num_iters: int, gradient_mode: str = "finite_difference"
) -> TerrainAwarePlannerConfig:
    return TerrainAwarePlannerConfig(
        num_control_points=int(num_control_points),
        num_samples=1000,
        num_iters=int(num_iters),
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
        w_smoothness=0.01,
        gradient_mode=gradient_mode,
        bounds_margin=2.0,
        normalize_costs=True,
        verbose=True,
    )


def parse_xy(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected XY as 'x,y'.")
    try:
        return np.asarray([float(parts[0]), float(parts[1])], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected XY as numeric 'x,y'.") from exc


def ground_airsim(nemo: Nemo, xy_airsim: np.ndarray) -> np.ndarray:
    ds_xy = airsim_to_dataset(np.append(xy_airsim, 0.0))[:2]
    xyz_ds = path_xyz(nemo, ds_xy[None])[0]
    return dataset_to_airsim_path(xyz_ds[None])[0]


def num_controls_for_spacing(
    path_xy: np.ndarray,
    spacing_m: float,
    *,
    min_control_points: int,
    max_control_points: int,
) -> int:
    total_length = float(cumulative_lengths(path_xy)[-1])
    spacing = max(float(spacing_m), 1e-6)
    total_controls = int(np.ceil(total_length / spacing)) + 1
    interior_controls = max(total_controls - 2, 0)
    return int(np.clip(interior_controls, int(min_control_points), int(max_control_points)))


def insert_adaptive_controls(
    controls: np.ndarray,
    sample_xy: np.ndarray,
    trajectory: dict[str, np.ndarray],
    cfg: TerrainAwarePlannerConfig,
    *,
    extra_controls: int,
    score_percentile: float,
    max_control_points: int,
) -> np.ndarray:
    controls = np.asarray(controls, dtype=np.float32)
    extra_budget = min(
        max(int(extra_controls), 0),
        max(int(max_control_points) - max(controls.shape[0] - 2, 0), 0),
    )
    if extra_budget <= 0 or controls.shape[0] < 2 or sample_xy.shape[0] < 2:
        return controls

    slope = np.abs(np.asarray(trajectory["slope_mag"], dtype=np.float64))
    uphill = np.maximum(np.asarray(trajectory["along_track_slope_rad"], dtype=np.float64), 0.0)
    downhill = np.maximum(-np.asarray(trajectory["along_track_slope_rad"], dtype=np.float64), 0.0)
    crosstrack = np.abs(np.asarray(trajectory["cross_track_slope_rad"], dtype=np.float64))
    steering = np.abs(np.asarray(trajectory["steering"], dtype=np.float64))
    throttle = np.abs(np.asarray(trajectory["throttle"], dtype=np.float64))
    point_score = (
        slope / max(float(cfg.uphill_limit_rad), 1e-8)
        + uphill / max(float(cfg.uphill_limit_rad), 1e-8)
        + downhill / max(float(cfg.downhill_limit_rad), 1e-8)
        + crosstrack / max(float(cfg.crosstrack_limit_rad), 1e-8)
        + steering / max(float(cfg.max_steering_rad), 1e-8)
        + throttle
    )
    sample_score = 0.5 * (point_score[:-1] + point_score[1:])
    if sample_score.size == 0 or not np.any(np.isfinite(sample_score)):
        return controls

    control_lengths = cumulative_lengths(controls)
    sample_lengths = cumulative_lengths(sample_xy)
    sample_mid_lengths = 0.5 * (sample_lengths[:-1] + sample_lengths[1:])
    segment_index = np.searchsorted(control_lengths, sample_mid_lengths, side="right") - 1
    segment_index = np.clip(segment_index, 0, controls.shape[0] - 2)

    control_scores = np.full(controls.shape[0] - 1, -np.inf, dtype=np.float64)
    np.maximum.at(control_scores, segment_index, sample_score)
    finite_scores = control_scores[np.isfinite(control_scores)]
    if finite_scores.size == 0:
        return controls

    threshold = float(np.percentile(finite_scores, float(score_percentile)))
    candidate_segments = np.flatnonzero(control_scores >= threshold)
    if candidate_segments.size == 0:
        candidate_segments = np.argsort(control_scores)[-extra_budget:]
    else:
        order = np.argsort(control_scores[candidate_segments])[::-1]
        candidate_segments = candidate_segments[order[:extra_budget]]
    candidate_segments = np.sort(np.unique(candidate_segments))

    refined: list[np.ndarray] = []
    insert_set = set(int(idx) for idx in candidate_segments)
    for idx in range(controls.shape[0] - 1):
        refined.append(controls[idx])
        if idx in insert_set:
            refined.append(0.5 * (controls[idx] + controls[idx + 1]))
    refined.append(controls[-1])
    return np.asarray(refined, dtype=np.float32)


def run_planner_with_refinement(
    nemo: Nemo,
    astar_xy: np.ndarray,
    cfg: TerrainAwarePlannerConfig,
    *,
    control_spacing_m: float,
    min_control_points: int,
    max_control_points: int,
    adaptive_refinement_passes: int,
    adaptive_extra_controls: int,
    adaptive_score_percentile: float,
):
    num_control_points = num_controls_for_spacing(
        astar_xy,
        control_spacing_m,
        min_control_points=min_control_points,
        max_control_points=max_control_points,
    )
    cfg = TerrainAwarePlannerConfig(**{**asdict(cfg), "num_control_points": num_control_points})
    controls = initialize_control_points(astar_xy, cfg.num_control_points)
    result = None
    refinement_history = []

    for refinement_pass in range(int(adaptive_refinement_passes) + 1):
        cfg = TerrainAwarePlannerConfig(
            **{**asdict(cfg), "num_control_points": controls.shape[0] - 2}
        )
        result = optimize_terrain_aware_path(nemo, astar_xy, cfg, initial_control_points=controls)
        refinement_history.append(
            {
                "pass": refinement_pass,
                "num_control_points": int(controls.shape[0] - 2),
                "total_cost": float(result.optimized_diagnostics.total_cost),
                "max_slope_deg": float(result.optimized_diagnostics.max_slope_deg),
                "max_abs_steering_deg": float(result.optimized_diagnostics.max_abs_steering_deg),
                "max_abs_throttle": float(result.optimized_diagnostics.max_abs_throttle),
            }
        )
        if refinement_pass >= int(adaptive_refinement_passes):
            break
        refined_controls = insert_adaptive_controls(
            result.control_points_optimized,
            result.optimized_path_xy,
            result.trajectory,
            cfg,
            extra_controls=adaptive_extra_controls,
            score_percentile=adaptive_score_percentile,
            max_control_points=max_control_points,
        )
        if refined_controls.shape[0] == result.control_points_optimized.shape[0]:
            break
        controls = refined_controls

    if result is None:
        raise RuntimeError("Planner did not produce a result.")
    return result, cfg, refinement_history


def trajectory_metrics(reference: dict[str, np.ndarray | dict[str, float]]) -> dict[str, float]:
    positions = np.asarray(reference["positions"], dtype=np.float64)
    curvature = np.asarray(reference["curvature"], dtype=np.float64)
    pitch = np.asarray(reference["pitch"], dtype=np.float64)
    speed = np.asarray(reference["speed"], dtype=np.float64)
    yaw_rate = np.asarray(reference["yaw_rate"], dtype=np.float64)
    accel = np.asarray(reference["accel"], dtype=np.float64)
    t = np.asarray(reference["t"], dtype=np.float64)
    return {
        "distance_3d_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "time_s": float(t[-1]) if t.size else 0.0,
        "max_pitch_deg": float(np.degrees(np.max(np.abs(pitch)))) if pitch.size else 0.0,
        "max_curvature_1pm": float(np.max(np.abs(curvature))) if curvature.size else 0.0,
        "max_yaw_rate_radps": float(np.max(np.abs(yaw_rate))) if yaw_rate.size else 0.0,
        "max_accel_mps2": float(np.max(np.abs(accel))) if accel.size else 0.0,
        "min_speed_mps": float(np.min(speed)) if speed.size else 0.0,
    }


def build_figure(
    nemo: Nemo,
    astar_airsim: np.ndarray,
    initial_airsim: np.ndarray,
    optimized_airsim: np.ndarray,
    start_airsim: np.ndarray,
    goal_airsim: np.ndarray,
    metrics: dict[str, dict[str, float]],
    terrain_res: int = 240,
    show_markers: bool = False,
) -> go.Figure:
    as_x, as_y, as_z, slope = sample_airsim_grid(nemo, res=int(terrain_res))
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "table"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.74, 0.26],
        row_heights=[0.62, 0.38],
        subplot_titles=("Terrain-Aware Crater Entry", "Diagnostics", "Slope Map", "XY Paths"),
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
    )
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
            colorbar=dict(title="Elevation<br>up (m)", x=0.745, y=0.73, len=0.38, thickness=12),
            name="NEMo surface",
        ),
        row=1,
        col=1,
    )
    # opt_color = "#ff8800"
    # opt_color = "#ff9ce4"
    opt_color = "#ef2dfc"
    paths = [
        (astar_airsim, "A* seed", "#ff0000", "lines+markers", 5, "dash"),
        (initial_airsim, "initial spline", "#94a3b8", "lines", 5, "dot"),
        (optimized_airsim, "terrain-aware optimized", opt_color, "lines", 8, None),
    ]
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
            row=1,
            col=1,
        )

    # Add start and goal markers
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
            row=1,
            col=1,
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
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Heatmap(
            x=as_x,
            y=as_y,
            z=slope,
            colorscale="Viridis",
            zmin=0,
            zmax=45,
            colorbar=dict(title="Slope<br>(deg)", x=0.745, y=0.19, len=0.26, thickness=12),
            name="slope",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for path, name, color, mode, _, dash in paths:
        line = dict(color=color, width=3)
        if dash is not None:
            line["dash"] = dash
        for row, col in ((2, 1), (2, 2)):
            fig.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode=mode,
                    line=line,
                    marker=dict(color=color, size=4),
                    name=name,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    # Add 2D start/goal markers
    if show_markers:
        for row, col in ((2, 1), (2, 2)):
            fig.add_trace(
                go.Scatter(
                    x=[start_airsim[0]],
                    y=[start_airsim[1]],
                    mode="markers",
                    marker=dict(color="#00ffff", size=10, symbol="circle"),
                    name="Start",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[goal_airsim[0]],
                    y=[goal_airsim[1]],
                    mode="markers",
                    marker=dict(color="#ff00ff", size=12, symbol="diamond"),
                    name="Goal",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    rows = ["initial", "optimized"]
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Path",
                    "Total",
                    "Length",
                    "Uphill",
                    "Downhill",
                    "Cross",
                    "Throttle",
                    "Steering",
                    "Dist m",
                    "Time s",
                    "Max slope",
                    "Max steer",
                ],
                align="left",
            ),
            cells=dict(
                values=[
                    rows,
                    [f"{metrics[k]['total_cost']:.3f}" for k in rows],
                    [f"{metrics[k]['length_cost']:.2f}" for k in rows],
                    [f"{metrics[k]['uphill_cost']:.2f}" for k in rows],
                    [f"{metrics[k]['downhill_cost']:.2f}" for k in rows],
                    [f"{metrics[k]['crosstrack_cost']:.2f}" for k in rows],
                    [f"{metrics[k]['throttle_cost']:.2f}" for k in rows],
                    [f"{metrics[k]['steering_cost']:.2f}" for k in rows],
                    [f"{metrics[k]['distance_3d_m']:.1f}" for k in rows],
                    [f"{metrics[k]['time_s']:.1f}" for k in rows],
                    [f"{metrics[k]['max_slope_deg']:.1f}" for k in rows],
                    [f"{metrics[k]['max_abs_steering_deg']:.1f}" for k in rows],
                ],
                align="left",
            ),
        ),
        row=1,
        col=2,
    )

    clean_axis = dict(
        showbackground=False,
        backgroundcolor="black",
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
        title="",
    )

    fig.update_layout(
        height=1300,
        width=1500,
        title_text="Crater Entry With Terrain-Aware Steering/Throttle Planner",
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=45, r=80, t=95, b=45),
        legend=dict(orientation="h", x=0.5, y=1.04, xanchor="center", yanchor="bottom"),
        scene=dict(
            aspectmode="data",
            xaxis=clean_axis,
            yaxis=dict(autorange="reversed", **clean_axis),
            zaxis=clean_axis,
            camera=dict(eye=dict(x=1.3, y=-1.5, z=0.9)),
        ),
        xaxis=dict(scaleanchor="y", scaleratio=1, **clean_axis),
        yaxis=dict(autorange="reversed", **clean_axis),
        xaxis2=dict(scaleanchor="y2", scaleratio=1, **clean_axis),
        yaxis2=dict(autorange="reversed", **clean_axis),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/terrain_aware_planner"))
    parser.add_argument(
        "--start",
        type=parse_xy,
        default=np.array([-250.0, -550.0]),
        help="AirSim start XY as 'x,y'.",
    )
    parser.add_argument(
        "--goal", type=parse_xy, default=np.array([1200.0, 900.0]), help="AirSim goal XY as 'x,y'."
    )
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--terrain-res", type=int, default=240)
    parser.add_argument("--control-point-spacing-m", type=float, default=20.0)
    parser.add_argument("--min-control-points", type=int, default=4)
    parser.add_argument("--max-control-points", type=int, default=200)
    parser.add_argument(
        "--gradients",
        choices=("fd", "autograd"),
        default="fd",
        help="Terrain slope source: finite differences ('fd') or PyTorch autograd.",
    )
    parser.add_argument(
        "--adaptive-refinement-passes",
        type=int,
        default=0,
        help="Number of optimize-score-insert-rerun refinement passes.",
    )
    parser.add_argument(
        "--adaptive-extra-controls",
        type=int,
        default=8,
        help="Maximum extra interior control points inserted per refinement pass.",
    )
    parser.add_argument(
        "--adaptive-score-percentile",
        type=float,
        default=85.0,
        help="Only control segments at or above this score percentile are refined.",
    )
    parser.add_argument(
        "--no-serve", action="store_true", help="Do not start an HTTP preview server."
    )
    parser.add_argument("--show-markers", action="store_true", help="Show start and goal markers.")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(args.checkpoint.expanduser(), map_location=device).to(device)

    start_airsim = ground_airsim(nemo, args.start)
    goal_airsim = ground_airsim(nemo, args.goal)

    start_ds = airsim_to_dataset(start_airsim)
    goal_ds = airsim_to_dataset(goal_airsim)

    astar_xy = build_astar_seed(
        nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
    )
    gradient_mode = "finite_difference" if args.gradients == "fd" else "autograd"
    cfg = build_planner_config(
        num_control_points=0,
        num_iters=int(args.num_iters),
        gradient_mode=gradient_mode,
    )
    result, cfg, refinement_history = run_planner_with_refinement(
        nemo,
        astar_xy,
        cfg,
        control_spacing_m=float(args.control_point_spacing_m),
        min_control_points=int(args.min_control_points),
        max_control_points=int(args.max_control_points),
        adaptive_refinement_passes=int(args.adaptive_refinement_passes),
        adaptive_extra_controls=int(args.adaptive_extra_controls),
        adaptive_score_percentile=float(args.adaptive_score_percentile),
    )

    vehicle = VehicleModelConfig(
        nominal_speed_mps=cfg.nominal_speed,
        wheelbase_m=cfg.wheelbase_m,
        max_yaw_rate_radps=1.2,
        max_lateral_accel_mps2=2.0,
        max_longitudinal_accel_mps2=cfg.max_throttle_accel,
    )
    astar_airsim = dataset_to_airsim_path(path_xyz(nemo, astar_xy))
    initial_airsim = dataset_to_airsim_path(result.initial_path_xyz)
    optimized_control_airsim = dataset_to_airsim_path(result.optimized_path_xyz)
    reference = generate_airsim_reference_trajectory(
        optimized_control_airsim, vehicle=vehicle, sample_spacing_m=2.0
    )
    optimized_airsim = np.asarray(reference["positions"], dtype=np.float32)
    initial_reference = generate_airsim_reference_trajectory(
        initial_airsim, vehicle=vehicle, sample_spacing_m=2.0
    )

    metrics = {
        "initial": {**asdict(result.initial_diagnostics), **trajectory_metrics(initial_reference)},
        "optimized": {**asdict(result.optimized_diagnostics), **trajectory_metrics(reference)},
    }
    payload = {
        "config": asdict(cfg),
        "checkpoint": str(args.checkpoint.expanduser()),
        "start_airsim": start_airsim.tolist(),
        "start_xy_airsim": np.asarray(args.start, dtype=float).tolist(),
        "crater_center_airsim_xy": CRATER_CENTER_AIRSIM_XY.tolist(),
        "goal_airsim": goal_airsim.tolist(),
        "goal_xy_airsim": np.asarray(args.goal, dtype=float).tolist(),
        "control_point_spacing_m": float(args.control_point_spacing_m),
        "adaptive_refinement_passes": int(args.adaptive_refinement_passes),
        "adaptive_extra_controls": int(args.adaptive_extra_controls),
        "adaptive_score_percentile": float(args.adaptive_score_percentile),
        "terrain_res": int(args.terrain_res),
        "refinement_history": refinement_history,
        "metrics": metrics,
        "cost_history": result.cost_history,
    }

    np.save(output_dir / "airsim_terrain_aware_astar_path.npy", astar_airsim)
    np.save(output_dir / "airsim_terrain_aware_initial_path.npy", initial_airsim)
    np.save(output_dir / "airsim_terrain_aware_control_path.npy", optimized_control_airsim)
    np.save(output_dir / "airsim_terrain_aware_path.npy", optimized_airsim)
    np.savez(output_dir / "airsim_terrain_aware_trajectory.npz", **reference)
    np.save(
        output_dir / "dataset_terrain_aware_control_points_initial.npy",
        result.control_points_initial,
    )
    np.save(
        output_dir / "dataset_terrain_aware_control_points_optimized.npy",
        result.control_points_optimized,
    )
    np.savez(output_dir / "dataset_terrain_aware_trajectory.npz", **result.trajectory)
    (output_dir / "terrain_aware_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fig = build_figure(
        nemo,
        astar_airsim,
        initial_airsim,
        optimized_airsim,
        start_airsim,
        goal_airsim,
        metrics,
        terrain_res=int(args.terrain_res),
        show_markers=args.show_markers,
    )
    html_path = output_dir / "terrain_aware_planner.html"
    fig.write_html(html_path)

    print(
        "initial: "
        f"total={result.initial_diagnostics.total_cost:.3f}, "
        f"distance={metrics['initial']['distance_3d_m']:.1f} m, "
        f"time={metrics['initial']['time_s']:.1f} s, "
        f"max_slope={result.initial_diagnostics.max_slope_deg:.1f} deg, "
        f"max_steer={result.initial_diagnostics.max_abs_steering_deg:.1f} deg, "
        f"max_throttle={result.initial_diagnostics.max_abs_throttle:.2f}"
    )
    print(
        "optimized: "
        f"total={result.optimized_diagnostics.total_cost:.3f}, "
        f"distance={metrics['optimized']['distance_3d_m']:.1f} m, "
        f"time={metrics['optimized']['time_s']:.1f} s, "
        f"max_slope={result.optimized_diagnostics.max_slope_deg:.1f} deg, "
        f"max_steer={result.optimized_diagnostics.max_abs_steering_deg:.1f} deg, "
        f"max_throttle={result.optimized_diagnostics.max_abs_throttle:.2f}"
    )
    print(f"Saved outputs to {output_dir}")
    print(
        "controls: "
        f"initial={result.control_points_initial.shape[0] - 2}, "
        f"optimized={result.control_points_optimized.shape[0] - 2}, "
        f"spacing={float(args.control_point_spacing_m):.1f} m"
    )
    if not args.no_serve:
        server_process, base_url = start_preview_server(output_dir.resolve())
        plot_url = f"{base_url}/{quote(html_path.name)}"
        print(f"Plot preview: {plot_url}")
        print(f"Preview server PID: {server_process.pid}")
        webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
