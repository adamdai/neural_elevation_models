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
    optimize_terrain_aware_path,
    path_xyz,
)
from scripts.compare_physical_objectives import (
    AIRSIM_START,
    AIRSIM_SPIRAL_CENTER,
    AIRSIM_Z_OFFSET,
    dataset_to_airsim_path,
    sample_airsim_grid,
)
from scripts.crater_test import (
    CRATER_CENTER_AIRSIM_XY,
    airsim_to_dataset,
    find_local_low_airsim,
    start_preview_server,
)


PATH_PLOT_Z_OFFSET_M = 2.0


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


def build_planner_config() -> TerrainAwarePlannerConfig:
    return TerrainAwarePlannerConfig(
        num_control_points=40,
        num_samples=460,
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
    show_markers: bool = False,
) -> go.Figure:
    as_x, as_y, as_z, slope = sample_airsim_grid(nemo, res=240)
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
    paths = [
        (astar_airsim, "A* seed", "#ff0000", "lines+markers", 5, "dash"),
        (initial_airsim, "initial spline", "#94a3b8", "lines", 5, "dot"),
        (optimized_airsim, "terrain-aware optimized", "#ff8800", "lines", 8, None),
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
    parser.add_argument("--show-markers", action="store_true", help="Show start and goal markers.")
    args = parser.parse_args()

    output_dir = Path("outputs/crater_test_terrain_aware_400_300")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
        "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(checkpoint, map_location=device).to(device)

    # --- Start and Goal Selection ---
    # Define XY coordinates; Z will be sampled from NEMo elevation.
    # START_XY = AIRSIM_START[:2]
    START_XY = np.array([0.0, 0.0])

    # GOAL_XY = find_local_low_airsim(nemo, CRATER_CENTER_AIRSIM_XY)[:2]
    # GOAL_XY = np.array([670.0, -53.0])
    # GOAL_XY = np.array([430.0, 330.0])
    GOAL_XY = np.array([1000.0, 330.0])

    def ground_airsim(xy: np.ndarray) -> np.ndarray:
        ds_xy = airsim_to_dataset(np.append(xy, 0.0))[:2]
        xyz_ds = path_xyz(nemo, ds_xy[None])[0]
        return dataset_to_airsim_path(xyz_ds[None])[0]

    start_airsim = ground_airsim(START_XY)
    goal_airsim = ground_airsim(GOAL_XY)

    start_ds = airsim_to_dataset(start_airsim)
    goal_ds = airsim_to_dataset(goal_airsim)

    astar_xy = build_astar_seed(
        nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
    )
    cfg = build_planner_config()
    result = optimize_terrain_aware_path(nemo, astar_xy, cfg)

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
        "start_airsim": start_airsim.tolist(),
        "crater_center_airsim_xy": CRATER_CENTER_AIRSIM_XY.tolist(),
        "goal_airsim": goal_airsim.tolist(),
        "metrics": metrics,
        "cost_history": result.cost_history,
    }

    np.save(output_dir / "airsim_crater_terrain_aware_astar_path.npy", astar_airsim)
    np.save(output_dir / "airsim_crater_terrain_aware_initial_path.npy", initial_airsim)
    np.save(output_dir / "airsim_crater_terrain_aware_control_path.npy", optimized_control_airsim)
    np.save(output_dir / "airsim_crater_terrain_aware_path.npy", optimized_airsim)
    np.savez(output_dir / "airsim_crater_terrain_aware_trajectory.npz", **reference)
    np.save(
        output_dir / "dataset_crater_terrain_aware_control_points_initial.npy",
        result.control_points_initial,
    )
    np.save(
        output_dir / "dataset_crater_terrain_aware_control_points_optimized.npy",
        result.control_points_optimized,
    )
    np.savez(output_dir / "dataset_crater_terrain_aware_trajectory.npz", **result.trajectory)
    (output_dir / "crater_terrain_aware_metrics.json").write_text(
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
        show_markers=args.show_markers,
    )
    html_path = output_dir / "crater_terrain_aware.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"

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
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
