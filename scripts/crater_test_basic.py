"""Crater-entry test using the basic slope-distance-control planner."""

from __future__ import annotations

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
from nemo.basic_terrain_planner import (
    BasicTerrainPlannerConfig,
    optimize_basic_terrain_path,
    path_xyz,
)
from nemo.physical_objective_planner import (
    PhysicalObjectivePlannerConfig,
    VehicleModelConfig,
    _build_astar_seed,
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


def build_basic_config() -> BasicTerrainPlannerConfig:
    return BasicTerrainPlannerConfig(
        num_control_points=26,
        num_samples=420,
        num_iters=1800,
        lr=0.45,
        w_length=0.1,
        w_slope=2400.0,
        w_control=0,
        w_control_points=0,
        bounds_margin=2.0,
    )


def trajectory_metrics(
    reference: dict[str, np.ndarray | dict[str, float]], diagnostics: dict[str, float]
) -> dict[str, float]:
    positions = np.asarray(reference["positions"], dtype=np.float64)
    curvature = np.asarray(reference["curvature"], dtype=np.float64)
    pitch = np.asarray(reference["pitch"], dtype=np.float64)
    speed = np.asarray(reference["speed"], dtype=np.float64)
    t = np.asarray(reference["t"], dtype=np.float64)
    return {
        **diagnostics,
        "distance_3d_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "time_s": float(t[-1]) if t.size else 0.0,
        "max_pitch_deg": float(np.degrees(np.max(np.abs(pitch)))) if pitch.size else 0.0,
        "max_curvature_1pm": float(np.max(np.abs(curvature))) if curvature.size else 0.0,
        "min_speed_mps": float(np.min(speed)) if speed.size else 0.0,
    }


def build_figure(
    nemo: Nemo,
    astar_airsim: np.ndarray,
    initial_airsim: np.ndarray,
    optimized_airsim: np.ndarray,
    metrics: dict[str, dict[str, float]],
) -> go.Figure:
    as_x, as_y, as_z, slope = sample_airsim_grid(nemo, res=240)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "table"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.74, 0.26],
        row_heights=[0.62, 0.38],
        subplot_titles=("Basic Planner Crater Entry", "Diagnostics", "Slope Map", "XY Paths"),
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
        (optimized_airsim, "basic optimized", "#22c55e", "lines", 8, None),
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
    row_names = ["initial", "optimized"]
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Path",
                    "Cost",
                    "Dist m",
                    "Time s",
                    "Mean slope deg",
                    "Max slope deg",
                    "Control",
                ],
                align="left",
            ),
            cells=dict(
                values=[
                    row_names,
                    [f"{metrics[k]['total_cost']:.2f}" for k in row_names],
                    [f"{metrics[k]['distance_3d_m']:.1f}" for k in row_names],
                    [f"{metrics[k]['time_s']:.1f}" for k in row_names],
                    [f"{np.degrees(np.arctan(metrics[k]['mean_slope'])):.1f}" for k in row_names],
                    [f"{np.degrees(np.arctan(metrics[k]['max_slope'])):.1f}" for k in row_names],
                    [f"{metrics[k]['control_effort']:.4f}" for k in row_names],
                ],
                align="left",
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=1300,
        width=1500,
        title_text="Crater Entry With Basic Terrain Planner",
        template="plotly_dark",
        margin=dict(l=45, r=80, t=95, b=45),
        legend=dict(orientation="h", x=0.5, y=1.04, xanchor="center", yanchor="bottom"),
        scene=dict(
            aspectmode="data",
            xaxis_title="AirSim X",
            yaxis_title="AirSim Y",
            zaxis_title="Elevation up (m)",
            yaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.3, y=-1.5, z=0.9)),
        ),
        xaxis=dict(title="AirSim X", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="AirSim Y", autorange="reversed"),
        xaxis2=dict(title="AirSim X", scaleanchor="y2", scaleratio=1),
        yaxis2=dict(title="AirSim Y", autorange="reversed"),
    )
    return fig


def main() -> None:
    output_dir = Path("outputs/crater_test_basic_400_300")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
        "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(checkpoint, map_location=device).to(device)
    crater_low_airsim = find_local_low_airsim(nemo, CRATER_CENTER_AIRSIM_XY)
    start_ds = airsim_to_dataset(AIRSIM_START)
    goal_ds = airsim_to_dataset(crater_low_airsim)

    astar_xy = build_astar_seed(
        nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
    )
    cfg = build_basic_config()
    result = optimize_basic_terrain_path(nemo, astar_xy, cfg)

    vehicle = VehicleModelConfig()
    astar_airsim = dataset_to_airsim_path(path_xyz(nemo, astar_xy))
    initial_airsim = dataset_to_airsim_path(path_xyz(nemo, result.initial_path_xy))
    optimized_control_airsim = dataset_to_airsim_path(path_xyz(nemo, result.optimized_path_xy))
    reference = generate_airsim_reference_trajectory(
        optimized_control_airsim, vehicle=vehicle, sample_spacing_m=2.0
    )
    optimized_airsim = np.asarray(reference["positions"], dtype=np.float32)

    initial_reference = generate_airsim_reference_trajectory(
        initial_airsim, vehicle=vehicle, sample_spacing_m=2.0
    )
    metrics = {
        "initial": trajectory_metrics(initial_reference, asdict(result.initial_diagnostics)),
        "optimized": trajectory_metrics(reference, asdict(result.optimized_diagnostics)),
    }
    payload = {
        "config": asdict(cfg),
        "start_airsim": AIRSIM_START.tolist(),
        "crater_center_airsim_xy": CRATER_CENTER_AIRSIM_XY.tolist(),
        "goal_airsim": crater_low_airsim.tolist(),
        "metrics": metrics,
        "cost_history": result.cost_history,
    }

    np.save(output_dir / "airsim_crater_basic_astar_path.npy", astar_airsim)
    np.save(output_dir / "airsim_crater_basic_initial_path.npy", initial_airsim)
    np.save(output_dir / "airsim_crater_basic_control_path.npy", optimized_control_airsim)
    np.save(output_dir / "airsim_crater_basic_path.npy", optimized_airsim)
    np.savez(output_dir / "airsim_crater_basic_trajectory.npz", **reference)
    np.save(
        output_dir / "dataset_crater_basic_control_points_initial.npy",
        result.control_points_initial,
    )
    np.save(
        output_dir / "dataset_crater_basic_control_points_optimized.npy",
        result.control_points_optimized,
    )
    (output_dir / "crater_basic_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fig = build_figure(nemo, astar_airsim, initial_airsim, optimized_airsim, metrics)
    html_path = output_dir / "crater_basic.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"

    print(f"Crater center AirSim XY: {CRATER_CENTER_AIRSIM_XY.tolist()}")
    print(f"Goal local low AirSim: {crater_low_airsim.tolist()}")
    print(
        "initial: "
        f"cost={result.initial_diagnostics.total_cost:.2f}, "
        f"distance={metrics['initial']['distance_3d_m']:.1f} m, "
        f"time={metrics['initial']['time_s']:.1f} s, "
        f"max_slope={np.degrees(np.arctan(result.initial_diagnostics.max_slope)):.1f} deg, "
        f"control={result.initial_diagnostics.control_effort:.4f}"
    )
    print(
        "optimized: "
        f"cost={result.optimized_diagnostics.total_cost:.2f}, "
        f"distance={metrics['optimized']['distance_3d_m']:.1f} m, "
        f"time={metrics['optimized']['time_s']:.1f} s, "
        f"max_slope={np.degrees(np.arctan(result.optimized_diagnostics.max_slope)):.1f} deg, "
        f"control={result.optimized_diagnostics.control_effort:.4f}"
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
