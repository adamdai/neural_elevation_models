"""Compare distance, time, and energy objectives with shared safety constraints."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
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
    SafetyConstraintConfig,
    VehicleModelConfig,
    plan_physical_objective_path,
)


AIRSIM_START = np.array([0.0, 0.0, 0.0])
AIRSIM_GOAL = np.array([1050.50, 323.84, -19.73])
AIRSIM_SPIRAL_CENTER = np.array([524.38, 168.34, 0.0])
AIRSIM_Z_OFFSET = 24.15
PATH_PLOT_Z_OFFSET_M = 8.0


def airsim_to_dataset(p_airsim: np.ndarray) -> np.ndarray:
    p_dataset = p_airsim.copy()
    p_dataset[0] = p_airsim[1] - AIRSIM_SPIRAL_CENTER[1]
    p_dataset[1] = p_airsim[0] - AIRSIM_SPIRAL_CENTER[0]
    p_dataset[2] = p_airsim[2] + AIRSIM_Z_OFFSET
    return p_dataset


def parse_airsim_point(values: list[float] | None, default: np.ndarray) -> np.ndarray:
    if values is None:
        return default.astype(np.float64)
    if len(values) == 2:
        return np.array([values[0], values[1], 0.0], dtype=np.float64)
    if len(values) == 3:
        return np.array(values, dtype=np.float64)
    raise ValueError("AirSim point must be provided as X Y or X Y Z.")


def dataset_to_airsim_path(path_dataset: np.ndarray) -> np.ndarray:
    path_dataset = np.asarray(path_dataset, dtype=np.float64)
    path_airsim = path_dataset.copy()
    path_airsim[:, 0] = path_dataset[:, 1] + AIRSIM_SPIRAL_CENTER[0]
    path_airsim[:, 1] = path_dataset[:, 0] + AIRSIM_SPIRAL_CENTER[1]
    path_airsim[:, 2] = path_dataset[:, 2] - AIRSIM_Z_OFFSET
    return path_airsim.astype(np.float32)


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def start_preview_server(directory: Path, host: str = "127.0.0.1") -> tuple[subprocess.Popen[bytes], str]:
    port = find_free_port(host)
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", host, "--directory", str(directory)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process, f"http://{host}:{port}"


def sample_airsim_grid(nemo: Nemo, res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    as_x_axis = np.linspace(AIRSIM_SPIRAL_CENTER[0] - 900, AIRSIM_SPIRAL_CENTER[0] + 900, res)
    as_y_axis = np.linspace(AIRSIM_SPIRAL_CENTER[1] - 900, AIRSIM_SPIRAL_CENTER[1] + 900, res)
    as_xx, as_yy = np.meshgrid(as_x_axis, as_y_axis, indexing="xy")
    ds_xx = as_yy - AIRSIM_SPIRAL_CENTER[1]
    ds_yy = as_xx - AIRSIM_SPIRAL_CENTER[0]
    xy_ds = np.stack([ds_xx.reshape(-1), ds_yy.reshape(-1)], axis=-1)
    heights = []
    grads = []
    for start in range(0, len(xy_ds), 65536):
        batch = torch.as_tensor(xy_ds[start : start + 65536], dtype=torch.float32, device=nemo.device)
        heights.append(nemo.h(batch).detach().cpu().numpy())
        grads.append(nemo.grad(batch).detach().cpu().numpy())
    as_z = np.concatenate(heights, axis=0).reshape(res, res) - AIRSIM_Z_OFFSET
    grad_ds = np.concatenate(grads, axis=0).reshape(res, res, 2)
    slope_deg = np.degrees(np.arctan(np.linalg.norm(grad_ds, axis=-1)))
    return as_x_axis, as_y_axis, as_z, slope_deg


def build_figure(nemo: Nemo, paths: dict[str, np.ndarray], metrics: dict[str, dict[str, float]], res: int) -> go.Figure:
    as_x, as_y, as_z, slope = sample_airsim_grid(nemo, res=res)
    colors = {"distance": "#22c55e", "time": "#38bdf8", "energy": "#f97316", "astar": "#ff0000"}
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "table"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.74, 0.26],
        row_heights=[0.62, 0.38],
        subplot_titles=("Optimized Paths", "Diagnostics", "Slope Map", "XY Paths"),
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
    )
    zmin = float(np.nanpercentile(as_z, 1.0))
    zmax = float(np.nanpercentile(as_z, 99.0))
    fig.add_trace(
        go.Surface(
            x=as_x,
            y=as_y,
            z=as_z,
            surfacecolor=as_z,
            colorscale="Viridis",
            cmin=zmin,
            cmax=zmax,
            showscale=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            colorbar=dict(title="Elevation<br>up (m)", x=0.745, y=0.73, len=0.38, thickness=12),
            name="NEMo surface",
        ),
        row=1,
        col=1,
    )
    astar = paths["astar"]
    fig.add_trace(
        go.Scatter3d(
            x=astar[:, 0],
            y=astar[:, 1],
            z=astar[:, 2] + PATH_PLOT_Z_OFFSET_M,
            mode="lines+markers",
            line=dict(color=colors["astar"], width=5, dash="dash"),
            marker=dict(color=colors["astar"], size=3),
            name="A* seed",
        ),
        row=1,
        col=1,
    )
    for name in ("distance", "time", "energy"):
        path = paths[name]
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2] + PATH_PLOT_Z_OFFSET_M,
                mode="lines",
                line=dict(color=colors[name], width=7),
                marker=dict(color=colors[name], size=4),
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
            zmax=25,
            colorbar=dict(title="Slope<br>(deg)", x=0.745, y=0.19, len=0.26, thickness=12),
            name="slope",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for name, path in paths.items():
        if name == "astar":
            line = dict(color=colors[name], width=3, dash="dash")
            mode = "lines+markers"
        else:
            line = dict(color=colors[name], width=4)
            mode = "lines"
        for row, col in ((2, 1), (2, 2)):
            fig.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode=mode,
                    line=line,
                    marker=dict(color=colors[name], size=4),
                    name=name,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    table_rows = ["distance", "time", "energy"]
    fig.add_trace(
        go.Table(
            header=dict(values=["Objective", "Dist m", "Time s", "Energy kJ", "Max slope", "Max curv"], align="left"),
            cells=dict(
                values=[
                    table_rows,
                    [f"{metrics[k]['distance_3d_m']:.1f}" for k in table_rows],
                    [f"{metrics[k]['time_s']:.1f}" for k in table_rows],
                    [f"{metrics[k]['energy_j'] / 1000.0:.1f}" for k in table_rows],
                    [f"{metrics[k]['max_slope_deg']:.1f}" for k in table_rows],
                    [f"{metrics[k]['max_curvature_1pm']:.3f}" for k in table_rows],
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
        title_text="Differentiable Physical Objective Planning",
        template="plotly_dark",
        margin=dict(l=45, r=80, t=95, b=45),
        legend=dict(orientation="h", x=0.5, y=1.04, xanchor="center", yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        scene=dict(
            aspectmode="data",
            xaxis_title="AirSim X (Right)",
            yaxis_title="AirSim Y (Down)",
            zaxis_title="Elevation up (m)",
            yaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.3, y=-1.5, z=0.9)),
        ),
        xaxis=dict(title="AirSim X (Right)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="AirSim Y (Down)", autorange="reversed"),
        xaxis2=dict(title="AirSim X (Right)", scaleanchor="y2", scaleratio=1),
        yaxis2=dict(title="AirSim Y (Down)", autorange="reversed"),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physical_objective_comparison"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num-waypoints", type=int, default=80)
    parser.add_argument("--surface-resolution", type=int, default=220)
    parser.add_argument("--trajectory-sample-spacing", type=float, default=2.0)
    parser.add_argument("--start-airsim", type=float, nargs="+", default=None, help="Start in AirSim frame: X Y [Z].")
    parser.add_argument("--goal-airsim", type=float, nargs="+", default=None, help="Goal in AirSim frame: X Y [Z].")
    parser.add_argument("--max-slope-deg", type=float, default=25.0)
    args = parser.parse_args()

    nemo = Nemo.load_checkpoint(args.checkpoint, map_location=args.device).to(args.device)
    start_airsim = parse_airsim_point(args.start_airsim, AIRSIM_START)
    goal_airsim = parse_airsim_point(args.goal_airsim, AIRSIM_GOAL)
    start_ds = airsim_to_dataset(start_airsim)
    goal_ds = airsim_to_dataset(goal_airsim)
    vehicle = VehicleModelConfig()
    safety = SafetyConstraintConfig(max_slope_deg=float(args.max_slope_deg))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, np.ndarray] = {}
    display_paths: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    configs: dict[str, dict[str, object]] = {}
    astar_written = False
    for objective in ("distance", "time", "energy"):
        cfg = PhysicalObjectivePlannerConfig(
            objective=objective,
            num_waypoints=int(args.num_waypoints),
            optimize_iterations=int(args.iterations),
            astar_max_slope_deg=float(args.max_slope_deg),
            vehicle=vehicle,
            safety=safety,
        )
        result = plan_physical_objective_path(
            nemo,
            start_xy=(float(start_ds[0]), float(start_ds[1])),
            goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
            config=cfg,
        )
        if not astar_written:
            paths["astar"] = dataset_to_airsim_path(result.astar_path_xyz)
            display_paths["astar"] = paths["astar"]
            np.save(output_dir / "airsim_astar_seed.npy", paths["astar"])
            astar_written = True
        paths[objective] = dataset_to_airsim_path(result.optimized_path_xyz)
        reference = generate_airsim_reference_trajectory(
            paths[objective],
            vehicle=vehicle,
            sample_spacing_m=float(args.trajectory_sample_spacing),
        )
        dense_path = np.asarray(reference["positions"], dtype=np.float32)
        display_paths[objective] = dense_path
        np.save(output_dir / f"airsim_{objective}_path.npy", dense_path)
        np.save(output_dir / f"airsim_{objective}_control_path.npy", paths[objective])
        np.savez(output_dir / f"airsim_{objective}_trajectory.npz", **reference)
        metrics[objective] = asdict(result.optimized_diagnostics)
        metrics[f"{objective}_initial"] = asdict(result.initial_diagnostics)
        configs[objective] = asdict(cfg)
        print(
            f"{objective}: distance={result.optimized_diagnostics.distance_3d_m:.1f} m, "
            f"time={result.optimized_diagnostics.time_s:.1f} s, "
            f"energy={result.optimized_diagnostics.energy_j / 1000.0:.1f} kJ"
        )

    (output_dir / "physical_objective_metrics.json").write_text(
        json.dumps({"metrics": metrics, "configs": configs}, indent=2),
        encoding="utf-8",
    )
    fig = build_figure(nemo, display_paths, metrics, int(args.surface_resolution))
    html_path = output_dir / "physical_objective_comparison.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"
    print(f"Saved outputs to {output_dir}")
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    opened = webbrowser.open(plot_url, new=2)
    print(f"Auto-open requested: {'opened' if opened else 'no browser reported success'}")


if __name__ == "__main__":
    main()
