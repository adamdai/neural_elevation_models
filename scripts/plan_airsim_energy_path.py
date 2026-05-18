"""Plan an AirSim path by minimizing expected SUV energy over a NEMo surface."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from nemo import Nemo
from nemo.energy_path_planning import EnergyPathPlanningConfig, EnergyModelConfig, plan_energy_path
from scripts.estimate_airsim_path_energy import estimate_energy, trajectory_from_airsim_path


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
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            host,
            "--directory",
            str(directory),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process, f"http://{host}:{port}"


def _plot_z(path_airsim: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    return path_airsim[:, 2] + float(z_offset)


def _sample_airsim_grid(nemo: Nemo, res: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def build_figure(
    nemo: Nemo,
    astar_airsim: np.ndarray,
    initial_airsim: np.ndarray,
    optimized_airsim: np.ndarray,
) -> go.Figure:
    as_x_axis, as_y_axis, as_z, slope_deg = _sample_airsim_grid(nemo)
    zmin = float(np.nanpercentile(as_z, 1.0))
    zmax = float(np.nanpercentile(as_z, 99.0))
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "surface"}], [{"type": "xy"}]],
        subplot_titles=("Energy-Optimized Path Over NEMo", "Slope Map"),
        row_heights=[0.62, 0.38],
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Surface(
            x=as_x_axis,
            y=as_y_axis,
            z=as_z,
            surfacecolor=as_z,
            colorscale="Viridis",
            cmin=zmin,
            cmax=zmax,
            showscale=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            colorbar=dict(title="Elevation<br>up (m)", x=1.02, y=0.72, len=0.42, thickness=14),
            name="NEMo surface",
            hovertemplate="X=%{x:.1f}<br>Y=%{y:.1f}<br>Elevation up=%{z:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=astar_airsim[:, 0],
            y=astar_airsim[:, 1],
            z=_plot_z(astar_airsim, PATH_PLOT_Z_OFFSET_M),
            mode="lines+markers",
            line=dict(color="#ff0000", width=6, dash="dash"),
            marker=dict(color="#ff0000", size=4),
            name="Initial A*",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=initial_airsim[:, 0],
            y=initial_airsim[:, 1],
            z=_plot_z(initial_airsim, PATH_PLOT_Z_OFFSET_M),
            mode="lines",
            line=dict(color="#38bdf8", width=5),
            name="Resampled A*",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=optimized_airsim[:, 0],
            y=optimized_airsim[:, 1],
            z=_plot_z(optimized_airsim, PATH_PLOT_Z_OFFSET_M),
            mode="lines+markers",
            line=dict(color="#f97316", width=8),
            marker=dict(color="#f97316", size=4),
            name="Energy optimized",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=as_x_axis,
            y=as_y_axis,
            z=slope_deg,
            colorscale="Viridis",
            zmin=0,
            zmax=25,
            colorbar=dict(title="Slope<br>(deg)", x=1.02, y=0.19, len=0.28, thickness=14),
            name="Slope",
        ),
        row=2,
        col=1,
    )
    for path, name, color, width, dash in (
        (astar_airsim, "Initial A*", "#ff0000", 3, "dash"),
        (initial_airsim, "Resampled A*", "#38bdf8", 3, None),
        (optimized_airsim, "Energy optimized", "#f97316", 4, None),
    ):
        line = dict(color=color, width=width)
        if dash is not None:
            line["dash"] = dash
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers" if name != "Resampled A*" else "lines",
                line=line,
                marker=dict(color=color, size=5),
                name=name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=1300,
        width=1250,
        title_text="NEMo Minimum-Energy Path Planning (AirSim Aligned)",
        template="plotly_dark",
        margin=dict(l=45, r=120, t=95, b=45),
        legend=dict(orientation="h", x=0.5, y=1.04, xanchor="center", yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        scene=dict(
            aspectmode="data",
            xaxis_title="AirSim X (Right)",
            yaxis_title="AirSim Y (Down)",
            zaxis_title="Elevation up (m)",
            yaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.3, y=-1.5, z=0.9)),
        ),
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/path_planning_energy"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--num-waypoints", type=int, default=80)
    parser.add_argument("--speed", type=float, default=6.0)
    args = parser.parse_args()

    nemo = Nemo.load_checkpoint(args.checkpoint, map_location=args.device).to(args.device)
    start_ds = airsim_to_dataset(AIRSIM_START)
    goal_ds = airsim_to_dataset(AIRSIM_GOAL)
    config = EnergyPathPlanningConfig(
        num_waypoints=int(args.num_waypoints),
        optimize_iterations=int(args.iterations),
        optimize_lr=float(args.lr),
        energy=EnergyModelConfig(target_speed_mps=float(args.speed)),
    )
    result = plan_energy_path(
        nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
        config=config,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    astar_airsim = dataset_to_airsim_path(result.astar_path_xyz)
    initial_airsim = dataset_to_airsim_path(result.initial_path_xyz)
    optimized_airsim = dataset_to_airsim_path(result.optimized_path_xyz)
    np.save(output_dir / "airsim_energy_astar_path.npy", astar_airsim)
    np.save(output_dir / "airsim_energy_initial_path.npy", initial_airsim)
    np.save(output_dir / "airsim_energy_path.npy", optimized_airsim)

    metrics = {
        "astar": estimate_energy(trajectory_from_airsim_path(astar_airsim, nominal_speed=float(args.speed))),
        "initial": estimate_energy(trajectory_from_airsim_path(initial_airsim, nominal_speed=float(args.speed))),
        "optimized": estimate_energy(trajectory_from_airsim_path(optimized_airsim, nominal_speed=float(args.speed))),
        "optimization": {
            "initial_objective": result.initial_objective,
            "optimized_objective": result.optimized_objective,
            "initial_energy_j": result.initial_energy_j,
            "optimized_energy_j": result.optimized_energy_j,
            "cost_history": result.cost_history,
        },
    }
    (output_dir / "energy_path_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig = build_figure(nemo, astar_airsim, initial_airsim, optimized_airsim)
    html_path = output_dir / "airsim_energy_path.html"
    fig.write_html(html_path)

    print(f"Initial energy objective energy: {result.initial_energy_j / 1000.0:.2f} kJ")
    print(f"Optimized energy objective energy: {result.optimized_energy_j / 1000.0:.2f} kJ")
    print(f"Offline AirSim initial energy: {metrics['initial']['energy_kj']:.2f} kJ")
    print(f"Offline AirSim optimized energy: {metrics['optimized']['energy_kj']:.2f} kJ")
    print(f"Saved outputs to {output_dir}")
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    opened = webbrowser.open(plot_url, new=2)
    print(f"Auto-open requested: {'opened' if opened else 'no browser reported success'}")


if __name__ == "__main__":
    main()
