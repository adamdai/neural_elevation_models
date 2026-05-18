"""Generate a path from start to goal in AirSim coordinates with harsh slope penalty."""

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

from nemo import Nemo
from nemo.path_planning import (
    PathPlanningConfig,
    plan_path,
)
from nemo.path_planning import path_xy_to_xyz

# AirSim Constants
AIRSIM_START = np.array([0.0, 0.0, 0.0])
AIRSIM_GOAL = np.array([1050.50, 323.84, -19.73])
AIRSIM_SPIRAL_CENTER = np.array([524.38, 168.34, 0.0])

def airsim_to_dataset(p_airsim: np.ndarray) -> np.ndarray:
    """Transform AirSim (x, y, z) to Dataset (x, y, z).
    X_as = Y_ds + 524.38 => Y_ds = X_as - 524.38
    Y_as = X_ds + 168.34 => X_ds = Y_as - 168.34
    """
    p_dataset = p_airsim.copy()
    p_dataset[0] = p_airsim[1] - AIRSIM_SPIRAL_CENTER[1]
    p_dataset[1] = p_airsim[0] - AIRSIM_SPIRAL_CENTER[0]
    return p_dataset

def dataset_to_airsim(p_dataset: np.ndarray) -> np.ndarray:
    """Transform Dataset (x, y, z) to AirSim (x, y, z)."""
    p_airsim = p_dataset.copy()
    p_airsim[0] = p_dataset[1] + AIRSIM_SPIRAL_CENTER[0]
    p_airsim[1] = p_dataset[0] + AIRSIM_SPIRAL_CENTER[1]
    # Apply bias correction: AirSim_Z = Dataset_Z - Bias
    p_airsim[2] = p_dataset[2] - 24.15
    return p_airsim

def airsim_path_to_plot_xyz(path_airsim: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    """Convert AirSim XYZ into plotting XYZ.

    This is only a Plotly display transform. The saved AirSim path keeps the
    original AirSim convention. Geometry and color stay in the same display
    frame; Plotly lighting is flattened separately to avoid shading artifacts.
    """
    path_plot = np.asarray(path_airsim, dtype=np.float64).copy()
    path_plot[:, 2] = path_plot[:, 2] + float(z_offset)
    return path_plot

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

def smooth_polyline_naive(path_xy: np.ndarray, iterations: int = 20, alpha: float = 0.45) -> np.ndarray:
    """Laplacian smooth a polyline while keeping endpoints fixed."""
    path = np.asarray(path_xy, dtype=np.float64).copy()
    if path.shape[0] <= 2:
        return path.astype(np.float32)
    alpha = float(np.clip(alpha, 0.0, 0.5))
    for _ in range(int(iterations)):
        prev_path = path.copy()
        path[1:-1] = (1.0 - 2.0 * alpha) * prev_path[1:-1] + alpha * (prev_path[:-2] + prev_path[2:])
    return path.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/path_planning/airsim_path_safe.npy"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. Load NEMo
    print(f"Loading NEMo checkpoint: {args.checkpoint}")
    nemo = Nemo.load_checkpoint(args.checkpoint, map_location=args.device).to(args.device)

    # 2. Plan in Dataset Coordinates
    start_ds = airsim_to_dataset(AIRSIM_START)
    goal_ds = airsim_to_dataset(AIRSIM_GOAL)

    cfg = PathPlanningConfig(
        astar_grid_resolution_x=256,
        astar_grid_resolution_y=256,
        buffer_fraction=0.1,
        num_waypoints=80,
        optimize_iterations=400,
        astar_slope_weight=20.0,
        astar_slope_reference_deg=10.0,
        astar_slope_exponent=3.0,
        astar_max_slope_deg=25.0,
        terrain_slope_weight=10.0,
        smoothness_weight=1.0,
        length_weight=0.1,
    )

    print("Planning path with harsh slope constraints...")
    result = plan_path(
        nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
        config=cfg,
    )

    # 3. Transform result back to AirSim
    path_ds = result.optimized_path_xyz 
    path_airsim = np.array([dataset_to_airsim(p) for p in path_ds])
    
    # 3a. Transform A* and a naive smoothed baseline for comparison/export.
    astar_ds = path_xy_to_xyz(nemo, result.astar_path_xy)
    path_astar_airsim = np.array([dataset_to_airsim(p) for p in astar_ds])
    smooth_path_xy = smooth_polyline_naive(result.astar_path_xy)
    smooth_ds = path_xy_to_xyz(nemo, smooth_path_xy)
    path_smooth_airsim = np.array([dataset_to_airsim(p) for p in smooth_ds])

    # Calculate slopes along optimized path
    xy_torch = torch.as_tensor(path_ds[:, :2], dtype=torch.float32, device=nemo.device)
    grads = nemo.grad(xy_torch).detach().cpu().numpy()
    slopes = np.degrees(np.arctan(np.linalg.norm(grads, axis=1)))
    
    print(f"\n--- Optimized Path Slope Metrics ---")
    print(f"Average Slope: {np.mean(slopes):.2f}°")
    print(f"Maximum Slope: {np.max(slopes):.2f}°")
    print(f"Percentage above 10°: {np.mean(slopes > 10.0)*100:.1f}%")
    print(f"---------------------------\n")

    # 4. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, path_airsim)
    astar_output = args.output.with_name("airsim_astar_path.npy")
    smooth_output = args.output.with_name("airsim_smooth_path.npy")
    np.save(astar_output, path_astar_airsim)
    np.save(smooth_output, path_smooth_airsim)
    print(f"Saved safe AirSim path to {args.output}")
    print(f"Saved A* AirSim path to {astar_output}")
    print(f"Saved naive smoothed AirSim path to {smooth_output}")

    # 5. Visualization
    print("Generating enhanced visualization...")
    
    # 5a. Create Grid directly in AirSim Space
    res = 256
    as_x_axis = np.linspace(AIRSIM_SPIRAL_CENTER[0] - 900, AIRSIM_SPIRAL_CENTER[0] + 900, res)
    as_y_axis = np.linspace(AIRSIM_SPIRAL_CENTER[1] - 900, AIRSIM_SPIRAL_CENTER[1] + 900, res)
    as_xx, as_yy = np.meshgrid(as_x_axis, as_y_axis, indexing="xy")
    
    # 5b. Map AS grid to DS for querying
    ds_xx = as_yy - AIRSIM_SPIRAL_CENTER[1]
    ds_yy = as_xx - AIRSIM_SPIRAL_CENTER[0]
    xy_ds = np.stack([ds_xx.flatten(), ds_yy.flatten()], axis=-1)
    
    # Query NEMo
    heights = []
    grads = []
    device = nemo.device
    for start in range(0, len(xy_ds), 65536):
        batch = torch.as_tensor(xy_ds[start : start + 65536], dtype=torch.float32, device=device)
        heights.append(nemo.h(batch).detach().cpu().numpy())
        grads.append(nemo.grad(batch).detach().cpu().numpy())

    as_z_grid = np.concatenate(heights, axis=0).reshape(res, res) - 24.15
    plot_z_grid = as_z_grid
    grad_ds = np.concatenate(grads, axis=0).reshape(res, res, 2)
    as_slope_grid = np.degrees(np.arctan(np.linalg.norm(grad_ds, axis=-1)))
    astar_plot = airsim_path_to_plot_xyz(path_astar_airsim, z_offset=8.0)
    smooth_plot = airsim_path_to_plot_xyz(path_smooth_airsim, z_offset=8.0)
    opt_plot = airsim_path_to_plot_xyz(path_airsim, z_offset=8.0)
    zmin = float(np.nanpercentile(plot_z_grid, 1.0))
    zmax = float(np.nanpercentile(plot_z_grid, 99.0))
    
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "surface"}], [{"type": "xy"}]],
        subplot_titles=("3D Aligned Terrain & Paths", "2D Slope Heatmap (Degrees)"),
        row_heights=[0.62, 0.38],
        vertical_spacing=0.08
    )

    # Row 1: 3D Surface
    # Geometry and color are AirSim-aligned; lighting is flattened to avoid
    # directional shading artifacts.
    fig.add_trace(go.Surface(
        x=as_x_axis,
        y=as_y_axis,
        z=plot_z_grid,
        surfacecolor=plot_z_grid,
        cmin=zmin,
        cmax=zmax,
        colorscale="Viridis", opacity=1.0, name="NEMo Surface",
        showscale=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        colorbar=dict(
            title="Elevation<br>up (m)",
            x=1.02,
            y=0.72,
            len=0.42,
            thickness=14,
        ),
        hovertemplate="X=%{x:.1f}<br>Y=%{y:.1f}<br>Elevation up=%{z:.2f} m<extra></extra>",
    ), row=1, col=1)

    # A* Path
    fig.add_trace(go.Scatter3d(
        x=astar_plot[:, 0], y=astar_plot[:, 1], z=astar_plot[:, 2],
        mode="lines+markers",
        line=dict(color="#ff0000", width=6, dash="dash"),
        marker=dict(color="#ff0000", size=4),
        name="Initial A* Path",
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=smooth_plot[:, 0], y=smooth_plot[:, 1], z=smooth_plot[:, 2],
        mode="lines",
        line=dict(color="#38bdf8", width=6),
        name="Naive Smoothed A*",
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter3d(
        x=opt_plot[:, 0], y=opt_plot[:, 1], z=opt_plot[:, 2],
        mode="lines", line=dict(color="#f97316", width=8), name="Optimized Path"
    ), row=1, col=1)

    # Row 2: 2D Slope Heatmap
    fig.add_trace(go.Heatmap(
        x=as_x_axis,
        y=as_y_axis,
        z=as_slope_grid,
        colorscale="Viridis",
        zmin=0, zmax=25,
        colorbar=dict(
            title="Slope<br>(deg)",
            x=1.02,
            y=0.19,
            len=0.28,
            thickness=14,
        ),
        name="Slope"
    ), row=2, col=1)

    # Overlay A* on heatmap
    fig.add_trace(go.Scatter(
        x=path_astar_airsim[:, 0], y=path_astar_airsim[:, 1],
        mode="lines+markers",
        line=dict(color="#ff0000", width=3, dash="dash"),
        marker=dict(color="#ff0000", size=5),
        name="A* Path (Initial)"
    ), row=2, col=1)

    # Overlay naive smoothed baseline on heatmap
    fig.add_trace(go.Scatter(
        x=path_smooth_airsim[:, 0], y=path_smooth_airsim[:, 1],
        mode="lines",
        line=dict(color="#38bdf8", width=3),
        name="Naive Smoothed A*"
    ), row=2, col=1)

    # Overlay Optimized on heatmap
    fig.add_trace(go.Scatter(
        x=path_airsim[:, 0], y=path_airsim[:, 1],
        mode="lines", line=dict(color="#f97316", width=4),
        name="Optimized Path"
    ), row=2, col=1)

    fig.update_layout(
        height=1300, width=1250,
        title_text="NEMo Safe Path Planning (AirSim Aligned)",
        template="plotly_dark",
        margin=dict(l=45, r=120, t=95, b=45),
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.04,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
        ),
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

    vis_html = args.output.with_suffix(".html")
    fig.write_html(vis_html)
    print(f"Saved plot to: {vis_html}")
    server_process, base_url = start_preview_server(vis_html.parent.resolve())
    plot_url = f"{base_url}/{quote(vis_html.name)}"
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    opened = webbrowser.open(plot_url, new=2)
    print(f"Auto-open requested: {'opened' if opened else 'no browser reported success'}")

if __name__ == "__main__":
    main()
