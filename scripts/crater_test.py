"""One-off crater planning test with no A* max-slope cutoff."""

from __future__ import annotations

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
from scripts.compare_physical_objectives import (
    AIRSIM_START,
    AIRSIM_SPIRAL_CENTER,
    AIRSIM_Z_OFFSET,
    dataset_to_airsim_path,
    sample_airsim_grid,
)


CRATER_CENTER_AIRSIM_XY = np.array([400.0, 300.0], dtype=np.float64)
CRATER_SEARCH_HALF_WIDTH_M = 125.0
CRATER_SEARCH_RESOLUTION = 181
PATH_PLOT_Z_OFFSET_M = 2.0


def airsim_to_dataset(p_airsim: np.ndarray) -> np.ndarray:
    p_dataset = p_airsim.copy()
    p_dataset[0] = p_airsim[1] - AIRSIM_SPIRAL_CENTER[1]
    p_dataset[1] = p_airsim[0] - AIRSIM_SPIRAL_CENTER[0]
    p_dataset[2] = p_airsim[2] + AIRSIM_Z_OFFSET
    return p_dataset


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


def find_local_low_airsim(
    nemo: Nemo,
    center_xy_airsim: np.ndarray,
    *,
    half_width_m: float = CRATER_SEARCH_HALF_WIDTH_M,
    resolution: int = CRATER_SEARCH_RESOLUTION,
) -> np.ndarray:
    """Find the NEMo local low point in an AirSim XY search window."""
    xs = np.linspace(float(center_xy_airsim[0]) - half_width_m, float(center_xy_airsim[0]) + half_width_m, resolution)
    ys = np.linspace(float(center_xy_airsim[1]) - half_width_m, float(center_xy_airsim[1]) + half_width_m, resolution)
    as_xx, as_yy = np.meshgrid(xs, ys, indexing="xy")
    ds_xx = as_yy - AIRSIM_SPIRAL_CENTER[1]
    ds_yy = as_xx - AIRSIM_SPIRAL_CENTER[0]
    points = np.stack([ds_xx.ravel(), ds_yy.ravel()], axis=1)
    device = next(nemo.field.parameters()).device
    heights = []
    with torch.no_grad():
        for start in range(0, len(points), 65536):
            xy = torch.as_tensor(points[start : start + 65536], device=device, dtype=torch.float32)
            heights.append(nemo.h(xy).detach().cpu().numpy().reshape(-1))
    z_dataset = np.concatenate(heights, axis=0)
    low_idx = int(np.nanargmin(z_dataset))
    return np.array(
        [
            float(as_xx.ravel()[low_idx]),
            float(as_yy.ravel()[low_idx]),
            float(z_dataset[low_idx] - AIRSIM_Z_OFFSET),
        ],
        dtype=np.float64,
    )


def build_figure(nemo: Nemo, astar: np.ndarray, optimized: np.ndarray, metrics: dict[str, float]) -> go.Figure:
    as_x, as_y, as_z, slope = sample_airsim_grid(nemo, res=240)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "table"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.74, 0.26],
        row_heights=[0.62, 0.38],
        subplot_titles=("Crater Low-Point Plan", "Diagnostics", "Slope Map", "XY Paths"),
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
    for path, name, color, width, dash in (
        (astar, "A* seed", "#ff0000", 5, "dash"),
        (optimized, "NEMo optimized", "#f97316", 8, None),
    ):
        line = dict(color=color, width=width)
        if dash is not None:
            line["dash"] = dash
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2] + PATH_PLOT_Z_OFFSET_M,
                mode="lines+markers" if name == "A* seed" else "lines",
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
    for path, name, color, dash in (
        (astar, "A* seed", "#ff0000", "dash"),
        (optimized, "NEMo optimized", "#f97316", None),
    ):
        line = dict(color=color, width=3)
        if dash is not None:
            line["dash"] = dash
        for row, col in ((2, 1), (2, 2)):
            fig.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode="lines+markers" if name == "A* seed" else "lines",
                    line=line,
                    marker=dict(color=color, size=4),
                    name=name,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    rows = [
        ("distance [m]", metrics["distance_3d_m"]),
        ("time [s]", metrics["time_s"]),
        ("energy [kJ]", metrics["energy_j"] / 1000.0),
        ("max slope [deg]", metrics["max_slope_deg"]),
        ("max roll [deg]", metrics["max_roll_deg"]),
        ("max curvature [1/m]", metrics["max_curvature_1pm"]),
        ("max yaw rate [rad/s]", metrics["max_yaw_rate_radps"]),
        ("max accel [m/s^2]", metrics["max_accel_mps2"]),
        ("max terrain bend [1/m]", metrics.get("max_forward_terrain_curvature_1pm", 0.0)),
        ("max breakover proxy [m]", metrics.get("max_breakover_proxy_m", 0.0)),
        ("max breakover viol [m]", metrics.get("max_breakover_violation_m", 0.0)),
        ("max footprint viol [m]", metrics.get("max_footprint_clearance_violation_m", 0.0)),
        ("mean footprint viol [m]", metrics.get("mean_footprint_clearance_violation_m", 0.0)),
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], align="left"),
            cells=dict(values=[[r[0] for r in rows], [f"{r[1]:.3f}" for r in rows]], align="left"),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=1300,
        width=1500,
        title_text="No-Max-Slope Crater Test",
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
    output_dir = Path("outputs/crater_test_400_300")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path("/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-16_threshold-vertical-fix_5000/nemo_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(checkpoint, map_location=device).to(device)
    crater_low_airsim = find_local_low_airsim(nemo, CRATER_CENTER_AIRSIM_XY)

    start_ds = airsim_to_dataset(AIRSIM_START)
    goal_ds = airsim_to_dataset(crater_low_airsim)
    vehicle = VehicleModelConfig(wheelbase_m=2.8, track_width_m=1.7, ground_clearance_m=0.35)
    safety = SafetyConstraintConfig(
        max_slope_deg=45.0,
        max_roll_deg=25.0,
        breakover_clearance_margin_m=0.005,
        breakover_weight=5e8,
        footprint_clearance_margin_m=0.35,
        footprint_clearance_weight=5e9,
    )
    config = PhysicalObjectivePlannerConfig(
        objective="energy",
        astar_max_slope_deg=89.0,
        astar_slope_weight=8.0,
        astar_grid_resolution_x=320,
        astar_grid_resolution_y=320,
        num_waypoints=180,
        optimize_iterations=1200,
        optimize_lr=1e-2,
        vehicle=vehicle,
        safety=safety,
    )
    result = plan_physical_objective_path(
        nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
        config=config,
    )
    astar = dataset_to_airsim_path(result.astar_path_xyz)
    control_path = dataset_to_airsim_path(result.optimized_path_xyz)
    reference = generate_airsim_reference_trajectory(control_path, vehicle=vehicle, sample_spacing_m=2.0)
    optimized = np.asarray(reference["positions"], dtype=np.float32)

    np.save(output_dir / "airsim_crater_astar_path.npy", astar)
    np.save(output_dir / "airsim_crater_control_path.npy", control_path)
    np.save(output_dir / "airsim_crater_path.npy", optimized)
    np.savez(output_dir / "airsim_crater_trajectory.npz", **reference)
    metrics = {
        "config": asdict(config),
        "start_airsim": AIRSIM_START.tolist(),
        "crater_center_airsim_xy": CRATER_CENTER_AIRSIM_XY.tolist(),
        "goal_airsim": crater_low_airsim.tolist(),
        "initial": asdict(result.initial_diagnostics),
        "optimized": asdict(result.optimized_diagnostics),
    }
    (output_dir / "crater_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig = build_figure(nemo, astar, optimized, metrics["optimized"])
    html_path = output_dir / "crater_test.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"

    print(f"Crater center AirSim XY: {CRATER_CENTER_AIRSIM_XY.tolist()}")
    print(f"Goal local low AirSim: {crater_low_airsim.tolist()}")
    print(f"A* max slope cutoff: disabled for this test ({config.astar_max_slope_deg} deg)")
    print(
        "optimized: "
        f"distance={result.optimized_diagnostics.distance_3d_m:.1f} m, "
        f"time={result.optimized_diagnostics.time_s:.1f} s, "
        f"energy={result.optimized_diagnostics.energy_j / 1000.0:.1f} kJ, "
        f"max_slope={result.optimized_diagnostics.max_slope_deg:.1f} deg, "
        f"max_roll={result.optimized_diagnostics.max_roll_deg:.1f} deg, "
        f"max_breakover={result.optimized_diagnostics.max_breakover_proxy_m:.3f} m, "
        f"max_footprint={result.optimized_diagnostics.max_footprint_clearance_violation_m:.3f} m"
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
