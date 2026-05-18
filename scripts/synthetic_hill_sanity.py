"""Synthetic hill sanity check for differentiable terrain planning."""

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

from nemo import HeightField, Nemo
from nemo.basic_terrain_planner import (
    BasicTerrainPlannerConfig,
    optimize_basic_terrain_path,
    path_xyz,
)


class GaussianHillField(HeightField):
    """Analytic hill used to verify that path optimization avoids steep terrain."""

    def __init__(
        self,
        *,
        height_m: float = 2.0,
        sigma_m: float = 0.33,
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((-2.0, 2.0), (-1.5, 1.5)),
    ) -> None:
        super().__init__(bounds)
        self.height_m = float(height_m)
        self.sigma_m = float(sigma_m)

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        r2 = xy[:, 0:1].pow(2) + xy[:, 1:2].pow(2)
        return self.height_m * torch.exp(-0.5 * r2 / (self.sigma_m**2))


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


def straight_seed_path(num_points: int = 70) -> np.ndarray:
    x = np.linspace(-1.65, 1.65, num_points, dtype=np.float32)
    y = np.zeros_like(x)
    # A perfectly centered path is left/right symmetric and can be stationary.
    # This tiny deterministic offset gives the optimizer a direction to escape.
    y += (0.06 * np.sin(np.linspace(0.0, np.pi, num_points))).astype(np.float32)
    path = np.column_stack([x, y]).astype(np.float32)
    path[0, 1] = 0.0
    path[-1, 1] = 0.0
    return path


def path_hill_metrics(nemo: Nemo, path_xy: np.ndarray) -> dict[str, float]:
    with torch.no_grad():
        xy = torch.as_tensor(path_xy, dtype=torch.float32, device=nemo.device)
        z = nemo.h(xy).detach().cpu().numpy().reshape(-1)
    radius = np.linalg.norm(path_xy, axis=1)
    return {
        "max_height_m": float(np.max(z)),
        "mean_height_m": float(np.mean(z)),
        "min_distance_to_hill_center_m": float(np.min(radius)),
        "max_abs_lateral_deviation_m": float(np.max(np.abs(path_xy[:, 1]))),
    }


def build_config() -> BasicTerrainPlannerConfig:
    return BasicTerrainPlannerConfig(
        num_control_points=8,
        num_samples=220,
        num_iters=1200,
        lr=2.0e-2,
        w_length=5.0,
        w_slope=1000.0,
        w_control=3.5,
        w_control_points=1.5,
    )


def build_figure(
    nemo: Nemo,
    initial_xy: np.ndarray,
    optimized_xy: np.ndarray,
    metrics: dict[str, object],
) -> go.Figure:
    x = np.linspace(nemo.field.bounds[0][0], nemo.field.bounds[0][1], 180)
    y = np.linspace(nemo.field.bounds[1][0], nemo.field.bounds[1][1], 150)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    with torch.no_grad():
        z = nemo.h(torch.as_tensor(xy, dtype=torch.float32, device=nemo.device)).cpu().numpy().reshape(xx.shape)
    initial_xyz = path_xyz(nemo, initial_xy)
    optimized_xyz = path_xyz(nemo, optimized_xy)
    z_lift = 0.08

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        column_widths=[0.65, 0.35],
        subplot_titles=("Synthetic Hill Planner Sanity Check", "Top-Down View"),
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=z,
            surfacecolor=z,
            colorscale="Viridis",
            cmin=float(np.min(z)),
            cmax=float(np.max(z)),
            lighting=dict(ambient=0.85, diffuse=0.25, specular=0.05, roughness=0.8),
            colorbar=dict(title="height<br>(m)", x=0.62, len=0.6, thickness=12),
            name="hill",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=initial_xyz[:, 0],
            y=initial_xyz[:, 1],
            z=initial_xyz[:, 2] + z_lift,
            mode="lines+markers",
            line=dict(color="#ff2d2d", width=6, dash="dash"),
            marker=dict(color="#ff2d2d", size=3),
            name="initial straight path",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=optimized_xyz[:, 0],
            y=optimized_xyz[:, 1],
            z=optimized_xyz[:, 2] + z_lift,
            mode="lines",
            line=dict(color="#22c55e", width=8),
            name="optimized path",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale="Viridis",
            showscale=False,
            name="height map",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=initial_xy[:, 0],
            y=initial_xy[:, 1],
            mode="lines+markers",
            line=dict(color="#ff2d2d", width=3, dash="dash"),
            marker=dict(color="#ff2d2d", size=4),
            name="initial straight path",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=optimized_xy[:, 0],
            y=optimized_xy[:, 1],
            mode="lines",
            line=dict(color="#22c55e", width=4),
            name="optimized path",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    initial = metrics["initial"]
    optimized = metrics["optimized"]
    annotation = (
        f"max hill height sampled: {initial['max_height_m']:.2f} -> {optimized['max_height_m']:.2f} m<br>"
        f"min center distance: {initial['min_distance_to_hill_center_m']:.2f} -> "
        f"{optimized['min_distance_to_hill_center_m']:.2f} m<br>"
        f"total cost: {metrics['initial_diagnostics']['total_cost']:.1f} -> "
        f"{metrics['optimized_diagnostics']['total_cost']:.1f}"
    )
    fig.add_annotation(
        text=annotation,
        xref="paper",
        yref="paper",
        x=0.66,
        y=0.02,
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0.62)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
    )
    fig.update_layout(
        template="plotly_dark",
        width=1500,
        height=850,
        margin=dict(l=30, r=45, t=70, b=35),
        legend=dict(orientation="h", x=0.32, y=1.04, xanchor="center", yanchor="bottom"),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.0, z=0.45),
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="height (m)",
            camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9)),
        ),
        xaxis=dict(title="x (m)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="y (m)"),
    )
    return fig


def main() -> None:
    output_dir = Path("outputs/synthetic_hill_sanity")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nemo = Nemo(GaussianHillField()).to(device)
    initial_xy = straight_seed_path()
    cfg = build_config()
    result = optimize_basic_terrain_path(nemo, initial_xy, cfg)
    optimized_xy = result.optimized_path_xy

    initial_metrics = path_hill_metrics(nemo, initial_xy)
    optimized_metrics = path_hill_metrics(nemo, optimized_xy)
    metrics: dict[str, object] = {
        "config": asdict(cfg),
        "initial": initial_metrics,
        "optimized": optimized_metrics,
        "initial_diagnostics": asdict(result.initial_diagnostics),
        "optimized_diagnostics": asdict(result.optimized_diagnostics),
        "cost_history": result.cost_history,
    }
    np.save(output_dir / "synthetic_hill_initial_path.npy", initial_xy)
    np.save(output_dir / "synthetic_hill_optimized_path.npy", optimized_xy)
    np.save(output_dir / "synthetic_hill_control_points_initial.npy", result.control_points_initial)
    np.save(output_dir / "synthetic_hill_control_points_optimized.npy", result.control_points_optimized)
    (output_dir / "synthetic_hill_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig = build_figure(nemo, initial_xy, optimized_xy, metrics)
    html_path = output_dir / "synthetic_hill_sanity.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"

    print(
        "initial: "
        f"cost={result.initial_diagnostics.total_cost:.2f}, "
        f"max_height={initial_metrics['max_height_m']:.2f} m, "
        f"min_center_distance={initial_metrics['min_distance_to_hill_center_m']:.2f} m, "
        f"max_lateral={initial_metrics['max_abs_lateral_deviation_m']:.2f} m"
    )
    print(
        "optimized: "
        f"cost={result.optimized_diagnostics.total_cost:.2f}, "
        f"max_height={optimized_metrics['max_height_m']:.2f} m, "
        f"min_center_distance={optimized_metrics['min_distance_to_hill_center_m']:.2f} m, "
        f"max_lateral={optimized_metrics['max_abs_lateral_deviation_m']:.2f} m, "
        f"control_effort={result.optimized_diagnostics.control_effort:.4f}"
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
