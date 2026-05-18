from __future__ import annotations

import json
import socket
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import tyro

from nemo import Nemo
from nemo.path_planning import (
    PathPlanningConfig,
    build_traversability_grid,
    compute_path_metrics,
    config_payload,
    plan_path,
    sample_height_and_gradient_grid,
)


@dataclass
class PathPlanningArgs:
    checkpoint_path: str
    output_dir: str | None = None
    output_root: str = "output/path_planning"
    output_html: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    start_xy: tuple[float, float] | None = None
    goal_xy: tuple[float, float] | None = None
    astar_grid_resolution_x: int = 128
    astar_grid_resolution_y: int = 128
    buffer_fraction: float = 0.2
    num_waypoints: int = 48
    optimize_iterations: int = 250
    optimize_lr: float = 2e-2
    terrain_height_weight: float = 0.0
    terrain_slope_weight: float = 0.5
    flatness_weight: float = 1.0
    smoothness_weight: float = 0.15
    length_weight: float = 0.05
    astar_height_weight: float = 0.0
    astar_slope_weight: float = 2.0
    astar_max_slope_deg: float = 20.0
    astar_slope_reference_deg: float = 10.0
    astar_slope_exponent: float = 2.0
    astar_step_weight: float = 1.0
    gravity: float = 9.81
    dt: float = 1.0
    batch_size: int = 65536
    surface_resolution_x: int = 256
    surface_resolution_y: int = 256
    surface_opacity: float = 0.92
    preview_server: bool = True
    preview_host: str = "127.0.0.1"
    preview_port: int = 0
    open_plot: bool = True


def _default_output_dir(args: PathPlanningArgs, checkpoint_path: Path) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path(args.output_root) / checkpoint_path.stem


def _resolve_output_html(args: PathPlanningArgs, run_dir: Path) -> Path:
    if args.output_html is None:
        return run_dir / "path_planning.html"
    output_html = Path(args.output_html)
    if not output_html.is_absolute():
        output_html = run_dir / output_html
    return output_html


def _path_to_trace(
    path_xyz: np.ndarray,
    *,
    name: str,
    color: str,
    width: int,
    show_markers: bool = False,
) -> go.Scatter3d:
    return go.Scatter3d(
        x=path_xyz[:, 0],
        y=path_xyz[:, 1],
        z=path_xyz[:, 2] + 1e-3,
        mode="lines+markers" if show_markers else "lines",
        line=dict(color=color, width=width),
        marker=dict(size=3),
        name=name,
        hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        showlegend=True,
    )


def _path_to_trace_2d(
    path_xy: np.ndarray,
    *,
    name: str,
    color: str,
    width: int,
    show_markers: bool = False,
) -> go.Scatter:
    return go.Scatter(
        x=path_xy[:, 0],
        y=path_xy[:, 1],
        mode="lines+markers" if show_markers else "lines",
        line=dict(color=color, width=width),
        marker=dict(size=5),
        name=name,
        hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
        showlegend=False,
    )


def _build_figure(nemo: Nemo, result: object, args: PathPlanningArgs) -> go.Figure:
    surface_x, surface_y, surface_z, surface_gx, surface_gy = sample_height_and_gradient_grid(
        nemo,
        resolution_x=int(args.surface_resolution_x),
        resolution_y=int(args.surface_resolution_y),
        batch_size=int(args.batch_size),
    )
    traversability = build_traversability_grid(
        surface_z,
        surface_gx,
        surface_gy,
        height_weight=float(args.astar_height_weight),
        slope_weight=float(args.astar_slope_weight),
        max_slope_deg=float(args.astar_max_slope_deg),
        slope_reference_deg=float(args.astar_slope_reference_deg),
        slope_exponent=float(args.astar_slope_exponent),
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "surface"}]],
        column_widths=[0.42, 0.58],
        subplot_titles=("Traversability cost", "NEMo path planning"),
    )
    fig.add_trace(
        go.Heatmap(
            x=surface_x[0, :],
            y=surface_y[:, 0],
            z=traversability.heatmap,
            colorscale="Viridis",
            colorbar=dict(title="cost"),
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>cost=%{z:.4f}<extra></extra>",
            name="traversability",
            showscale=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _path_to_trace_2d(
            result.astar_path_xy if hasattr(result, "astar_path_xy") else result["astar_path_xy"],
            name="A* path",
            color="#c026d3",
            width=2,
            show_markers=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _path_to_trace_2d(
            result.optimized_path_xy if hasattr(result, "optimized_path_xy") else result["optimized_path_xy"],
            name="Optimized path",
            color="#f59e0b",
            width=4,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=surface_z,
            colorscale="Viridis",
            opacity=float(args.surface_opacity),
            showscale=True,
            name="terrain",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        _path_to_trace(
            _path_xy_to_xyz(
                nemo,
                result.astar_path_xy if hasattr(result, "astar_path_xy") else result["astar_path_xy"],
            ),
            name="A* path",
            color="#c026d3",
            width=5,
            show_markers=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        _path_to_trace(
            result.initial_path_xyz if hasattr(result, "initial_path_xyz") else result["initial_path_xyz"],
            name="Resampled A*",
            color="#dc2626",
            width=7,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        _path_to_trace(
            result.optimized_path_xyz if hasattr(result, "optimized_path_xyz") else result["optimized_path_xyz"],
            name="Optimized path",
            color="#f59e0b",
            width=10,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[result.start_xy[0] if hasattr(result, "start_xy") else result["start_xy"][0]],
            y=[result.start_xy[1] if hasattr(result, "start_xy") else result["start_xy"][1]],
            z=[float((result.initial_path_xyz if hasattr(result, "initial_path_xyz") else result["initial_path_xyz"])[0, 2])],
            mode="markers",
            marker=dict(size=6, color="#14532d", symbol="circle"),
            name="start",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[result.goal_xy[0] if hasattr(result, "goal_xy") else result["goal_xy"][0]],
            y=[result.goal_xy[1] if hasattr(result, "goal_xy") else result["goal_xy"][1]],
            z=[float((result.optimized_path_xyz if hasattr(result, "optimized_path_xyz") else result["optimized_path_xyz"])[-1, 2])],
            mode="markers",
            marker=dict(size=6, color="#7f1d1d", symbol="diamond"),
            name="goal",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="x", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_layout(
        title="NEMo Path Planning",
        template="plotly_white",
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(
            aspectmode="data",
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    return fig


def _path_xy_to_xyz(nemo: Nemo, path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    xy = torch.as_tensor(path_xy, dtype=torch.float32, device=nemo.device)
    z = nemo.h(xy).detach().cpu().numpy()
    return np.column_stack([path_xy[:, 0], path_xy[:, 1], z.squeeze(-1)]).astype(np.float32)


def _xyz_path_length(path_xyz: np.ndarray) -> float:
    path_xyz = np.asarray(path_xyz, dtype=np.float32)
    if path_xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path_xyz, axis=0), axis=1).sum())


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _start_preview_server(
    directory: Path,
    *,
    host: str,
    port: int,
) -> tuple[subprocess.Popen[bytes], str]:
    port = int(port) if int(port) > 0 else _find_free_port(host)
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


def _path_metrics(nemo: Nemo, path_xy: np.ndarray, cfg: PathPlanningConfig) -> dict[str, float]:
    metrics = compute_path_metrics(
        nemo,
        path_xy,
        terrain_height_weight=cfg.terrain_height_weight,
        terrain_slope_weight=cfg.terrain_slope_weight,
        flatness_weight=cfg.flatness_weight,
        smoothness_weight=cfg.smoothness_weight,
        length_weight=cfg.length_weight,
        gravity=cfg.gravity,
        dt=cfg.dt,
    )
    path_xyz = _path_xy_to_xyz(nemo, path_xy)
    metrics["path_length_3d"] = _xyz_path_length(path_xyz)
    metrics["num_waypoints"] = int(path_xy.shape[0])
    return metrics


def main(args: PathPlanningArgs) -> None:
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    nemo = Nemo.load_checkpoint(checkpoint_path, map_location=args.device).to(args.device)
    cfg = PathPlanningConfig(
        astar_grid_resolution_x=int(args.astar_grid_resolution_x),
        astar_grid_resolution_y=int(args.astar_grid_resolution_y),
        buffer_fraction=float(args.buffer_fraction),
        num_waypoints=int(args.num_waypoints),
        optimize_iterations=int(args.optimize_iterations),
        optimize_lr=float(args.optimize_lr),
        astar_height_weight=float(args.astar_height_weight),
        astar_slope_weight=float(args.astar_slope_weight),
        astar_max_slope_deg=float(args.astar_max_slope_deg),
        astar_slope_reference_deg=float(args.astar_slope_reference_deg),
        astar_slope_exponent=float(args.astar_slope_exponent),
        astar_step_weight=float(args.astar_step_weight),
        terrain_height_weight=float(args.terrain_height_weight),
        terrain_slope_weight=float(args.terrain_slope_weight),
        flatness_weight=float(args.flatness_weight),
        smoothness_weight=float(args.smoothness_weight),
        length_weight=float(args.length_weight),
        gravity=float(args.gravity),
        dt=float(args.dt),
        batch_size=int(args.batch_size),
    )

    result = plan_path(
        nemo,
        start_xy=args.start_xy,
        goal_xy=args.goal_xy,
        config=cfg,
    )

    run_dir = _default_output_dir(args, checkpoint_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_html = _resolve_output_html(args, run_dir)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    figure = _build_figure(nemo, result, args)
    figure.write_html(str(output_html))

    np.save(run_dir / "initial_path_xy.npy", result.initial_path_xy)
    np.save(run_dir / "optimized_path_xy.npy", result.optimized_path_xy)
    np.save(run_dir / "initial_path_xyz.npy", result.initial_path_xyz)
    np.save(run_dir / "optimized_path_xyz.npy", result.optimized_path_xyz)
    np.save(run_dir / "astar_path_xy.npy", result.astar_path_xy)
    _, _, astar_z, astar_gx, astar_gy = sample_height_and_gradient_grid(
        nemo,
        resolution_x=cfg.astar_grid_resolution_x,
        resolution_y=cfg.astar_grid_resolution_y,
        batch_size=cfg.batch_size,
    )
    traversability_grid = build_traversability_grid(
        astar_z,
        astar_gx,
        astar_gy,
        height_weight=cfg.astar_height_weight,
        slope_weight=cfg.astar_slope_weight,
        max_slope_deg=cfg.astar_max_slope_deg,
        slope_reference_deg=cfg.astar_slope_reference_deg,
        slope_exponent=cfg.astar_slope_exponent,
    )
    np.save(run_dir / "astar_slope_deg.npy", traversability_grid.slope_deg)
    np.save(run_dir / "astar_traversable.npy", traversability_grid.traversable)
    np.save(run_dir / "astar_traversability_cost.npy", traversability_grid.cost)
    np.save(run_dir / "astar_traversability_heatmap.npy", traversability_grid.heatmap)

    path_metrics = {
        "astar": _path_metrics(nemo, result.astar_path_xy, cfg),
        "resampled_astar": _path_metrics(nemo, result.initial_path_xy, cfg),
        "smoothed": _path_metrics(nemo, result.optimized_path_xy, cfg),
        "optimization": {
            "initial_objective": float(result.initial_objective),
            "final_objective": float(result.final_objective),
            "cost_history": [float(value) for value in result.cost_history],
        },
        "traversability": {
            "traversable_fraction": float(np.mean(traversability_grid.traversable)),
            "max_slope_deg": float(np.nanmax(traversability_grid.slope_deg)),
            "mean_finite_cost": float(np.mean(traversability_grid.cost[np.isfinite(traversability_grid.cost)])),
        },
    }

    results = {
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(run_dir.resolve()),
        "output_html": str(output_html.resolve()),
        "config": config_payload(cfg),
        "start_xy": list(result.start_xy),
        "goal_xy": list(result.goal_xy),
        "metrics": path_metrics,
        "cost_history": result.cost_history,
    }
    (run_dir / "path_metrics.json").write_text(json.dumps(path_metrics, indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Start: {result.start_xy}")
    print(f"Goal: {result.goal_xy}")
    print(f"Initial objective: {result.initial_objective:.6f}")
    print(f"Final objective: {result.final_objective:.6f}")
    print(f"Saved outputs to {run_dir}")
    print(f"Saved plot to {output_html}")
    print(f"Saved metrics to {run_dir / 'path_metrics.json'}")
    if bool(args.preview_server):
        server_process, base_url = _start_preview_server(
            output_html.parent.resolve(),
            host=str(args.preview_host),
            port=int(args.preview_port),
        )
        plot_url = f"{base_url}/{quote(output_html.name)}"
        print(f"Plot preview: {plot_url}")
        print(f"Preview server PID: {server_process.pid}")
        if bool(args.open_plot):
            opened = webbrowser.open(plot_url, new=2)
            print(f"Auto-open requested: {'opened' if opened else 'no browser reported success'}")


if __name__ == "__main__":
    main(tyro.cli(PathPlanningArgs))
