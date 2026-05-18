"""Sweep crater-entry optimizer settings and compare deviation from A*."""

from __future__ import annotations

import csv
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
    SafetyConstraintConfig,
    VehicleModelConfig,
    plan_physical_objective_path,
)
from scripts.compare_physical_objectives import AIRSIM_START, dataset_to_airsim_path, sample_airsim_grid
from scripts.crater_test import (
    CRATER_CENTER_AIRSIM_XY,
    airsim_to_dataset,
    find_local_low_airsim,
    start_preview_server,
)


CHECKPOINT = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
)


def sweep_configs() -> list[dict[str, float | int | str]]:
    return [
        {
            "name": "baseline_breakover",
            "lr": 1e-2,
            "iterations": 1200,
            "anchor_weight": 1e-3,
            "smoothness_weight": 1e3,
            "curvature_weight": 5e10,
            "breakover_weight": 5e8,
            "breakover_margin": 0.005,
            "num_waypoints": 180,
        },
        {
            "name": "more_iters",
            "lr": 1e-2,
            "iterations": 2500,
            "anchor_weight": 1e-3,
            "smoothness_weight": 1e3,
            "curvature_weight": 5e10,
            "breakover_weight": 5e8,
            "breakover_margin": 0.005,
            "num_waypoints": 180,
        },
        {
            "name": "higher_lr",
            "lr": 3e-2,
            "iterations": 1200,
            "anchor_weight": 1e-3,
            "smoothness_weight": 1e3,
            "curvature_weight": 5e10,
            "breakover_weight": 5e8,
            "breakover_margin": 0.005,
            "num_waypoints": 180,
        },
        {
            "name": "low_anchor",
            "lr": 1e-2,
            "iterations": 1600,
            "anchor_weight": 0.0,
            "smoothness_weight": 1e3,
            "curvature_weight": 5e10,
            "breakover_weight": 5e8,
            "breakover_margin": 0.005,
            "num_waypoints": 180,
        },
        {
            "name": "low_smooth",
            "lr": 1e-2,
            "iterations": 1600,
            "anchor_weight": 0.0,
            "smoothness_weight": 1e2,
            "curvature_weight": 1e10,
            "breakover_weight": 5e8,
            "breakover_margin": 0.005,
            "num_waypoints": 180,
        },
        {
            "name": "strong_breakover",
            "lr": 1e-2,
            "iterations": 1800,
            "anchor_weight": 0.0,
            "smoothness_weight": 1e2,
            "curvature_weight": 1e10,
            "breakover_weight": 5e9,
            "breakover_margin": 0.005,
            "num_waypoints": 180,
        },
        {
            "name": "very_strong_breakover",
            "lr": 1e-2,
            "iterations": 2000,
            "anchor_weight": 0.0,
            "smoothness_weight": 1e2,
            "curvature_weight": 1e10,
            "breakover_weight": 5e10,
            "breakover_margin": 0.005,
            "num_waypoints": 180,
        },
        {
            "name": "more_control_points",
            "lr": 8e-3,
            "iterations": 1800,
            "anchor_weight": 0.0,
            "smoothness_weight": 1e2,
            "curvature_weight": 1e10,
            "breakover_weight": 5e9,
            "breakover_margin": 0.005,
            "num_waypoints": 260,
        },
    ]


def nearest_path_distance(points_xy: np.ndarray, reference_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    reference = np.asarray(reference_xy, dtype=np.float64)
    dists = np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=-1)
    return np.min(dists, axis=1)


def deviation_metrics(path_xy: np.ndarray, reference_xy: np.ndarray) -> dict[str, float]:
    d = nearest_path_distance(path_xy, reference_xy)
    d_ref = nearest_path_distance(reference_xy, path_xy)
    return {
        "mean_dev_m": float(np.mean(d)),
        "max_dev_m": float(np.max(d)),
        "hausdorff_dev_m": float(max(np.max(d), np.max(d_ref))),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_figure(nemo: Nemo, astar: np.ndarray, candidates: dict[str, np.ndarray], rows: list[dict[str, object]]) -> go.Figure:
    as_x, as_y, as_z, slope = sample_airsim_grid(nemo, res=240)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "table"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.72, 0.28],
        row_heights=[0.62, 0.38],
        subplot_titles=("Crater Optimizer Sweep", "Top Runs", "Slope Map", "XY Paths"),
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
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
            colorbar=dict(title="Z<br>(m)", x=0.72, y=0.73, len=0.35, thickness=12),
            name="NEMo surface",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=astar[:, 0],
            y=astar[:, 1],
            z=astar[:, 2] + 8.0,
            mode="lines+markers",
            line=dict(color="#ff0000", width=5, dash="dash"),
            marker=dict(color="#ff0000", size=3),
            name="A* seed",
        ),
        row=1,
        col=1,
    )
    palette = ["#f97316", "#22c55e", "#38bdf8", "#e879f9", "#facc15"]
    for idx, (name, path) in enumerate(candidates.items()):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2] + 8.0,
                mode="lines",
                line=dict(color=color, width=7),
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
            colorbar=dict(title="Slope<br>(deg)", x=0.72, y=0.19, len=0.25, thickness=12),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for row, col in ((2, 1), (2, 2)):
        fig.add_trace(
            go.Scatter(
                x=astar[:, 0],
                y=astar[:, 1],
                mode="lines+markers",
                line=dict(color="#ff0000", width=3, dash="dash"),
                marker=dict(color="#ff0000", size=4),
                name="A* seed",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        for idx, (name, path) in enumerate(candidates.items()):
            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=name,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    top_rows = rows[:5]
    fig.add_trace(
        go.Table(
            header=dict(values=["name", "energy kJ", "mean dev", "max breakover"], align="left"),
            cells=dict(
                values=[
                    [r["name"] for r in top_rows],
                    [f"{float(r['energy_kj']):.1f}" for r in top_rows],
                    [f"{float(r['mean_dev_m']):.1f}" for r in top_rows],
                    [f"{float(r['max_breakover_proxy_m']):.3f}" for r in top_rows],
                ],
                align="left",
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=1300,
        width=1550,
        title_text="Crater Entry Optimizer Sweep",
        template="plotly_dark",
        margin=dict(l=45, r=80, t=95, b=45),
        legend=dict(orientation="h", x=0.5, y=1.04, xanchor="center", yanchor="bottom"),
        scene=dict(
            aspectmode="data",
            xaxis_title="AirSim X",
            yaxis_title="AirSim Y",
            zaxis_title="Z",
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
    output_dir = Path("outputs/crater_optimizer_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(CHECKPOINT, map_location=device).to(device)
    goal_airsim = find_local_low_airsim(nemo, CRATER_CENTER_AIRSIM_XY)
    start_ds = airsim_to_dataset(AIRSIM_START)
    goal_ds = airsim_to_dataset(goal_airsim)

    rows: list[dict[str, object]] = []
    paths: dict[str, np.ndarray] = {}
    astar_airsim: np.ndarray | None = None
    for spec in sweep_configs():
        name = str(spec["name"])
        vehicle = VehicleModelConfig(wheelbase_m=2.8)
        safety = SafetyConstraintConfig(
            max_slope_deg=45.0,
            max_roll_deg=25.0,
            anchor_weight=float(spec["anchor_weight"]),
            smoothness_weight=float(spec["smoothness_weight"]),
            curvature_weight=float(spec["curvature_weight"]),
            breakover_clearance_margin_m=float(spec["breakover_margin"]),
            breakover_weight=float(spec["breakover_weight"]),
        )
        cfg = PhysicalObjectivePlannerConfig(
            objective="energy",
            astar_max_slope_deg=89.0,
            astar_slope_weight=8.0,
            astar_grid_resolution_x=320,
            astar_grid_resolution_y=320,
            num_waypoints=int(spec["num_waypoints"]),
            optimize_iterations=int(spec["iterations"]),
            optimize_lr=float(spec["lr"]),
            vehicle=vehicle,
            safety=safety,
        )
        result = plan_physical_objective_path(
            nemo,
            start_xy=(float(start_ds[0]), float(start_ds[1])),
            goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
            config=cfg,
        )
        astar_airsim = dataset_to_airsim_path(result.astar_path_xyz)
        control_path = dataset_to_airsim_path(result.optimized_path_xyz)
        reference = generate_airsim_reference_trajectory(control_path, vehicle=vehicle, sample_spacing_m=2.0)
        dense_path = np.asarray(reference["positions"], dtype=np.float32)
        paths[name] = dense_path
        np.save(output_dir / f"{name}_path.npy", dense_path)
        np.save(output_dir / f"{name}_control_path.npy", control_path)
        np.savez(output_dir / f"{name}_trajectory.npz", **reference)
        dev = deviation_metrics(result.optimized_path_xy, result.initial_path_xy)
        diag = asdict(result.optimized_diagnostics)
        row = {
            "name": name,
            "lr": spec["lr"],
            "iterations": spec["iterations"],
            "anchor_weight": spec["anchor_weight"],
            "smoothness_weight": spec["smoothness_weight"],
            "curvature_weight": spec["curvature_weight"],
            "breakover_weight": spec["breakover_weight"],
            "breakover_margin": spec["breakover_margin"],
            "num_waypoints": spec["num_waypoints"],
            "distance_3d_m": diag["distance_3d_m"],
            "time_s": diag["time_s"],
            "energy_kj": diag["energy_j"] / 1000.0,
            "max_slope_deg": diag["max_slope_deg"],
            "max_roll_deg": diag["max_roll_deg"],
            "max_curvature_1pm": diag["max_curvature_1pm"],
            "max_breakover_proxy_m": diag["max_breakover_proxy_m"],
            "max_breakover_violation_m": diag["max_breakover_violation_m"],
            **dev,
        }
        rows.append(row)
        print(
            f"{name}: energy={row['energy_kj']:.1f} kJ, mean_dev={row['mean_dev_m']:.1f} m, "
            f"max_dev={row['max_dev_m']:.1f} m, breakover={row['max_breakover_proxy_m']:.3f} m"
        )

    rows_by_dev = sorted(rows, key=lambda r: float(r["mean_dev_m"]), reverse=True)
    rows_by_energy = sorted(rows, key=lambda r: float(r["energy_kj"]))
    write_csv(output_dir / "sweep_results_by_deviation.csv", rows_by_dev)
    write_csv(output_dir / "sweep_results_by_energy.csv", rows_by_energy)
    (output_dir / "sweep_results.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    if astar_airsim is None:
        raise RuntimeError("Sweep did not produce any paths.")
    selected = {str(row["name"]): paths[str(row["name"])] for row in rows_by_dev[:5]}
    np.save(output_dir / "astar_seed.npy", astar_airsim)
    fig = build_figure(nemo, astar_airsim, selected, rows_by_dev)
    html_path = output_dir / "crater_optimizer_sweep.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"
    print(f"Saved sweep to {output_dir}")
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
