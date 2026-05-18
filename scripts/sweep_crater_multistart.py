"""Run crater-entry optimization from multiple biased initial paths."""

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
from nemo.path_planning import path_xy_to_xyz, resample_polyline
from nemo.physical_objective_planner import (
    PhysicalObjectivePlannerConfig,
    SafetyConstraintConfig,
    VehicleModelConfig,
    _build_astar_seed,
    optimize_physical_path,
)
from scripts.compare_physical_objectives import AIRSIM_START, dataset_to_airsim_path, sample_airsim_grid
from scripts.crater_test import (
    CRATER_CENTER_AIRSIM_XY,
    airsim_to_dataset,
    find_local_low_airsim,
    start_preview_server,
)
from scripts.sweep_crater_optimization import CHECKPOINT, deviation_metrics


def make_base_config() -> PhysicalObjectivePlannerConfig:
    vehicle = VehicleModelConfig(wheelbase_m=2.8)
    safety = SafetyConstraintConfig(
        max_slope_deg=45.0,
        max_roll_deg=25.0,
        anchor_weight=0.0,
        smoothness_weight=1e2,
        curvature_weight=1e10,
        breakover_clearance_margin_m=0.005,
        breakover_weight=5e9,
    )
    return PhysicalObjectivePlannerConfig(
        objective="energy",
        astar_max_slope_deg=89.0,
        astar_slope_weight=8.0,
        astar_grid_resolution_x=320,
        astar_grid_resolution_y=320,
        num_waypoints=180,
        optimize_iterations=1600,
        optimize_lr=1e-2,
        vehicle=vehicle,
        safety=safety,
    )


def path_tangent_normals(path_xy: np.ndarray) -> np.ndarray:
    d = np.gradient(path_xy.astype(np.float64), axis=0)
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    tangent = d / np.clip(norm, 1e-8, None)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def biased_seed(path_xy: np.ndarray, *, amplitude_m: float, mode: str, rng: np.random.Generator | None = None) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float64)
    n = path.shape[0]
    u = np.linspace(0.0, 1.0, n)
    normals = path_tangent_normals(path)

    if mode == "sin":
        profile = np.sin(np.pi * u)
        offset = amplitude_m * profile[:, None] * normals
    elif mode == "late":
        profile = np.exp(-0.5 * ((u - 0.72) / 0.18) ** 2)
        offset = amplitude_m * profile[:, None] * normals
    elif mode == "arc":
        start = path[0]
        goal = path[-1]
        chord = goal - start
        chord_len = max(float(np.linalg.norm(chord)), 1e-8)
        normal = np.array([-chord[1], chord[0]], dtype=np.float64) / chord_len
        profile = np.sin(np.pi * u)
        offset = amplitude_m * profile[:, None] * normal[None, :]
    elif mode == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        control_u = np.linspace(0.0, 1.0, 8)
        values = rng.normal(0.0, 1.0, size=8)
        values[0] = 0.0
        values[-1] = 0.0
        profile = np.interp(u, control_u, values)
        profile *= np.sin(np.pi * u)
        max_abs = max(float(np.max(np.abs(profile))), 1e-8)
        profile = profile / max_abs
        offset = amplitude_m * profile[:, None] * normals
    else:
        raise ValueError(f"Unknown seed mode: {mode}")

    out = path + offset
    out[0] = path[0]
    out[-1] = path[-1]
    return out.astype(np.float32)


def seed_specs() -> list[tuple[str, float, str, int | None]]:
    specs: list[tuple[str, float, str, int | None]] = [("astar", 0.0, "sin", None)]
    for amp in (-80.0, -50.0, -25.0, 25.0, 50.0, 80.0):
        specs.append((f"arc_{amp:+.0f}m", amp, "arc", None))
    for amp in (-60.0, -35.0, 35.0, 60.0):
        specs.append((f"late_{amp:+.0f}m", amp, "late", None))
    for seed, amp in enumerate((25.0, 40.0, 60.0), start=1):
        specs.append((f"random_{seed}_{amp:.0f}m", amp, "random", seed))
    return specs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_figure(
    nemo: Nemo,
    astar: np.ndarray,
    initial_seeds: dict[str, np.ndarray],
    optimized: dict[str, np.ndarray],
    rows: list[dict[str, object]],
) -> go.Figure:
    as_x, as_y, as_z, slope = sample_airsim_grid(nemo, res=240)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "table"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.72, 0.28],
        row_heights=[0.62, 0.38],
        subplot_titles=("Crater Multistart Optimization", "Top Runs", "Slope Map", "XY Paths"),
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
    palette = ["#f97316", "#22c55e", "#38bdf8", "#e879f9", "#facc15", "#a78bfa"]
    selected = [str(row["name"]) for row in rows[:6]]
    for idx, name in enumerate(selected):
        color = palette[idx % len(palette)]
        seed = initial_seeds[name]
        path = optimized[name]
        fig.add_trace(
            go.Scatter3d(
                x=seed[:, 0],
                y=seed[:, 1],
                z=seed[:, 2] + 5.0,
                mode="lines",
                line=dict(color=color, width=3, dash="dot"),
                name=f"{name} init",
                opacity=0.6,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2] + 8.0,
                mode="lines",
                line=dict(color=color, width=7),
                name=f"{name} opt",
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
        for idx, name in enumerate(selected):
            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=initial_seeds[name][:, 0],
                    y=initial_seeds[name][:, 1],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=optimized[name][:, 0],
                    y=optimized[name][:, 1],
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    fig.add_trace(
        go.Table(
            header=dict(values=["name", "energy kJ", "seed dev", "opt dev"], align="left"),
            cells=dict(
                values=[
                    [r["name"] for r in rows[:6]],
                    [f"{float(r['energy_kj']):.1f}" for r in rows[:6]],
                    [f"{float(r['seed_mean_dev_m']):.1f}" for r in rows[:6]],
                    [f"{float(r['opt_mean_dev_m']):.1f}" for r in rows[:6]],
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
        title_text="Crater Entry Multistart Optimization",
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
    output_dir = Path("outputs/crater_multistart")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(CHECKPOINT, map_location=device).to(device)
    goal_airsim = find_local_low_airsim(nemo, CRATER_CENTER_AIRSIM_XY)
    start_ds = airsim_to_dataset(AIRSIM_START)
    goal_ds = airsim_to_dataset(goal_airsim)
    cfg = make_base_config()

    astar_xy = _build_astar_seed(
        nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
        cfg=cfg,
    )
    base_seed = resample_polyline(astar_xy, cfg.num_waypoints)
    base_seed[0] = start_ds[:2]
    base_seed[-1] = goal_ds[:2]
    astar_airsim = dataset_to_airsim_path(path_xy_to_xyz(nemo, astar_xy))

    rows: list[dict[str, object]] = []
    initial_seed_paths: dict[str, np.ndarray] = {}
    optimized_paths: dict[str, np.ndarray] = {}

    for name, amplitude, mode, seed in seed_specs():
        rng = np.random.default_rng(seed) if seed is not None else None
        seed_xy = base_seed.copy() if name == "astar" else biased_seed(base_seed, amplitude_m=amplitude, mode=mode, rng=rng)
        optimized_xy, history, initial_diag, optimized_diag = optimize_physical_path(nemo, seed_xy, cfg)
        seed_airsim = dataset_to_airsim_path(path_xy_to_xyz(nemo, seed_xy))
        control_path = dataset_to_airsim_path(path_xy_to_xyz(nemo, optimized_xy))
        reference = generate_airsim_reference_trajectory(control_path, vehicle=cfg.vehicle, sample_spacing_m=2.0)
        dense_path = np.asarray(reference["positions"], dtype=np.float32)
        initial_seed_paths[name] = seed_airsim
        optimized_paths[name] = dense_path
        np.save(output_dir / f"{name}_initial_path.npy", seed_airsim)
        np.save(output_dir / f"{name}_optimized_path.npy", dense_path)
        np.save(output_dir / f"{name}_control_path.npy", control_path)
        np.savez(output_dir / f"{name}_trajectory.npz", **reference)

        seed_dev = deviation_metrics(seed_xy, base_seed)
        opt_dev = deviation_metrics(optimized_xy, base_seed)
        diag = asdict(optimized_diag)
        row = {
            "name": name,
            "mode": mode,
            "amplitude_m": amplitude,
            "iterations": cfg.optimize_iterations,
            "lr": cfg.optimize_lr,
            "distance_3d_m": diag["distance_3d_m"],
            "time_s": diag["time_s"],
            "energy_kj": diag["energy_j"] / 1000.0,
            "total_cost": diag["total_cost"],
            "max_slope_deg": diag["max_slope_deg"],
            "max_roll_deg": diag["max_roll_deg"],
            "max_curvature_1pm": diag["max_curvature_1pm"],
            "max_breakover_proxy_m": diag["max_breakover_proxy_m"],
            "max_breakover_violation_m": diag["max_breakover_violation_m"],
            "seed_mean_dev_m": seed_dev["mean_dev_m"],
            "seed_max_dev_m": seed_dev["max_dev_m"],
            "opt_mean_dev_m": opt_dev["mean_dev_m"],
            "opt_max_dev_m": opt_dev["max_dev_m"],
            "history_initial": float(history[0]) if history else float("nan"),
            "history_final": float(history[-1]) if history else float("nan"),
        }
        rows.append(row)
        print(
            f"{name}: energy={row['energy_kj']:.1f} kJ, seed_dev={row['seed_mean_dev_m']:.1f} m, "
            f"opt_dev={row['opt_mean_dev_m']:.1f} m, breakover={row['max_breakover_proxy_m']:.3f} m"
        )

    rows_by_cost = sorted(rows, key=lambda r: float(r["total_cost"]))
    rows_by_deviation = sorted(rows, key=lambda r: float(r["opt_mean_dev_m"]), reverse=True)
    write_csv(output_dir / "multistart_results_by_cost.csv", rows_by_cost)
    write_csv(output_dir / "multistart_results_by_deviation.csv", rows_by_deviation)
    (output_dir / "multistart_results.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    np.save(output_dir / "astar_seed.npy", astar_airsim)

    selected_names = [str(row["name"]) for row in rows_by_cost[:3]]
    for row in rows_by_deviation:
        name = str(row["name"])
        if name not in selected_names:
            selected_names.append(name)
        if len(selected_names) >= 6:
            break
    selected_rows = [next(row for row in rows if row["name"] == name) for name in selected_names]
    fig = build_figure(
        nemo,
        astar_airsim,
        {name: initial_seed_paths[name] for name in selected_names},
        {name: optimized_paths[name] for name in selected_names},
        selected_rows,
    )
    html_path = output_dir / "crater_multistart.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"
    print(f"Saved multistart results to {output_dir}")
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
