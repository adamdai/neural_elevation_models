"""Prepare defense-demo AirSim paths and offline metrics tables.

This script packages the currently generated planner outputs into stable
``data/paths`` files for AirSim tracking, and writes a manifest plus offline
energy/time/distance tables for slides.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from nemo import Nemo
from nemo.airsim_trajectory import generate_airsim_reference_trajectory
from scripts.compare_physical_objectives import AIRSIM_SPIRAL_CENTER, AIRSIM_Z_OFFSET
from scripts.estimate_airsim_path_energy import estimate_energy, trajectory_from_airsim_path


DEFAULT_CHECKPOINT = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
)


@dataclass(frozen=True)
class DefensePathSpec:
    name: str
    source_path: Path
    kind: str
    route: str
    display_name: str


def smooth_polyline_naive(path_xy: np.ndarray, *, iterations: int = 35, alpha: float = 0.45) -> np.ndarray:
    """Laplacian smooth a polyline while keeping endpoints fixed."""
    path = np.asarray(path_xy, dtype=np.float64).copy()
    if path.shape[0] <= 2:
        return path
    alpha = float(np.clip(alpha, 0.0, 0.5))
    for _ in range(int(iterations)):
        previous = path.copy()
        path[1:-1] = (1.0 - 2.0 * alpha) * previous[1:-1] + alpha * (previous[:-2] + previous[2:])
    return path


def airsim_xy_to_dataset_xy(path_xy_airsim: np.ndarray) -> np.ndarray:
    path_xy_airsim = np.asarray(path_xy_airsim, dtype=np.float64)
    return np.column_stack(
        [
            path_xy_airsim[:, 1] - AIRSIM_SPIRAL_CENTER[1],
            path_xy_airsim[:, 0] - AIRSIM_SPIRAL_CENTER[0],
        ]
    )


def query_airsim_z(nemo: Nemo, path_xy_airsim: np.ndarray, *, batch_size: int = 65536) -> np.ndarray:
    xy_ds = airsim_xy_to_dataset_xy(path_xy_airsim)
    zs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(xy_ds), batch_size):
            batch = torch.as_tensor(xy_ds[start : start + batch_size], dtype=torch.float32, device=nemo.device)
            z_ds = nemo.h(batch).detach().cpu().numpy().reshape(-1)
            zs.append(z_ds - AIRSIM_Z_OFFSET)
    return np.concatenate(zs, axis=0)


def lift_airsim_xy(nemo: Nemo, path_xy_airsim: np.ndarray) -> np.ndarray:
    z = query_airsim_z(nemo, path_xy_airsim)
    return np.column_stack([path_xy_airsim, z]).astype(np.float32)


def make_smoothed_baseline(nemo: Nemo, astar_path: np.ndarray) -> np.ndarray:
    smoothed_xy = smooth_polyline_naive(astar_path[:, :2])
    smoothed = lift_airsim_xy(nemo, smoothed_xy)
    smoothed[0] = astar_path[0]
    smoothed[-1] = astar_path[-1]
    return smoothed.astype(np.float32)


def path_specs() -> list[DefensePathSpec]:
    return [
        DefensePathSpec(
            name="main_astar",
            source_path=Path("outputs/physical_objective_comparison/airsim_astar_seed.npy"),
            kind="baseline_astar",
            route="main",
            display_name="Main A*",
        ),
        DefensePathSpec(
            name="main_distance",
            source_path=Path("outputs/physical_objective_comparison/airsim_distance_path.npy"),
            kind="optimized_distance",
            route="main",
            display_name="Main distance objective",
        ),
        DefensePathSpec(
            name="main_time",
            source_path=Path("outputs/physical_objective_comparison/airsim_time_path.npy"),
            kind="optimized_time",
            route="main",
            display_name="Main time objective",
        ),
        DefensePathSpec(
            name="main_energy",
            source_path=Path("outputs/physical_objective_comparison/airsim_energy_path.npy"),
            kind="optimized_energy",
            route="main",
            display_name="Main energy objective",
        ),
        DefensePathSpec(
            name="crater_entry_astar",
            source_path=Path("outputs/crater_test_400_300/airsim_crater_astar_path.npy"),
            kind="baseline_astar",
            route="crater_entry",
            display_name="Crater entry A*",
        ),
        DefensePathSpec(
            name="crater_entry_optimized",
            source_path=Path("outputs/crater_test_400_300/airsim_crater_path.npy"),
            kind="optimized_energy",
            route="crater_entry",
            display_name="Crater entry optimized",
        ),
        DefensePathSpec(
            name="crater_exit_astar",
            source_path=Path("outputs/crater_exit_400_300/airsim_crater_exit_astar_path.npy"),
            kind="baseline_astar",
            route="crater_exit",
            display_name="Crater exit A*",
        ),
        DefensePathSpec(
            name="crater_exit_optimized",
            source_path=Path("outputs/crater_exit_400_300/airsim_crater_exit_path.npy"),
            kind="optimized_energy",
            route="crater_exit",
            display_name="Crater exit optimized",
        ),
    ]


def diagnostics_for_path(path: np.ndarray, *, speed_mps: float) -> dict[str, float]:
    trajectory = trajectory_from_airsim_path(path, nominal_speed=float(speed_mps))
    energy = estimate_energy(trajectory)
    dxy = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    dz = np.diff(path[:, 2])
    slope = np.zeros_like(dxy)
    valid = dxy > 1e-8
    slope[valid] = dz[valid] / dxy[valid]
    curvature = _planar_curvature(path[:, :2])
    energy.update(
        {
            "num_points": int(path.shape[0]),
            "start_x": float(path[0, 0]),
            "start_y": float(path[0, 1]),
            "start_z": float(path[0, 2]),
            "goal_x": float(path[-1, 0]),
            "goal_y": float(path[-1, 1]),
            "goal_z": float(path[-1, 2]),
            "max_abs_slope_deg": float(np.degrees(np.arctan(np.max(np.abs(slope), initial=0.0)))),
            "mean_abs_slope_deg": float(np.degrees(np.arctan(np.mean(np.abs(slope)) if slope.size else 0.0))),
            "max_abs_curvature_1pm": float(np.max(np.abs(curvature), initial=0.0)),
        }
    )
    return energy


def _planar_curvature(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 3:
        return np.zeros(xy.shape[0], dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if s[-1] <= 1e-8:
        return np.zeros(xy.shape[0], dtype=np.float64)
    dx = np.gradient(xy[:, 0], s, edge_order=1)
    dy = np.gradient(xy[:, 1], s, edge_order=1)
    ddx = np.gradient(dx, s, edge_order=1)
    ddy = np.gradient(dy, s, edge_order=1)
    denom = np.clip((dx**2 + dy**2) ** 1.5, 1e-8, None)
    return (dx * ddy - dy * ddx) / denom


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/defense_demo"))
    parser.add_argument("--data-paths-dir", type=Path, default=Path("data/paths"))
    parser.add_argument("--speed", type=float, default=6.0)
    parser.add_argument("--sample-spacing", type=float, default=2.0)
    parser.add_argument("--no-copy-data-paths", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_paths_dir = args.output_dir / "paths"
    output_traj_dir = args.output_dir / "trajectories"
    output_paths_dir.mkdir(parents=True, exist_ok=True)
    output_traj_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_data_paths:
        args.data_paths_dir.mkdir(parents=True, exist_ok=True)

    nemo = Nemo.load_checkpoint(args.checkpoint, map_location=args.device).to(args.device)
    specs = path_specs()
    generated: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    manifest_paths: list[dict[str, object]] = []

    for spec in specs:
        if not spec.source_path.exists():
            raise FileNotFoundError(f"Missing source path for {spec.name}: {spec.source_path}")
        path = np.load(spec.source_path).astype(np.float32)
        generated[spec.name] = path

    generated["crater_entry_smooth"] = make_smoothed_baseline(nemo, generated["crater_entry_astar"])
    generated["crater_exit_smooth"] = make_smoothed_baseline(nemo, generated["crater_exit_astar"])
    specs.extend(
        [
            DefensePathSpec(
                name="crater_entry_smooth",
                source_path=Path("<generated from crater_entry_astar>"),
                kind="baseline_smoothed_astar",
                route="crater_entry",
                display_name="Crater entry smoothed A*",
            ),
            DefensePathSpec(
                name="crater_exit_smooth",
                source_path=Path("<generated from crater_exit_astar>"),
                kind="baseline_smoothed_astar",
                route="crater_exit",
                display_name="Crater exit smoothed A*",
            ),
        ]
    )

    for spec in specs:
        path = generated[spec.name]
        output_path = output_paths_dir / f"{spec.name}.npy"
        np.save(output_path, path)
        if not args.no_copy_data_paths:
            shutil.copy2(output_path, args.data_paths_dir / output_path.name)

        trajectory = generate_airsim_reference_trajectory(path, sample_spacing_m=float(args.sample_spacing))
        trajectory_path = output_traj_dir / f"{spec.name}_trajectory.npz"
        np.savez(trajectory_path, **trajectory)

        diagnostics = diagnostics_for_path(path, speed_mps=float(args.speed))
        row = {
            "name": spec.name,
            "route": spec.route,
            "kind": spec.kind,
            "display_name": spec.display_name,
            "path_file": str(output_path),
            "tracker_path_file": str(args.data_paths_dir / output_path.name) if not args.no_copy_data_paths else "",
            "trajectory_file": str(trajectory_path),
            **diagnostics,
        }
        rows.append(row)
        manifest_paths.append(row)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "speed_mps": float(args.speed),
        "sample_spacing_m": float(args.sample_spacing),
        "data_paths_dir": "" if args.no_copy_data_paths else str(args.data_paths_dir),
        "paths": manifest_paths,
    }
    manifest_path = args.output_dir / "defense_paths_manifest.json"
    metrics_csv = args.output_dir / "defense_offline_metrics.csv"
    metrics_md = args.output_dir / "defense_offline_metrics.md"
    planner_csv = args.output_dir / "defense_planner_metrics.csv"
    planner_md = args.output_dir / "defense_planner_metrics.md"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(metrics_csv, rows)
    metrics_md.write_text(_markdown_table(rows), encoding="utf-8")
    planner_rows = _planner_metric_rows()
    write_csv(planner_csv, planner_rows)
    planner_md.write_text(_markdown_table(planner_rows), encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote metrics CSV: {metrics_csv}")
    print(f"Wrote metrics Markdown: {metrics_md}")
    print(f"Wrote planner metrics CSV: {planner_csv}")
    print(f"Wrote planner metrics Markdown: {planner_md}")
    if not args.no_copy_data_paths:
        print(f"Copied tracker paths to: {args.data_paths_dir}")


def _markdown_table(rows: list[dict[str, object]]) -> str:
    preferred = [
        "name",
        "route",
        "kind",
        "distance_3d_m",
        "time_s",
        "energy_kj",
        "max_slope_deg",
        "max_abs_slope_deg",
        "max_curvature_1pm",
        "max_abs_curvature_1pm",
    ]
    if not rows:
        return ""
    columns = [col for col in preferred if col in rows[0]]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out) + "\n"


def _planner_metric_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(
        _physical_objective_rows(
            Path("outputs/physical_objective_comparison/physical_objective_metrics.json"),
            route="main",
        )
    )
    rows.extend(
        _physical_objective_rows(
            Path("outputs/physical_objective_crater/physical_objective_metrics.json"),
            route="crater_far",
        )
    )
    rows.extend(
        _single_crater_rows(
            Path("outputs/crater_test_400_300/crater_test_metrics.json"),
            route="crater_entry",
        )
    )
    rows.extend(
        _single_crater_rows(
            Path("outputs/crater_exit_400_300/crater_exit_metrics.json"),
            route="crater_exit",
        )
    )
    return rows


def _physical_objective_rows(path: Path, *, route: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    rows = []
    for name in ("distance", "time", "energy"):
        if name not in metrics:
            continue
        row = _metric_row(f"{route}_{name}", route, f"optimized_{name}", metrics[name])
        rows.append(row)
    if "energy_initial" in metrics:
        rows.append(_metric_row(f"{route}_astar_seed", route, "baseline_astar", metrics["energy_initial"]))
    return rows


def _single_crater_rows(path: Path, *, route: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        _metric_row(f"{route}_astar_seed", route, "baseline_astar", data["initial"]),
        _metric_row(f"{route}_optimized", route, "optimized_energy", data["optimized"]),
    ]


def _metric_row(name: str, route: str, kind: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "route": route,
        "kind": kind,
        "distance_3d_m": float(metrics.get("distance_3d_m", np.nan)),
        "time_s": float(metrics.get("time_s", np.nan)),
        "energy_kj": float(metrics.get("energy_j", np.nan)) / 1000.0,
        "max_slope_deg": float(metrics.get("max_slope_deg", np.nan)),
        "max_curvature_1pm": float(metrics.get("max_curvature_1pm", np.nan)),
        "max_yaw_rate_radps": float(metrics.get("max_yaw_rate_radps", np.nan)),
        "max_accel_mps2": float(metrics.get("max_accel_mps2", np.nan)),
    }


if __name__ == "__main__":
    main()
