"""One-off crater exit planning test from the x=400, y=300 crater to the main goal."""

from __future__ import annotations

import json
import webbrowser
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from nemo import Nemo
from nemo.airsim_trajectory import generate_airsim_reference_trajectory
from nemo.physical_objective_planner import (
    PhysicalObjectivePlannerConfig,
    SafetyConstraintConfig,
    VehicleModelConfig,
    plan_physical_objective_path,
)
from scripts.compare_physical_objectives import AIRSIM_GOAL, dataset_to_airsim_path
from scripts.crater_test import (
    CRATER_CENTER_AIRSIM_XY,
    airsim_to_dataset,
    build_figure,
    find_local_low_airsim,
    start_preview_server,
)


def main() -> None:
    output_dir = Path("outputs/crater_exit_400_300")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
        "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(checkpoint, map_location=device).to(device)

    start_airsim = find_local_low_airsim(nemo, CRATER_CENTER_AIRSIM_XY)
    goal_airsim = AIRSIM_GOAL.astype(np.float64)
    start_ds = airsim_to_dataset(start_airsim)
    goal_ds = airsim_to_dataset(goal_airsim)

    vehicle = VehicleModelConfig()
    safety = SafetyConstraintConfig(max_slope_deg=45.0, max_roll_deg=25.0)
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

    np.save(output_dir / "airsim_crater_exit_astar_path.npy", astar)
    np.save(output_dir / "airsim_crater_exit_control_path.npy", control_path)
    np.save(output_dir / "airsim_crater_exit_path.npy", optimized)
    np.savez(output_dir / "airsim_crater_exit_trajectory.npz", **reference)
    metrics = {
        "config": asdict(config),
        "crater_center_airsim_xy": CRATER_CENTER_AIRSIM_XY.tolist(),
        "start_airsim": start_airsim.tolist(),
        "goal_airsim": goal_airsim.tolist(),
        "initial": asdict(result.initial_diagnostics),
        "optimized": asdict(result.optimized_diagnostics),
    }
    (output_dir / "crater_exit_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig = build_figure(nemo, astar, optimized, metrics["optimized"])
    fig.update_layout(title_text="No-Max-Slope Crater Exit Test")
    html_path = output_dir / "crater_exit.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"

    print(f"Crater center AirSim XY: {CRATER_CENTER_AIRSIM_XY.tolist()}")
    print(f"Start local low AirSim: {start_airsim.tolist()}")
    print(f"Goal AirSim: {goal_airsim.tolist()}")
    print(f"A* max slope cutoff: disabled for this test ({config.astar_max_slope_deg} deg)")
    print(
        "optimized: "
        f"distance={result.optimized_diagnostics.distance_3d_m:.1f} m, "
        f"time={result.optimized_diagnostics.time_s:.1f} s, "
        f"energy={result.optimized_diagnostics.energy_j / 1000.0:.1f} kJ, "
        f"max_slope={result.optimized_diagnostics.max_slope_deg:.1f} deg, "
        f"max_roll={result.optimized_diagnostics.max_roll_deg:.1f} deg"
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
