"""Crater-entry planning sanity check directly over the GT DEM grid."""

from __future__ import annotations

import json
import webbrowser
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as F

from nemo import HeightField, Nemo
from nemo.airsim_trajectory import generate_airsim_reference_trajectory
from nemo.physical_objective_planner import (
    PhysicalObjectivePlannerConfig,
    SafetyConstraintConfig,
    VehicleModelConfig,
    plan_physical_objective_path,
)
from scripts.compare_physical_objectives import AIRSIM_START, AIRSIM_Z_OFFSET, dataset_to_airsim_path
from scripts.crater_test import (
    CRATER_CENTER_AIRSIM_XY,
    airsim_to_dataset,
    build_figure,
    find_local_low_airsim,
    start_preview_server,
)


DEM_PATH = Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/dem_pc.npy")


class TorchGridDEMField(HeightField):
    """Differentiable bilinear height-field wrapper around a regular DEM grid."""

    def __init__(self, x_axis: np.ndarray, y_axis: np.ndarray, z_grid: np.ndarray) -> None:
        bounds = ((float(x_axis[0]), float(x_axis[-1])), (float(y_axis[0]), float(y_axis[-1])))
        super().__init__(bounds)
        self.register_buffer("x_axis", torch.as_tensor(x_axis, dtype=torch.float32))
        self.register_buffer("y_axis", torch.as_tensor(y_axis, dtype=torch.float32))
        self.register_buffer("z_grid", torch.as_tensor(z_grid, dtype=torch.float32)[None, None])
        self._device_anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0]
        y = xy[:, 1]
        x0, x1 = self.x_axis[0], self.x_axis[-1]
        y0, y1 = self.y_axis[0], self.y_axis[-1]
        x_norm = 2.0 * (x - x0) / torch.clamp(x1 - x0, min=1e-8) - 1.0
        y_norm = 2.0 * (y - y0) / torch.clamp(y1 - y0, min=1e-8) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).reshape(1, -1, 1, 2)
        z = F.grid_sample(
            self.z_grid.to(dtype=xy.dtype, device=xy.device),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return z.reshape(-1, 1)


def load_gt_dem_nemo(path: Path, *, device: str) -> Nemo:
    dem = np.load(path).astype(np.float32)
    x = dem[:, 0]
    y = dem[:, 1]
    z = dem[:, 2]
    x_axis = np.unique(x)
    nx = x_axis.size
    if dem.shape[0] % nx != 0:
        raise ValueError(f"DEM point count {dem.shape[0]} is not divisible by x-axis size {nx}.")
    ny = dem.shape[0] // nx
    y_axis_in_file = y.reshape(ny, nx)[:, 0]
    z_grid = z.reshape(ny, nx)
    if y_axis_in_file[0] > y_axis_in_file[-1]:
        y_axis = y_axis_in_file[::-1].copy()
        z_grid = z_grid[::-1].copy()
    else:
        y_axis = y_axis_in_file.copy()
    field = TorchGridDEMField(x_axis=x_axis, y_axis=y_axis, z_grid=z_grid)
    return Nemo(field).to(device)


def main() -> None:
    output_dir = Path("outputs/crater_test_gt_dem")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dem_nemo = load_gt_dem_nemo(DEM_PATH, device=device)

    crater_low_airsim = find_local_low_airsim(dem_nemo, CRATER_CENTER_AIRSIM_XY)
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
        dem_nemo,
        start_xy=(float(start_ds[0]), float(start_ds[1])),
        goal_xy=(float(goal_ds[0]), float(goal_ds[1])),
        config=config,
    )

    astar = dataset_to_airsim_path(result.astar_path_xyz)
    control_path = dataset_to_airsim_path(result.optimized_path_xyz)
    reference = generate_airsim_reference_trajectory(control_path, vehicle=vehicle, sample_spacing_m=2.0)
    optimized = np.asarray(reference["positions"], dtype=np.float32)

    np.save(output_dir / "airsim_crater_gt_dem_astar_path.npy", astar)
    np.save(output_dir / "airsim_crater_gt_dem_control_path.npy", control_path)
    np.save(output_dir / "airsim_crater_gt_dem_path.npy", optimized)
    np.savez(output_dir / "airsim_crater_gt_dem_trajectory.npz", **reference)

    metrics = {
        "dem_path": str(DEM_PATH),
        "config": asdict(config),
        "start_airsim": AIRSIM_START.tolist(),
        "crater_center_airsim_xy": CRATER_CENTER_AIRSIM_XY.tolist(),
        "goal_airsim": crater_low_airsim.tolist(),
        "initial": asdict(result.initial_diagnostics),
        "optimized": asdict(result.optimized_diagnostics),
    }
    (output_dir / "crater_gt_dem_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig = build_figure(dem_nemo, astar, optimized, metrics["optimized"])
    fig.update_layout(title_text="GT DEM Crater Entry Planning")
    html_path = output_dir / "crater_gt_dem.html"
    fig.write_html(html_path)
    server_process, base_url = start_preview_server(output_dir.resolve())
    plot_url = f"{base_url}/{quote(html_path.name)}"

    print(f"GT DEM path: {DEM_PATH}")
    print(f"Crater center AirSim XY: {CRATER_CENTER_AIRSIM_XY.tolist()}")
    print(f"Goal local low AirSim: {crater_low_airsim.tolist()}")
    print(
        "optimized: "
        f"distance={result.optimized_diagnostics.distance_3d_m:.1f} m, "
        f"time={result.optimized_diagnostics.time_s:.1f} s, "
        f"energy={result.optimized_diagnostics.energy_j / 1000.0:.1f} kJ, "
        f"max_slope={result.optimized_diagnostics.max_slope_deg:.1f} deg, "
        f"max_roll={result.optimized_diagnostics.max_roll_deg:.1f} deg, "
        f"max_footprint={result.optimized_diagnostics.max_footprint_clearance_violation_m:.3f} m"
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Plot preview: {plot_url}")
    print(f"Preview server PID: {server_process.pid}")
    webbrowser.open(plot_url, new=2)


if __name__ == "__main__":
    main()
