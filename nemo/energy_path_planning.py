from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nemo.nemo import Nemo
from nemo.path_planning import (
    astar_grid_path,
    build_astar_cost_grid,
    buffered_corner_points,
    path_xy_to_xyz,
    resample_polyline,
    sample_height_and_gradient_grid,
    _xy_to_rc,
)


@dataclass(frozen=True)
class EnergyModelConfig:
    mass_kg: float = 2000.0
    gravity_mps2: float = 9.81
    drag_coefficient: float = 0.3
    frontal_area_m2: float = 2.5
    rolling_resistance_coeff: float = 0.015
    air_density_kg_m3: float = 1.225
    drivetrain_efficiency: float = 0.8
    target_speed_mps: float = 6.0


@dataclass(frozen=True)
class EnergyPathPlanningConfig:
    astar_grid_resolution_x: int = 256
    astar_grid_resolution_y: int = 256
    astar_height_weight: float = 0.0
    astar_slope_weight: float = 20.0
    astar_max_slope_deg: float = 25.0
    astar_slope_reference_deg: float = 10.0
    astar_slope_exponent: float = 3.0
    astar_step_weight: float = 1.0
    buffer_fraction: float = 0.1
    num_waypoints: int = 80
    optimize_iterations: int = 500
    optimize_lr: float = 2e-2
    energy_weight: float = 1.0
    smoothness_weight: float = 1e-3
    anchor_weight: float = 1e-4
    slope_violation_weight: float = 1e5
    max_slope_deg: float = 25.0
    batch_size: int = 65536
    energy: EnergyModelConfig = EnergyModelConfig()


@dataclass(frozen=True)
class EnergyPathPlanningResult:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    astar_path_xy: np.ndarray
    initial_path_xy: np.ndarray
    optimized_path_xy: np.ndarray
    astar_path_xyz: np.ndarray
    initial_path_xyz: np.ndarray
    optimized_path_xyz: np.ndarray
    initial_energy_j: float
    optimized_energy_j: float
    initial_objective: float
    optimized_objective: float
    cost_history: list[float]


def energy_objective(
    nemo: Nemo,
    path_xy: torch.Tensor,
    *,
    reference_xy: torch.Tensor | None = None,
    config: EnergyPathPlanningConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable path energy objective using the AirSim tracker SUV model."""
    if path_xy.ndim != 2 or path_xy.shape[-1] != 2:
        raise ValueError("Expected path_xy with shape (N, 2).")
    if path_xy.shape[0] < 3:
        raise ValueError("Energy path optimization requires at least three waypoints.")

    z = nemo.field.h(path_xy).squeeze(-1)
    xyz = torch.cat([path_xy, z[:, None]], dim=-1)
    dxyz = xyz[1:] - xyz[:-1]
    dxy = path_xy[1:] - path_xy[:-1]
    ds_xy = torch.linalg.norm(dxy, dim=-1).clamp_min(1e-8)
    ds_3d = torch.linalg.norm(dxyz, dim=-1).clamp_min(1e-8)
    dz = z[1:] - z[:-1]

    sin_pitch = dz / ds_3d
    cos_pitch = ds_xy / ds_3d
    speed = float(config.energy.target_speed_mps)
    dt = ds_3d / max(speed, 1e-8)

    mass = float(config.energy.mass_kg)
    gravity = float(config.energy.gravity_mps2)
    drag = 0.5 * float(config.energy.air_density_kg_m3) * float(config.energy.drag_coefficient) * float(
        config.energy.frontal_area_m2
    ) * speed**2
    rolling = mass * gravity * float(config.energy.rolling_resistance_coeff) * cos_pitch
    slope = mass * gravity * sin_pitch
    force = drag + rolling + slope
    power = torch.relu(force * speed / max(float(config.energy.drivetrain_efficiency), 1e-8))
    energy_j = torch.sum(power * dt)

    curvature = path_xy[2:] - 2.0 * path_xy[1:-1] + path_xy[:-2]
    smoothness = curvature.pow(2).sum(dim=-1).mean()
    if reference_xy is None:
        anchor = path_xy.new_tensor(0.0)
    else:
        anchor = (path_xy[1:-1] - reference_xy[1:-1]).pow(2).sum(dim=-1).mean()

    slope_angle = torch.atan2(torch.abs(dz), ds_xy)
    max_slope = np.deg2rad(float(config.max_slope_deg))
    slope_violation = torch.relu(slope_angle - float(max_slope)).pow(2).mean()

    objective = (
        float(config.energy_weight) * energy_j
        + float(config.smoothness_weight) * smoothness
        + float(config.anchor_weight) * anchor
        + float(config.slope_violation_weight) * slope_violation
    )
    return objective, {
        "energy_j": energy_j,
        "path_length_3d_m": torch.sum(ds_3d),
        "time_s": torch.sum(dt),
        "smoothness": smoothness,
        "anchor": anchor,
        "slope_violation": slope_violation,
        "max_slope_rad": torch.max(slope_angle),
    }


def optimize_energy_path(
    nemo: Nemo,
    path_xy: np.ndarray,
    *,
    config: EnergyPathPlanningConfig,
) -> tuple[np.ndarray, list[float], float, float, float, float]:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError("Expected path_xy with shape (N, 2).")
    if len(path_xy) < 3:
        raise ValueError("Need at least three points to optimize a path.")

    device = nemo.device
    start = torch.as_tensor(path_xy[0], dtype=torch.float32, device=device)
    goal = torch.as_tensor(path_xy[-1], dtype=torch.float32, device=device)
    reference = torch.as_tensor(path_xy, dtype=torch.float32, device=device)
    interior = torch.nn.Parameter(reference[1:-1].detach().clone())
    optimizer = torch.optim.Adam([interior], lr=float(config.optimize_lr))

    with torch.no_grad():
        initial_objective_t, initial_metrics = energy_objective(nemo, reference, reference_xy=reference, config=config)
        initial_objective = float(initial_objective_t.detach().cpu().item())
        initial_energy = float(initial_metrics["energy_j"].detach().cpu().item())

    best_objective = initial_objective
    best_energy = initial_energy
    best_interior = interior.detach().clone()
    cost_history: list[float] = []
    x_min, x_max = float(nemo.field.bounds[0][0]), float(nemo.field.bounds[0][1])
    y_min, y_max = float(nemo.field.bounds[1][0]), float(nemo.field.bounds[1][1])

    for _ in range(int(config.optimize_iterations)):
        optimizer.zero_grad(set_to_none=True)
        path = torch.cat([start[None], interior, goal[None]], dim=0)
        objective, _ = energy_objective(nemo, path, reference_xy=reference, config=config)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            interior[:, 0].clamp_(x_min, x_max)
            interior[:, 1].clamp_(y_min, y_max)
            candidate = torch.cat([start[None], interior, goal[None]], dim=0)
            candidate_objective_t, candidate_metrics = energy_objective(
                nemo,
                candidate,
                reference_xy=reference,
                config=config,
            )
            candidate_objective = float(candidate_objective_t.detach().cpu().item())
            candidate_energy = float(candidate_metrics["energy_j"].detach().cpu().item())
            if candidate_objective < best_objective:
                best_objective = candidate_objective
                best_energy = candidate_energy
                best_interior = interior.detach().clone()
        cost_history.append(candidate_objective)

    optimized_path = torch.cat([start[None], best_interior, goal[None]], dim=0)
    return (
        optimized_path.detach().cpu().numpy().astype(np.float32),
        cost_history,
        initial_objective,
        best_objective,
        initial_energy,
        best_energy,
    )


def plan_energy_path(
    nemo: Nemo,
    *,
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    config: EnergyPathPlanningConfig | None = None,
) -> EnergyPathPlanningResult:
    cfg = config or EnergyPathPlanningConfig()
    if start_xy is None or goal_xy is None:
        default_start, default_goal = buffered_corner_points(nemo.field.bounds, buffer_fraction=cfg.buffer_fraction)
        start_xy = start_xy or default_start
        goal_xy = goal_xy or default_goal

    xx, yy, z, gx, gy = sample_height_and_gradient_grid(
        nemo,
        resolution_x=cfg.astar_grid_resolution_x,
        resolution_y=cfg.astar_grid_resolution_y,
        batch_size=cfg.batch_size,
    )
    cost_grid = build_astar_cost_grid(
        z,
        gx,
        gy,
        height_weight=cfg.astar_height_weight,
        slope_weight=cfg.astar_slope_weight,
        max_slope_deg=cfg.astar_max_slope_deg,
        slope_reference_deg=cfg.astar_slope_reference_deg,
        slope_exponent=cfg.astar_slope_exponent,
    )
    start_rc = _xy_to_rc(start_xy, xx[0, :], yy[:, 0])
    goal_rc = _xy_to_rc(goal_xy, xx[0, :], yy[:, 0])
    astar_rc = astar_grid_path(
        cost_grid,
        start_rc,
        goal_rc,
        step_weight=cfg.astar_step_weight,
        x_axis=xx[0, :],
        y_axis=yy[:, 0],
    )
    astar_path_xy = np.column_stack([xx[astar_rc[:, 0], astar_rc[:, 1]], yy[astar_rc[:, 0], astar_rc[:, 1]]]).astype(
        np.float32
    )
    astar_path_xy[0] = np.asarray(start_xy, dtype=np.float32)
    astar_path_xy[-1] = np.asarray(goal_xy, dtype=np.float32)

    initial_path_xy = resample_polyline(astar_path_xy, cfg.num_waypoints)
    initial_path_xy[0] = np.asarray(start_xy, dtype=np.float32)
    initial_path_xy[-1] = np.asarray(goal_xy, dtype=np.float32)

    optimized_path_xy, history, initial_objective, final_objective, initial_energy, final_energy = optimize_energy_path(
        nemo,
        initial_path_xy,
        config=cfg,
    )
    optimized_path_xy[0] = np.asarray(start_xy, dtype=np.float32)
    optimized_path_xy[-1] = np.asarray(goal_xy, dtype=np.float32)

    return EnergyPathPlanningResult(
        start_xy=(float(start_xy[0]), float(start_xy[1])),
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        astar_path_xy=astar_path_xy,
        initial_path_xy=initial_path_xy,
        optimized_path_xy=optimized_path_xy,
        astar_path_xyz=path_xy_to_xyz(nemo, astar_path_xy),
        initial_path_xyz=path_xy_to_xyz(nemo, initial_path_xy),
        optimized_path_xyz=path_xy_to_xyz(nemo, optimized_path_xy),
        initial_energy_j=initial_energy,
        optimized_energy_j=final_energy,
        initial_objective=initial_objective,
        optimized_objective=final_objective,
        cost_history=history,
    )
