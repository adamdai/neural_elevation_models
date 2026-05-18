from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


ObjectiveName = Literal["distance", "time", "energy"]


@dataclass(frozen=True)
class VehicleModelConfig:
    mass_kg: float = 2000.0
    gravity_mps2: float = 9.81
    drag_coefficient: float = 0.3
    frontal_area_m2: float = 2.5
    rolling_resistance_coeff: float = 0.015
    air_density_kg_m3: float = 1.225
    drivetrain_efficiency: float = 0.8
    nominal_speed_mps: float = 6.0
    max_lateral_accel_mps2: float = 2.0
    max_longitudinal_accel_mps2: float = 1.5
    max_yaw_rate_radps: float = 0.6
    min_speed_mps: float = 0.5
    slope_speed_scale: float = 1.5
    wheelbase_m: float = 2.8
    track_width_m: float = 1.7
    ground_clearance_m: float = 0.25


@dataclass(frozen=True)
class SafetyConstraintConfig:
    max_slope_deg: float = 25.0
    max_roll_deg: float = 15.0
    min_turn_radius_m: float = 12.0
    max_yaw_rate_radps: float = 0.6
    max_accel_mps2: float = 1.5
    slope_weight: float = 5e5
    roll_weight: float = 5e5
    curvature_weight: float = 5e10
    yaw_rate_weight: float = 1e4
    accel_weight: float = 1e4
    smoothness_weight: float = 1e3
    anchor_weight: float = 1e-3
    breakover_clearance_margin_m: float = 0.15
    breakover_weight: float = 0.0
    footprint_clearance_margin_m: float = 0.05
    footprint_clearance_weight: float = 0.0


@dataclass(frozen=True)
class PhysicalObjectivePlannerConfig:
    objective: ObjectiveName = "energy"
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
    objective_scale: float = 1.0
    batch_size: int = 65536
    vehicle: VehicleModelConfig = VehicleModelConfig()
    safety: SafetyConstraintConfig = SafetyConstraintConfig()


@dataclass(frozen=True)
class PhysicalPlanDiagnostics:
    objective: float
    primary_cost: float
    total_cost: float
    distance_3d_m: float
    time_s: float
    energy_j: float
    max_slope_deg: float
    max_roll_deg: float
    max_curvature_1pm: float
    max_yaw_rate_radps: float
    max_accel_mps2: float
    max_forward_terrain_curvature_1pm: float = 0.0
    max_breakover_proxy_m: float = 0.0
    max_breakover_violation_m: float = 0.0
    max_footprint_clearance_violation_m: float = 0.0
    mean_footprint_clearance_violation_m: float = 0.0


@dataclass(frozen=True)
class PhysicalObjectivePlanningResult:
    objective: ObjectiveName
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    astar_path_xy: np.ndarray
    initial_path_xy: np.ndarray
    optimized_path_xy: np.ndarray
    astar_path_xyz: np.ndarray
    initial_path_xyz: np.ndarray
    optimized_path_xyz: np.ndarray
    initial_diagnostics: PhysicalPlanDiagnostics
    optimized_diagnostics: PhysicalPlanDiagnostics
    cost_history: list[float]


def plan_physical_objective_path(
    nemo: Nemo,
    *,
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    config: PhysicalObjectivePlannerConfig | None = None,
) -> PhysicalObjectivePlanningResult:
    cfg = config or PhysicalObjectivePlannerConfig()
    if start_xy is None or goal_xy is None:
        default_start, default_goal = buffered_corner_points(nemo.field.bounds, buffer_fraction=cfg.buffer_fraction)
        start_xy = start_xy or default_start
        goal_xy = goal_xy or default_goal

    astar_path_xy = _build_astar_seed(nemo, start_xy=start_xy, goal_xy=goal_xy, cfg=cfg)
    initial_path_xy = resample_polyline(astar_path_xy, cfg.num_waypoints)
    initial_path_xy[0] = np.asarray(start_xy, dtype=np.float32)
    initial_path_xy[-1] = np.asarray(goal_xy, dtype=np.float32)
    optimized_path_xy, history, initial_diag, optimized_diag = optimize_physical_path(nemo, initial_path_xy, cfg)

    return PhysicalObjectivePlanningResult(
        objective=cfg.objective,
        start_xy=(float(start_xy[0]), float(start_xy[1])),
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        astar_path_xy=astar_path_xy,
        initial_path_xy=initial_path_xy,
        optimized_path_xy=optimized_path_xy,
        astar_path_xyz=path_xy_to_xyz(nemo, astar_path_xy),
        initial_path_xyz=path_xy_to_xyz(nemo, initial_path_xy),
        optimized_path_xyz=path_xy_to_xyz(nemo, optimized_path_xy),
        initial_diagnostics=initial_diag,
        optimized_diagnostics=optimized_diag,
        cost_history=history,
    )


def optimize_physical_path(
    nemo: Nemo,
    path_xy: np.ndarray,
    cfg: PhysicalObjectivePlannerConfig,
) -> tuple[np.ndarray, list[float], PhysicalPlanDiagnostics, PhysicalPlanDiagnostics]:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    device = nemo.device
    start = torch.as_tensor(path_xy[0], dtype=torch.float32, device=device)
    goal = torch.as_tensor(path_xy[-1], dtype=torch.float32, device=device)
    reference = torch.as_tensor(path_xy, dtype=torch.float32, device=device)
    interior = torch.nn.Parameter(reference[1:-1].detach().clone())
    optimizer = torch.optim.Adam([interior], lr=float(cfg.optimize_lr))

    initial_total, initial_terms = physical_objective(nemo, reference, reference_xy=reference, cfg=cfg)
    initial_diag = _diagnostics_from_terms(initial_terms, cfg.objective)
    best_total = float(initial_total.detach().cpu().item())
    best_terms = {key: value.detach().clone() for key, value in initial_terms.items()}
    best_interior = interior.detach().clone()
    cost_history: list[float] = []
    x_min, x_max = float(nemo.field.bounds[0][0]), float(nemo.field.bounds[0][1])
    y_min, y_max = float(nemo.field.bounds[1][0]), float(nemo.field.bounds[1][1])

    for _ in range(int(cfg.optimize_iterations)):
        optimizer.zero_grad(set_to_none=True)
        path = torch.cat([start[None], interior, goal[None]], dim=0)
        total, _ = physical_objective(nemo, path, reference_xy=reference, cfg=cfg)
        total.backward()
        optimizer.step()
        with torch.no_grad():
            interior[:, 0].clamp_(x_min, x_max)
            interior[:, 1].clamp_(y_min, y_max)
            candidate = torch.cat([start[None], interior, goal[None]], dim=0)
            candidate_total, candidate_terms = physical_objective(nemo, candidate, reference_xy=reference, cfg=cfg)
            candidate_value = float(candidate_total.detach().cpu().item())
            if candidate_value < best_total:
                best_total = candidate_value
                best_terms = {key: value.detach().clone() for key, value in candidate_terms.items()}
                best_interior = interior.detach().clone()
        cost_history.append(candidate_value)

    optimized_path = torch.cat([start[None], best_interior, goal[None]], dim=0)
    optimized_diag = _diagnostics_from_terms(best_terms, cfg.objective)
    return optimized_path.detach().cpu().numpy().astype(np.float32), cost_history, initial_diag, optimized_diag


def physical_objective(
    nemo: Nemo,
    path_xy: torch.Tensor,
    *,
    reference_xy: torch.Tensor | None,
    cfg: PhysicalObjectivePlannerConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rollout = differentiable_rollout(nemo, path_xy, cfg.vehicle)
    safety = cfg.safety
    max_curvature = 1.0 / max(float(safety.min_turn_radius_m), 1e-8)

    slope_violation = torch.relu(torch.abs(rollout["slope_angle"]) - np.deg2rad(float(safety.max_slope_deg))).pow(2).mean()
    roll_violation = torch.relu(torch.abs(rollout["roll"]) - np.deg2rad(float(safety.max_roll_deg))).pow(2).mean()
    curvature_violation = torch.relu(torch.abs(rollout["curvature"]) - max_curvature).pow(2).mean()
    yaw_rate_violation = torch.relu(torch.abs(rollout["yaw_rate"]) - float(safety.max_yaw_rate_radps)).pow(2).mean()
    accel_violation = torch.relu(torch.abs(rollout["accel"]) - float(safety.max_accel_mps2)).pow(2).mean()
    breakover_violation = torch.relu(rollout["breakover_proxy"] - float(safety.breakover_clearance_margin_m)).pow(2).mean()
    footprint_clearance_values = torch.relu(
        rollout["footprint_clearance_raw"] + float(safety.footprint_clearance_margin_m)
    )
    footprint_clearance_violation = footprint_clearance_values.pow(2).mean()
    smoothness = rollout["curvature"].pow(2).mean() + 0.01 * rollout["yaw_rate"].pow(2).mean()

    if reference_xy is None:
        anchor = path_xy.new_tensor(0.0)
    else:
        anchor = (path_xy[1:-1] - reference_xy[1:-1]).pow(2).sum(dim=-1).mean()

    primary = {
        "distance": rollout["distance_3d"],
        "time": rollout["time"],
        "energy": rollout["energy"],
    }[cfg.objective]
    safety_cost = (
        float(safety.slope_weight) * slope_violation
        + float(safety.roll_weight) * roll_violation
        + float(safety.curvature_weight) * curvature_violation
        + float(safety.yaw_rate_weight) * yaw_rate_violation
        + float(safety.accel_weight) * accel_violation
        + float(safety.breakover_weight) * breakover_violation
        + float(safety.footprint_clearance_weight) * footprint_clearance_violation
        + float(safety.smoothness_weight) * smoothness
        + float(safety.anchor_weight) * anchor
    )
    total = float(cfg.objective_scale) * primary + safety_cost
    terms = {
        **rollout,
        "primary": primary,
        "total": total,
        "slope_violation": slope_violation,
        "roll_violation": roll_violation,
        "curvature_violation": curvature_violation,
        "yaw_rate_violation": yaw_rate_violation,
        "accel_violation": accel_violation,
        "breakover_violation": breakover_violation,
        "breakover_clearance_margin_m": path_xy.new_tensor(float(safety.breakover_clearance_margin_m)),
        "footprint_clearance_violation": footprint_clearance_values,
        "footprint_clearance_violation_cost": footprint_clearance_violation,
        "smoothness": smoothness,
        "anchor": anchor,
    }
    return total, terms


def differentiable_rollout(
    nemo: Nemo,
    path_xy: torch.Tensor,
    vehicle: VehicleModelConfig,
) -> dict[str, torch.Tensor]:
    z = nemo.field.h(path_xy).squeeze(-1)
    xyz = torch.cat([path_xy, z[:, None]], dim=-1)
    dxy = path_xy[1:] - path_xy[:-1]
    dz = z[1:] - z[:-1]
    ds_xy = torch.linalg.norm(dxy, dim=-1).clamp_min(1e-8)
    ds_3d = torch.sqrt(ds_xy.pow(2) + dz.pow(2)).clamp_min(1e-8)
    slope = dz / ds_xy
    slope_angle = torch.atan(slope)

    yaw = torch.atan2(dxy[:, 1], dxy[:, 0])
    yaw_unwrapped = _unwrap_torch(yaw)
    heading_change = yaw_unwrapped[1:] - yaw_unwrapped[:-1]
    avg_ds = 0.5 * (ds_xy[1:] + ds_xy[:-1]).clamp_min(1e-8)
    curvature_inner = heading_change / avg_ds
    curvature = torch.zeros_like(ds_xy)
    if curvature_inner.numel():
        curvature[:-1] = curvature_inner
        curvature[-1] = curvature_inner[-1]

    grad = _finite_difference_grad(nemo, path_xy)
    hessian = _finite_difference_hessian(nemo, path_xy)
    forward = dxy / ds_xy[:, None]
    lateral = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)
    grad_mid = 0.5 * (grad[:-1] + grad[1:])
    hessian_mid = 0.5 * (hessian[:-1] + hessian[1:])
    cross_slope = torch.sum(grad_mid * lateral, dim=-1)
    roll = torch.atan(cross_slope)
    forward_terrain_curvature = torch.einsum("bi,bij,bj->b", forward, hessian_mid, forward)
    breakover_proxy = torch.abs(forward_terrain_curvature) * (float(vehicle.wheelbase_m) ** 2) / 8.0
    footprint_clearance_raw = _footprint_clearance_raw(nemo, path_xy, forward, lateral, vehicle)
    footprint_clearance_violation = torch.relu(footprint_clearance_raw)

    v_nom = float(vehicle.nominal_speed_mps)
    abs_curvature = torch.abs(curvature).clamp_min(1e-5)
    v_curve = torch.sqrt(float(vehicle.max_lateral_accel_mps2) / abs_curvature)
    v_yaw = float(vehicle.max_yaw_rate_radps) / abs_curvature
    v_slope = v_nom / (1.0 + float(vehicle.slope_speed_scale) * torch.abs(slope))
    speed = torch.minimum(torch.full_like(ds_3d, v_nom), torch.minimum(torch.minimum(v_curve, v_yaw), v_slope)).clamp_min(float(vehicle.min_speed_mps))
    dt = ds_3d / speed
    accel = torch.zeros_like(speed)
    if speed.numel() > 1:
        accel_mid_dt = (0.5 * (dt[1:] + dt[:-1])).clamp_min(1e-8)
        accel_inner = (speed[1:] - speed[:-1]) / accel_mid_dt
        accel = torch.cat([accel_inner[:1], accel_inner])
    yaw_rate = speed * curvature

    mass = float(vehicle.mass_kg)
    gravity = float(vehicle.gravity_mps2)
    drag_force = 0.5 * float(vehicle.air_density_kg_m3) * float(vehicle.drag_coefficient) * float(vehicle.frontal_area_m2) * speed.pow(2)
    rolling_force = mass * gravity * float(vehicle.rolling_resistance_coeff) * torch.cos(slope_angle)
    slope_force = mass * gravity * torch.sin(slope_angle)
    accel_force = mass * accel
    drive_force = drag_force + rolling_force + slope_force + accel_force
    power = torch.relu(drive_force * speed / max(float(vehicle.drivetrain_efficiency), 1e-8))
    energy = torch.sum(power * dt)

    return {
        "z": z,
        "distance_3d": torch.sum(ds_3d),
        "time": torch.sum(dt),
        "energy": energy,
        "slope_angle": slope_angle,
        "roll": roll,
        "curvature": curvature,
        "yaw_rate": yaw_rate,
        "accel": accel,
        "forward_terrain_curvature": forward_terrain_curvature,
        "breakover_proxy": breakover_proxy,
        "footprint_clearance_raw": footprint_clearance_raw,
        "footprint_clearance_violation": footprint_clearance_violation,
    }


def _footprint_clearance_raw(
    nemo: Nemo,
    path_xy: torch.Tensor,
    forward: torch.Tensor,
    lateral: torch.Tensor,
    vehicle: VehicleModelConfig,
) -> torch.Tensor:
    """Signed differentiable chassis clearance proxy from wheel plane and belly samples."""
    mid_xy = 0.5 * (path_xy[:-1] + path_xy[1:])
    num_segments = mid_xy.shape[0]
    if num_segments == 0:
        return path_xy.new_zeros((0,))

    wheelbase = float(vehicle.wheelbase_m)
    track = float(vehicle.track_width_m)
    clearance = float(vehicle.ground_clearance_m)
    wheel_local = path_xy.new_tensor(
        [
            [-0.5 * wheelbase, -0.5 * track],
            [-0.5 * wheelbase, 0.5 * track],
            [0.5 * wheelbase, -0.5 * track],
            [0.5 * wheelbase, 0.5 * track],
        ]
    )
    body_local = path_xy.new_tensor(
        [
            [-0.35 * wheelbase, -0.30 * track],
            [-0.35 * wheelbase, 0.0],
            [-0.35 * wheelbase, 0.30 * track],
            [0.0, -0.35 * track],
            [0.0, 0.0],
            [0.0, 0.35 * track],
            [0.35 * wheelbase, -0.30 * track],
            [0.35 * wheelbase, 0.0],
            [0.35 * wheelbase, 0.30 * track],
        ]
    )

    wheel_xy = _local_offsets_to_world(mid_xy, forward, lateral, wheel_local)
    wheel_z = nemo.field.h(wheel_xy.reshape(-1, 2)).reshape(num_segments, wheel_local.shape[0])
    rear_z = 0.5 * (wheel_z[:, 0] + wheel_z[:, 1])
    front_z = 0.5 * (wheel_z[:, 2] + wheel_z[:, 3])
    left_z = 0.5 * (wheel_z[:, 0] + wheel_z[:, 2])
    right_z = 0.5 * (wheel_z[:, 1] + wheel_z[:, 3])
    center_z = wheel_z.mean(dim=1)
    plane_du = (front_z - rear_z) / max(wheelbase, 1e-8)
    plane_dv = (right_z - left_z) / max(track, 1e-8)

    body_xy = _local_offsets_to_world(mid_xy, forward, lateral, body_local)
    body_z = nemo.field.h(body_xy.reshape(-1, 2)).reshape(num_segments, body_local.shape[0])
    support_z = (
        center_z[:, None]
        + plane_du[:, None] * body_local[None, :, 0]
        + plane_dv[:, None] * body_local[None, :, 1]
    )
    underside_z = support_z + clearance
    return body_z - underside_z


def _local_offsets_to_world(
    centers: torch.Tensor,
    forward: torch.Tensor,
    lateral: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return (
        centers[:, None, :]
        + offsets[None, :, 0:1] * forward[:, None, :]
        + offsets[None, :, 1:2] * lateral[:, None, :]
    )


def _build_astar_seed(
    nemo: Nemo,
    *,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    cfg: PhysicalObjectivePlannerConfig,
) -> np.ndarray:
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
    astar_rc = astar_grid_path(cost_grid, start_rc, goal_rc, step_weight=cfg.astar_step_weight, x_axis=xx[0, :], y_axis=yy[:, 0])
    path = np.column_stack([xx[astar_rc[:, 0], astar_rc[:, 1]], yy[astar_rc[:, 0], astar_rc[:, 1]]]).astype(np.float32)
    path[0] = np.asarray(start_xy, dtype=np.float32)
    path[-1] = np.asarray(goal_xy, dtype=np.float32)
    return path


def _finite_difference_grad(nemo: Nemo, xy: torch.Tensor) -> torch.Tensor:
    x_range = float(nemo.field.bounds[0][1] - nemo.field.bounds[0][0])
    y_range = float(nemo.field.bounds[1][1] - nemo.field.bounds[1][0])
    eps = max(min(x_range, y_range) / 512.0, 1e-3)
    dx = xy.new_tensor([eps, 0.0])
    dy = xy.new_tensor([0.0, eps])
    gx = (nemo.field.h(xy + dx).squeeze(-1) - nemo.field.h(xy - dx).squeeze(-1)) / (2.0 * eps)
    gy = (nemo.field.h(xy + dy).squeeze(-1) - nemo.field.h(xy - dy).squeeze(-1)) / (2.0 * eps)
    return torch.stack([gx, gy], dim=-1)


def _finite_difference_hessian(nemo: Nemo, xy: torch.Tensor) -> torch.Tensor:
    x_range = float(nemo.field.bounds[0][1] - nemo.field.bounds[0][0])
    y_range = float(nemo.field.bounds[1][1] - nemo.field.bounds[1][0])
    eps = max(min(x_range, y_range) / 512.0, 1e-3)
    dx = xy.new_tensor([eps, 0.0])
    dy = xy.new_tensor([0.0, eps])
    z = nemo.field.h(xy).squeeze(-1)
    z_xp = nemo.field.h(xy + dx).squeeze(-1)
    z_xm = nemo.field.h(xy - dx).squeeze(-1)
    z_yp = nemo.field.h(xy + dy).squeeze(-1)
    z_ym = nemo.field.h(xy - dy).squeeze(-1)
    z_xpyp = nemo.field.h(xy + dx + dy).squeeze(-1)
    z_xpym = nemo.field.h(xy + dx - dy).squeeze(-1)
    z_xmyp = nemo.field.h(xy - dx + dy).squeeze(-1)
    z_xmym = nemo.field.h(xy - dx - dy).squeeze(-1)
    inv_eps2 = 1.0 / (eps**2)
    hxx = (z_xp - 2.0 * z + z_xm) * inv_eps2
    hyy = (z_yp - 2.0 * z + z_ym) * inv_eps2
    hxy = (z_xpyp - z_xpym - z_xmyp + z_xmym) * (0.25 * inv_eps2)
    return torch.stack(
        [
            torch.stack([hxx, hxy], dim=-1),
            torch.stack([hxy, hyy], dim=-1),
        ],
        dim=-2,
    )


def _unwrap_torch(angle: torch.Tensor) -> torch.Tensor:
    if angle.numel() <= 1:
        return angle
    diffs = angle[1:] - angle[:-1]
    diffs = torch.atan2(torch.sin(diffs), torch.cos(diffs))
    return torch.cat([angle[:1], angle[:1] + torch.cumsum(diffs, dim=0)])


def _diagnostics_from_terms(terms: dict[str, torch.Tensor], objective: ObjectiveName) -> PhysicalPlanDiagnostics:
    def scalar(name: str) -> float:
        return float(terms[name].detach().cpu().item())

    breakover_margin = terms.get("breakover_clearance_margin_m")
    if breakover_margin is None:
        max_breakover_violation = 0.0
    else:
        max_breakover_violation = float(
            torch.max(torch.relu(terms["breakover_proxy"] - breakover_margin)).detach().cpu().item()
        )
    footprint_violation = terms.get("footprint_clearance_violation")
    if footprint_violation is None or footprint_violation.numel() == 0:
        max_footprint_violation = 0.0
        mean_footprint_violation = 0.0
    else:
        max_footprint_violation = float(torch.max(footprint_violation).detach().cpu().item())
        mean_footprint_violation = float(torch.mean(footprint_violation).detach().cpu().item())

    return PhysicalPlanDiagnostics(
        objective=float(scalar("primary")),
        primary_cost=float(scalar("primary")),
        total_cost=float(scalar("total")),
        distance_3d_m=float(scalar("distance_3d")),
        time_s=float(scalar("time")),
        energy_j=float(scalar("energy")),
        max_slope_deg=float(torch.rad2deg(torch.max(torch.abs(terms["slope_angle"]))).detach().cpu().item()),
        max_roll_deg=float(torch.rad2deg(torch.max(torch.abs(terms["roll"]))).detach().cpu().item()),
        max_curvature_1pm=float(torch.max(torch.abs(terms["curvature"])).detach().cpu().item()),
        max_yaw_rate_radps=float(torch.max(torch.abs(terms["yaw_rate"])).detach().cpu().item()),
        max_accel_mps2=float(torch.max(torch.abs(terms["accel"])).detach().cpu().item()),
        max_forward_terrain_curvature_1pm=float(
            torch.max(torch.abs(terms["forward_terrain_curvature"])).detach().cpu().item()
        ),
        max_breakover_proxy_m=float(torch.max(terms["breakover_proxy"]).detach().cpu().item()),
        max_breakover_violation_m=max_breakover_violation,
        max_footprint_clearance_violation_m=max_footprint_violation,
        mean_footprint_clearance_violation_m=mean_footprint_violation,
    )
