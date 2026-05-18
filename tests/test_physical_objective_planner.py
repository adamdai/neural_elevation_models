from __future__ import annotations

import numpy as np
import torch

from nemo import HeightField, Nemo
from nemo.basic_terrain_planner import BasicTerrainPlannerConfig, optimize_basic_terrain_path
from nemo.physical_objective_planner import (
    PhysicalObjectivePlannerConfig,
    SafetyConstraintConfig,
    VehicleModelConfig,
    plan_physical_objective_path,
)


class RollingField(HeightField):
    def __init__(self) -> None:
        super().__init__(((-1.0, 1.0), (-1.0, 1.0)))

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, :1]
        y = xy[:, 1:2]
        return 0.15 * torch.sin(3.0 * x) + 0.1 * torch.cos(2.0 * y)


class GaussianHillField(HeightField):
    def __init__(self) -> None:
        super().__init__(((-2.0, 2.0), (-1.5, 1.5)))

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        r2 = xy[:, :1].pow(2) + xy[:, 1:2].pow(2)
        return 2.0 * torch.exp(-0.5 * r2 / (0.33**2))


def test_physical_objective_planner_preserves_endpoints_and_improves_cost() -> None:
    nemo = Nemo(RollingField())
    cfg = PhysicalObjectivePlannerConfig(
        objective="energy",
        astar_grid_resolution_x=32,
        astar_grid_resolution_y=32,
        num_waypoints=18,
        optimize_iterations=20,
        optimize_lr=1e-2,
        vehicle=VehicleModelConfig(nominal_speed_mps=2.0),
        safety=SafetyConstraintConfig(curvature_weight=1e6),
        batch_size=2048,
    )

    result = plan_physical_objective_path(nemo, start_xy=(-0.8, -0.8), goal_xy=(0.8, 0.8), config=cfg)

    assert result.optimized_path_xy.shape == (18, 2)
    assert np.allclose(result.optimized_path_xy[0], (-0.8, -0.8))
    assert np.allclose(result.optimized_path_xy[-1], (0.8, 0.8))
    assert result.optimized_diagnostics.total_cost <= result.initial_diagnostics.total_cost
    assert len(result.cost_history) == cfg.optimize_iterations


def test_physical_objective_planner_supports_all_primary_objectives() -> None:
    nemo = Nemo(RollingField())
    for objective in ("distance", "time", "energy"):
        cfg = PhysicalObjectivePlannerConfig(
            objective=objective,
            astar_grid_resolution_x=24,
            astar_grid_resolution_y=24,
            num_waypoints=12,
            optimize_iterations=2,
            vehicle=VehicleModelConfig(nominal_speed_mps=2.0),
            batch_size=1024,
        )
        result = plan_physical_objective_path(nemo, start_xy=(-0.8, -0.8), goal_xy=(0.8, 0.8), config=cfg)
        assert result.objective == objective
        assert np.isfinite(result.optimized_diagnostics.distance_3d_m)
        assert np.isfinite(result.optimized_diagnostics.time_s)
        assert np.isfinite(result.optimized_diagnostics.energy_j)


def test_basic_terrain_optimizer_routes_smoothly_around_synthetic_hill() -> None:
    nemo = Nemo(GaussianHillField())
    num_points = 70
    x = np.linspace(-1.65, 1.65, num_points, dtype=np.float32)
    y = (0.06 * np.sin(np.linspace(0.0, np.pi, num_points))).astype(np.float32)
    initial_path = np.column_stack([x, y]).astype(np.float32)
    initial_path[0, 1] = 0.0
    initial_path[-1, 1] = 0.0

    cfg = BasicTerrainPlannerConfig(
        num_control_points=8,
        num_samples=220,
        num_iters=1200,
        lr=2.0e-2,
        w_length=5.0,
        w_slope=1000.0,
        w_control=3.5,
        w_control_points=1.5,
    )
    result = optimize_basic_terrain_path(nemo, initial_path, cfg)

    with torch.no_grad():
        initial_height = nemo.h(torch.as_tensor(initial_path, dtype=torch.float32)).numpy().reshape(-1)
        optimized_height = nemo.h(torch.as_tensor(result.optimized_path_xy, dtype=torch.float32)).numpy().reshape(-1)

    assert result.optimized_diagnostics.total_cost < result.initial_diagnostics.total_cost
    assert float(np.max(optimized_height)) < 0.5 * float(np.max(initial_height))
    assert float(np.max(np.abs(result.optimized_path_xy[:, 1]))) > 0.45
    assert result.optimized_diagnostics.control_effort < 20.0
    assert np.allclose(result.optimized_path_xy[0], initial_path[0])
    assert np.allclose(result.optimized_path_xy[-1], initial_path[-1])
