from __future__ import annotations

import numpy as np
import torch

from nemo import HeightField, Nemo
from nemo.path_planning import PathPlanningConfig, buffered_corner_points, plan_path


class RidgeField(HeightField):
    def __init__(self) -> None:
        super().__init__(((-1.0, 1.0), (-1.0, 1.0)))

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, :1]
        y = xy[:, 1:2]
        ridge = 1.8 * torch.exp(-((x / 0.18) ** 2 + (y / 0.45) ** 2))
        bowl = 0.15 * (x**2 + 0.5 * y**2)
        return ridge + bowl


def test_buffered_corner_points_returns_interior_corners() -> None:
    start, goal = buffered_corner_points(((-2.0, 2.0), (-4.0, 4.0)), buffer_fraction=0.2)
    assert start == (-1.2, -2.4)
    assert goal == (1.2, 2.4)


def test_plan_path_uses_astar_and_improves_objective() -> None:
    nemo = Nemo(RidgeField())
    cfg = PathPlanningConfig(
        astar_grid_resolution_x=40,
        astar_grid_resolution_y=40,
        num_waypoints=24,
        optimize_iterations=80,
        optimize_lr=3e-2,
        terrain_height_weight=1.0,
        terrain_slope_weight=0.3,
        flatness_weight=0.5,
        smoothness_weight=0.08,
        length_weight=0.02,
        batch_size=2048,
    )

    result = plan_path(nemo, config=cfg)

    assert result.initial_path_xy.shape == (24, 2)
    assert result.optimized_path_xy.shape == (24, 2)
    assert np.allclose(result.initial_path_xy[0], result.start_xy)
    assert np.allclose(result.initial_path_xy[-1], result.goal_xy)
    assert np.allclose(result.optimized_path_xy[0], result.start_xy)
    assert np.allclose(result.optimized_path_xy[-1], result.goal_xy)
    assert result.final_objective < result.initial_objective
    assert len(result.cost_history) == cfg.optimize_iterations
    assert result.cost_history[-1] <= result.cost_history[0]

