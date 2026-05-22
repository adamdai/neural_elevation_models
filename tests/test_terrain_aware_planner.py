from __future__ import annotations

import numpy as np

from nemo.terrain_aware_planner import (
    TerrainAwarePlannerConfig,
    _FakeNemo,
    optimize_terrain_aware_path,
    terrain_grad,
)


def test_terrain_aware_planner_reduces_slope_and_throttle_on_hill() -> None:
    nemo = _FakeNemo()
    x = np.linspace(-1.65, 1.65, 80, dtype=np.float32)
    y = (0.15 * np.sin(np.linspace(0.0, np.pi, x.size))).astype(np.float32)
    path = np.column_stack([x, y]).astype(np.float32)
    path[0, 1] = 0.0
    path[-1, 1] = 0.0
    cfg = TerrainAwarePlannerConfig(
        num_control_points=12,
        num_samples=180,
        num_iters=400,
        lr=3e-2,
        w_length=0.02,
        w_uphill=0.80,
        w_downhill=0.80,
        w_crosstrack=0.10,
        w_throttle=0.35,
        w_steering=0.01,
    )

    result = optimize_terrain_aware_path(nemo, path, cfg)

    assert result.optimized_diagnostics.total_cost < result.initial_diagnostics.total_cost
    assert result.optimized_diagnostics.uphill_cost < result.initial_diagnostics.uphill_cost
    assert result.optimized_diagnostics.throttle_cost < result.initial_diagnostics.throttle_cost
    assert float(np.max(np.abs(result.optimized_path_xy[:, 1]))) > 0.5
    assert np.allclose(result.optimized_path_xy[0], path[0])
    assert np.allclose(result.optimized_path_xy[-1], path[-1])
    assert set(result.trajectory) == {
        "x",
        "y",
        "z",
        "yaw",
        "curvature",
        "slope_mag",
        "along_track_slope",
        "cross_track_slope",
        "along_track_slope_rad",
        "cross_track_slope_rad",
        "speed",
        "steering",
        "throttle",
    }


def test_terrain_aware_planner_autograd_gradient_matches_gaussian() -> None:
    nemo = _FakeNemo()
    xy = np.asarray([[-0.25, 0.1], [0.4, -0.2]], dtype=np.float32)
    xy_t = nemo.field._dummy.new_tensor(xy).requires_grad_(True)
    cfg = TerrainAwarePlannerConfig(gradient_mode="autograd")

    grad = terrain_grad(nemo, xy_t, cfg).detach().cpu().numpy()

    sigma2 = 0.33**2
    z = 2.0 * np.exp(-0.5 * np.sum(xy * xy, axis=1) / sigma2)
    expected = -(xy / sigma2) * z[:, None]
    assert np.allclose(grad, expected, rtol=1e-5, atol=1e-5)


def test_terrain_aware_planner_autograd_mode_optimizes_path() -> None:
    nemo = _FakeNemo()
    x = np.linspace(-1.65, 1.65, 80, dtype=np.float32)
    y = (0.15 * np.sin(np.linspace(0.0, np.pi, x.size))).astype(np.float32)
    path = np.column_stack([x, y]).astype(np.float32)
    path[0, 1] = 0.0
    path[-1, 1] = 0.0
    cfg = TerrainAwarePlannerConfig(
        num_control_points=12,
        num_samples=180,
        num_iters=120,
        lr=3e-2,
        w_length=0.02,
        w_uphill=0.80,
        w_downhill=0.80,
        w_crosstrack=0.10,
        w_throttle=0.35,
        w_steering=0.01,
        gradient_mode="autograd",
    )

    result = optimize_terrain_aware_path(nemo, path, cfg)

    assert result.optimized_diagnostics.total_cost < result.initial_diagnostics.total_cost
    assert result.optimized_diagnostics.throttle_cost < result.initial_diagnostics.throttle_cost
