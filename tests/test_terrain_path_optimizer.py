from __future__ import annotations

import numpy as np

from nemo.terrain_dynamics import DEMTerrainModel
from nemo.terrain_path_optimizer import (
    PathOptimizationConfig,
    TerrainAwarePathOptimizer,
    compute_energy_time_diagnostics,
    nearest_path_distance,
    planar_curvature,
    relu_squared_violation,
    unwrap_angle,
)


def _flat_terrain() -> DEMTerrainModel:
    axis = np.linspace(-5.0, 5.0, 51)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return DEMTerrainModel(xx, yy, np.zeros_like(xx))


def test_optimizer_initializes_fixed_number_of_control_points() -> None:
    optimizer = TerrainAwarePathOptimizer(
        _flat_terrain(),
        PathOptimizationConfig(num_control_points=4),
    )
    waypoints = np.array([[-4.0, 0.0], [-1.0, 1.0], [2.0, -1.0], [4.0, 0.0]])

    controls = optimizer.initialize_control_points(waypoints)
    reconstructed = optimizer.reconstruct_waypoints_from_control_points(
        waypoints[0],
        controls,
        waypoints[-1],
    )

    assert controls.shape == (4, 2)
    assert np.allclose(reconstructed[0], waypoints[0])
    assert np.allclose(reconstructed[-1], waypoints[-1])


def test_helper_functions() -> None:
    line = np.column_stack([np.linspace(0.0, 1.0, 5), np.zeros(5)])
    assert np.allclose(planar_curvature(line), 0.0)
    assert np.allclose(unwrap_angle(np.array([3.0, -3.0]))[1] > 3.0, True)
    assert np.isclose(relu_squared_violation(np.array([0.0, 2.0, 3.0]), 1.0), (1.0 + 4.0) / 3.0)
    points = np.array([[0.5, 1.0], [2.0, 0.0]])
    reference = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(nearest_path_distance(points, reference), [1.0, 1.0])


def test_optimizer_reduces_cost_on_zigzag_flat_terrain() -> None:
    terrain = _flat_terrain()
    waypoints = np.array(
        [
            [-4.0, 0.0],
            [-2.5, 2.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [2.5, -2.0],
            [4.0, 0.0],
        ],
        dtype=np.float64,
    )
    cfg = PathOptimizationConfig(
        num_control_points=4,
        num_iters=25,
        sample_spacing=0.5,
        w_length=1.0,
        w_curvature=20.0,
        w_yaw_rate=0.0,
        w_accel=0.0,
        w_anchor=0.01,
        verbose=False,
    )

    result = TerrainAwarePathOptimizer(terrain, cfg).optimize(waypoints)

    assert result.optimized_trajectory.states.shape[1] == 8
    assert result.optimized_cost <= result.initial_cost
    assert np.allclose(result.optimized_trajectory.positions[0, :2], waypoints[0])
    assert np.allclose(result.optimized_trajectory.positions[-1, :2], waypoints[-1])
    assert len(result.cost_history) > 0


def test_energy_time_flat_path_scales_with_length() -> None:
    terrain = _flat_terrain()
    short = generate_flat_trajectory(terrain, np.array([[0.0, 0.0], [1.0, 0.0]]))
    long = generate_flat_trajectory(terrain, np.array([[0.0, 0.0], [2.0, 0.0]]))

    short_diag = compute_energy_time_diagnostics(
        short,
        mass_kg=10.0,
        rolling_resistance_coeff=0.05,
        drivetrain_efficiency=0.8,
        nominal_speed_mps=0.2,
    )
    long_diag = compute_energy_time_diagnostics(
        long,
        mass_kg=10.0,
        rolling_resistance_coeff=0.05,
        drivetrain_efficiency=0.8,
        nominal_speed_mps=0.2,
    )

    assert long_diag.time_seconds > short_diag.time_seconds
    assert long_diag.energy_joules > short_diag.energy_joules


def test_energy_time_uphill_more_energy_than_flat_same_length() -> None:
    axis = np.linspace(0.0, 2.0, 41)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    flat = DEMTerrainModel(xx, yy, np.zeros_like(xx))
    uphill = DEMTerrainModel(xx, yy, 0.2 * xx)
    waypoints = np.array([[0.0, 1.0], [2.0, 1.0]])

    flat_diag = compute_energy_time_diagnostics(
        generate_flat_trajectory(flat, waypoints),
        mass_kg=10.0,
        rolling_resistance_coeff=0.05,
        drivetrain_efficiency=0.8,
        nominal_speed_mps=0.2,
    )
    uphill_diag = compute_energy_time_diagnostics(
        generate_flat_trajectory(uphill, waypoints),
        mass_kg=10.0,
        rolling_resistance_coeff=0.05,
        drivetrain_efficiency=0.8,
        nominal_speed_mps=0.2,
    )

    assert uphill_diag.energy_joules > flat_diag.energy_joules


def test_downhill_energy_nonnegative_without_regen() -> None:
    axis = np.linspace(0.0, 2.0, 41)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    downhill = DEMTerrainModel(xx, yy, -0.5 * xx)
    trajectory = generate_flat_trajectory(downhill, np.array([[0.0, 1.0], [2.0, 1.0]]))

    diagnostics = compute_energy_time_diagnostics(
        trajectory,
        mass_kg=10.0,
        rolling_resistance_coeff=0.05,
        drivetrain_efficiency=0.8,
        nominal_speed_mps=0.2,
        allow_regen=False,
    )

    assert diagnostics.energy_joules >= 0.0


def generate_flat_trajectory(terrain: DEMTerrainModel, waypoints: np.ndarray):
    from nemo.terrain_dynamics import generate_terrain_aware_trajectory

    return generate_terrain_aware_trajectory(
        waypoints,
        terrain,
        nominal_speed=0.2,
        sample_spacing=0.25,
        smooth_path=False,
        compute_controls=True,
    )
