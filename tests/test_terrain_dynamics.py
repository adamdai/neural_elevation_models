from __future__ import annotations

import numpy as np
import torch

from nemo.terrain_dynamics import (
    DEMTerrainModel,
    NEMOTerrainModel,
    compute_attitude_from_gradient,
    generate_terrain_aware_trajectory,
)


def test_dem_terrain_model_queries_plane() -> None:
    x_axis = np.linspace(-2.0, 2.0, 17)
    y_axis = np.linspace(-3.0, 3.0, 19)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    zz = 2.0 * xx - 0.5 * yy + 1.0

    terrain = DEMTerrainModel(xx, yy, zz)
    x = np.array([-1.3, 0.2, 1.4])
    y = np.array([-2.0, 0.5, 2.3])

    assert np.allclose(terrain.height(x, y), 2.0 * x - 0.5 * y + 1.0)
    assert np.allclose(terrain.gradient(x, y), np.tile([2.0, -0.5], (3, 1)))
    assert np.allclose(terrain.normal(x, y), np.tile([-2.0, 0.5, 1.0], (3, 1)) / np.sqrt(5.25))


def test_compute_attitude_from_gradient_body_frame() -> None:
    gradient = np.array([[0.1, 0.2]])
    yaw = np.array([0.0])

    pitch, roll, along, cross = compute_attitude_from_gradient(gradient, yaw)

    assert np.allclose(along, [0.1])
    assert np.allclose(cross, [0.2])
    assert np.allclose(pitch, np.arctan([0.1]))
    assert np.allclose(roll, np.arctan([0.2]))


def test_generate_terrain_aware_trajectory_shapes_and_controls() -> None:
    x_axis = np.linspace(-2.0, 2.0, 41)
    y_axis = np.linspace(-2.0, 2.0, 41)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    zz = 0.1 * xx + 0.05 * yy
    terrain = DEMTerrainModel(xx, yy, zz)
    waypoints = np.array([[-1.5, -1.0], [0.0, 0.2], [1.5, 1.0]])

    traj = generate_terrain_aware_trajectory(
        waypoints,
        terrain,
        nominal_speed=0.2,
        sample_spacing=0.25,
        smooth_path=False,
        rover_mass=10.0,
    )

    assert traj.positions.shape[1] == 3
    assert traj.states.shape == (traj.t.shape[0], 8)
    assert traj.controls["v_cmd"].shape == traj.t.shape
    assert traj.controls["yaw_rate_cmd"].shape == traj.t.shape
    assert traj.controls["accel_cmd"].shape == traj.t.shape
    assert traj.controls["force_cmd"].shape == traj.t.shape
    assert np.all(np.diff(traj.t) >= 0.0)
    assert np.allclose(traj.states[:, :3], traj.positions)


class TorchParaboloid(torch.nn.Module):
    def h(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, :1]
        y = xy[:, 1:2]
        return x**2 + 3.0 * x * y + 2.0 * y**2


def test_nemo_terrain_model_autograd_gradient_and_hessian() -> None:
    terrain = NEMOTerrainModel(TorchParaboloid(), use_autograd=True)
    x = np.array([1.0, -0.5])
    y = np.array([2.0, 0.25])

    grad = terrain.gradient(x, y)
    hess = terrain.hessian(x, y)

    assert np.allclose(grad[:, 0], 2.0 * x + 3.0 * y)
    assert np.allclose(grad[:, 1], 3.0 * x + 4.0 * y)
    assert np.allclose(hess, np.tile([[2.0, 3.0], [3.0, 4.0]], (2, 1, 1)))
