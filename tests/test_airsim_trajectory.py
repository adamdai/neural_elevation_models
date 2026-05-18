from __future__ import annotations

import numpy as np

from nemo.airsim_trajectory import generate_airsim_reference_trajectory
from nemo.physical_objective_planner import VehicleModelConfig


def test_generate_airsim_reference_trajectory_fields_and_shapes() -> None:
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, -1.0],
            [20.0, 10.0, -2.0],
        ],
        dtype=np.float32,
    )
    trajectory = generate_airsim_reference_trajectory(
        path,
        vehicle=VehicleModelConfig(nominal_speed_mps=4.0, max_lateral_accel_mps2=2.0),
        sample_spacing_m=2.0,
    )

    n = trajectory["positions"].shape[0]
    assert n > path.shape[0]
    for key in ("yaw", "speed", "yaw_rate", "curvature", "accel", "pitch", "roll", "t", "arc_length"):
        assert trajectory[key].shape == (n,)
    assert trajectory["states"].shape == (n, 8)
    assert trajectory["controls"].shape == (n, 3)
    assert np.all(np.diff(trajectory["t"]) >= 0.0)
    assert np.all(trajectory["speed"] <= 4.0 + 1e-6)
    assert np.allclose(trajectory["positions"][0], path[0])
    assert np.allclose(trajectory["positions"][-1], path[-1])
