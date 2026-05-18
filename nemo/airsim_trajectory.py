from __future__ import annotations

from dataclasses import asdict

import numpy as np

from nemo.physical_objective_planner import VehicleModelConfig


def generate_airsim_reference_trajectory(
    path_xyz: np.ndarray,
    *,
    vehicle: VehicleModelConfig | None = None,
    sample_spacing_m: float = 2.0,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Generate a dense AirSim-frame reference trajectory for path tracking.

    Input and output positions use the AirSim frame. The returned dictionary is
    intended to be saved with ``np.savez`` and consumed by a feedforward +
    feedback tracker.
    """
    vehicle = vehicle or VehicleModelConfig()
    positions = resample_path_xyz(path_xyz, sample_spacing_m=sample_spacing_m)
    xy = positions[:, :2]
    if positions.shape[0] < 2:
        raise ValueError("Reference trajectory requires at least two path points.")

    segment_xy = np.diff(xy, axis=0)
    segment_xyz = np.diff(positions, axis=0)
    ds_xy = np.linalg.norm(segment_xy, axis=1)
    ds_3d = np.linalg.norm(segment_xyz, axis=1)
    ds_xy = np.clip(ds_xy, 1e-8, None)
    ds_3d = np.clip(ds_3d, 1e-8, None)

    yaw = compute_yaw(xy)
    curvature = compute_curvature(xy)
    pitch = compute_pitch_from_path(positions)
    roll = np.zeros_like(pitch)
    speed = speed_profile_from_geometry(
        positions,
        curvature=curvature,
        vehicle=vehicle,
    )
    t = time_from_segments(ds_3d, speed)
    yaw_rate = speed * curvature
    accel = derivative(speed, t)

    states = np.column_stack([positions, yaw, pitch, roll, speed, yaw_rate])
    controls = np.column_stack([speed, yaw_rate, accel])
    arc_length = np.concatenate([[0.0], np.cumsum(ds_3d)])
    return {
        "positions": positions.astype(np.float32),
        "yaw": yaw.astype(np.float32),
        "speed": speed.astype(np.float32),
        "yaw_rate": yaw_rate.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "accel": accel.astype(np.float32),
        "pitch": pitch.astype(np.float32),
        "roll": roll.astype(np.float32),
        "t": t.astype(np.float32),
        "arc_length": arc_length.astype(np.float32),
        "states": states.astype(np.float32),
        "controls": controls.astype(np.float32),
        "control_columns": np.array(["v_cmd", "yaw_rate_cmd", "accel_cmd"]),
        "state_columns": np.array(["x", "y", "z", "yaw", "pitch", "roll", "v", "yaw_rate"]),
        "vehicle_config": asdict(vehicle),
    }


def save_airsim_reference_trajectory(
    output_path: str,
    path_xyz: np.ndarray,
    *,
    vehicle: VehicleModelConfig | None = None,
    sample_spacing_m: float = 2.0,
) -> dict[str, np.ndarray | dict[str, float]]:
    trajectory = generate_airsim_reference_trajectory(
        path_xyz,
        vehicle=vehicle,
        sample_spacing_m=sample_spacing_m,
    )
    np.savez(output_path, **trajectory)
    return trajectory


def resample_path_xyz(path_xyz: np.ndarray, *, sample_spacing_m: float) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("Expected path_xyz with shape (N, 3).")
    if path.shape[0] < 2:
        raise ValueError("At least two path points are required.")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-8:
        return path[:1].copy()

    spacing = max(float(sample_spacing_m), 1e-3)
    count = max(int(np.ceil(total / spacing)) + 1, 2)
    target = np.linspace(0.0, total, count)
    out = np.column_stack([np.interp(target, cumulative, path[:, i]) for i in range(3)])
    out[0] = path[0]
    out[-1] = path[-1]
    return out


def compute_yaw(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    dx = np.gradient(xy[:, 0], edge_order=1)
    dy = np.gradient(xy[:, 1], edge_order=1)
    return np.unwrap(np.arctan2(dy, dx))


def compute_curvature(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 3:
        return np.zeros(xy.shape[0], dtype=np.float64)
    ds = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if ds[-1] <= 1e-8:
        return np.zeros(xy.shape[0], dtype=np.float64)
    dx = np.gradient(xy[:, 0], ds, edge_order=1)
    dy = np.gradient(xy[:, 1], ds, edge_order=1)
    ddx = np.gradient(dx, ds, edge_order=1)
    ddy = np.gradient(dy, ds, edge_order=1)
    denom = np.clip((dx**2 + dy**2) ** 1.5, 1e-8, None)
    return (dx * ddy - dy * ddx) / denom


def compute_pitch_from_path(path_xyz: np.ndarray) -> np.ndarray:
    positions = np.asarray(path_xyz, dtype=np.float64)
    dxy = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    dz = np.diff(positions[:, 2])
    segment_pitch = np.arctan2(dz, np.clip(dxy, 1e-8, None))
    pitch = np.zeros(positions.shape[0], dtype=np.float64)
    pitch[0] = segment_pitch[0]
    pitch[-1] = segment_pitch[-1]
    if positions.shape[0] > 2:
        pitch[1:-1] = 0.5 * (segment_pitch[:-1] + segment_pitch[1:])
    return pitch


def speed_profile_from_geometry(
    path_xyz: np.ndarray,
    *,
    curvature: np.ndarray,
    vehicle: VehicleModelConfig,
) -> np.ndarray:
    pitch = compute_pitch_from_path(path_xyz)
    nominal = float(vehicle.nominal_speed_mps)
    curvature = np.asarray(curvature, dtype=np.float64)
    abs_curvature = np.clip(np.abs(curvature), 1e-5, None)
    v_curve = np.sqrt(float(vehicle.max_lateral_accel_mps2) / abs_curvature)
    v_yaw = float(vehicle.max_yaw_rate_radps) / abs_curvature
    v_slope = nominal / (1.0 + float(vehicle.slope_speed_scale) * np.abs(np.tan(pitch)))
    speed = np.minimum(nominal, np.minimum(np.minimum(v_curve, v_yaw), v_slope))
    speed = np.clip(speed, float(vehicle.min_speed_mps), nominal)
    ds = np.linalg.norm(np.diff(path_xyz, axis=0), axis=1)
    max_accel = max(float(vehicle.max_longitudinal_accel_mps2), 1e-8)
    for i in range(1, speed.shape[0]):
        speed[i] = min(speed[i], np.sqrt(speed[i - 1] ** 2 + 2.0 * max_accel * max(float(ds[i - 1]), 1e-8)))
    for i in range(speed.shape[0] - 2, -1, -1):
        speed[i] = min(speed[i], np.sqrt(speed[i + 1] ** 2 + 2.0 * max_accel * max(float(ds[i]), 1e-8)))
    return np.clip(speed, float(vehicle.min_speed_mps), nominal)


def time_from_segments(segment_lengths: np.ndarray, speed: np.ndarray) -> np.ndarray:
    segment_speed = 0.5 * (speed[:-1] + speed[1:])
    dt = segment_lengths / np.clip(segment_speed, 1e-8, None)
    return np.concatenate([[0.0], np.cumsum(dt)])


def derivative(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2 or np.allclose(t, t[0]):
        return np.zeros_like(values)
    return np.gradient(values, t, edge_order=1)
