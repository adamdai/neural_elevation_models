"""Estimate AirSim path traversal energy from a saved 3D waypoint path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nemo.terrain_dynamics import (
    Trajectory,
    compute_yaw_from_path,
    _time_derivative,
    _time_from_path_lengths,
)


MASS = 2000.0
G = 9.81
CD = 0.3
AREA = 2.5
FRR = 0.015
RHO = 1.225
EFFICIENCY = 0.8
TARGET_SPEED = 6.0


def estimate_power(speed: np.ndarray, accel: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Same SUV power model used by the AirSim tracker."""
    f_accel = MASS * accel
    f_drag = 0.5 * RHO * CD * AREA * (speed**2)
    f_rolling = MASS * G * FRR * np.cos(pitch)
    f_gravity = MASS * G * np.sin(pitch)
    f_total = f_accel + f_drag + f_rolling + f_gravity
    return np.maximum(0.0, f_total * speed / EFFICIENCY)


def trajectory_from_airsim_path(path_xyz: np.ndarray, nominal_speed: float) -> Trajectory:
    """Build a nominal terrain-aware trajectory directly from an AirSim XYZ path."""
    positions = np.asarray(path_xyz, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("Expected path_xyz with shape (N, 3), N >= 2.")

    xy = positions[:, :2]
    dxyz = np.diff(positions, axis=0)
    dxy = dxyz[:, :2]
    ds_xy = np.linalg.norm(dxy, axis=1)
    ds_3d = np.linalg.norm(dxyz, axis=1)
    dz = dxyz[:, 2]
    valid = ds_xy > 1e-12

    segment_slope = np.zeros_like(ds_xy)
    segment_slope[valid] = dz[valid] / ds_xy[valid]
    point_slope = np.empty(positions.shape[0], dtype=np.float64)
    point_slope[0] = segment_slope[0]
    point_slope[-1] = segment_slope[-1]
    if positions.shape[0] > 2:
        point_slope[1:-1] = 0.5 * (segment_slope[:-1] + segment_slope[1:])

    yaw = compute_yaw_from_path(xy)
    pitch = np.arctan(point_slope)
    roll = np.zeros_like(pitch)
    speed = np.full(positions.shape[0], float(nominal_speed), dtype=np.float64)
    t = _time_from_path_lengths(ds_3d, speed)
    yaw_rate = _time_derivative(np.unwrap(yaw), t)
    accel = _time_derivative(speed, t)

    forward = np.column_stack([np.cos(yaw), np.sin(yaw)])
    gradient = point_slope[:, None] * forward
    normals = np.column_stack([-gradient, np.ones(positions.shape[0])])
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

    states = np.column_stack([positions, yaw, pitch, roll, speed, yaw_rate])
    controls = {
        "v_cmd": speed.copy(),
        "yaw_rate_cmd": yaw_rate.copy(),
        "accel_cmd": accel.copy(),
    }
    terrain_info = {
        "height": positions[:, 2].copy(),
        "gradient": gradient,
        "hessian": np.zeros((positions.shape[0], 2, 2), dtype=np.float64),
        "normal": normals,
        "slope_magnitude": np.abs(point_slope),
        "along_track_slope": point_slope,
        "cross_track_slope": np.zeros_like(point_slope),
        "arc_length": np.concatenate([[0.0], np.cumsum(ds_xy)]),
        "segment_length_3d": ds_3d,
    }
    return Trajectory(
        t=t,
        positions=positions,
        yaws=yaw,
        pitches=pitch,
        rolls=roll,
        velocities=speed,
        yaw_rates=yaw_rate,
        states=states,
        controls=controls,
        terrain_info=terrain_info,
    )


def estimate_energy(trajectory: Trajectory) -> dict[str, float]:
    positions = trajectory.positions
    ds_3d = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = np.diff(trajectory.t)
    speed = 0.5 * (trajectory.velocities[:-1] + trajectory.velocities[1:])
    accel = 0.5 * (trajectory.controls["accel_cmd"][:-1] + trajectory.controls["accel_cmd"][1:])
    pitch = 0.5 * (trajectory.pitches[:-1] + trajectory.pitches[1:])
    power = estimate_power(speed, accel, pitch)
    energy = power * dt
    horizontal_distance = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
    distance_3d = float(ds_3d.sum())
    return {
        "energy_j": float(energy.sum()),
        "energy_kj": float(energy.sum() / 1000.0),
        "energy_wh": float(energy.sum() / 3600.0),
        "time_s": float(trajectory.t[-1] - trajectory.t[0]),
        "distance_horizontal_m": horizontal_distance,
        "distance_3d_m": distance_3d,
        "mean_power_w": float(energy.sum() / max(trajectory.t[-1] - trajectory.t[0], 1e-12)),
        "max_power_w": float(power.max(initial=0.0)),
        "mean_pitch_deg": float(np.degrees(np.mean(trajectory.pitches))),
        "min_pitch_deg": float(np.degrees(np.min(trajectory.pitches))),
        "max_pitch_deg": float(np.degrees(np.max(trajectory.pitches))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/path_planning/airsim_path_safe.npy"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-states", type=Path, default=None)
    parser.add_argument("--output-controls", type=Path, default=None)
    parser.add_argument("--speed", type=float, default=TARGET_SPEED)
    args = parser.parse_args()

    path = np.load(args.path).astype(np.float64)
    trajectory = trajectory_from_airsim_path(path, nominal_speed=float(args.speed))
    summary = estimate_energy(trajectory)
    summary["path"] = str(args.path)
    summary["target_speed_mps"] = float(args.speed)
    summary["mass_kg"] = MASS
    summary["drag_coefficient"] = CD
    summary["frontal_area_m2"] = AREA
    summary["rolling_resistance_coeff"] = FRR
    summary["air_density_kg_m3"] = RHO
    summary["efficiency"] = EFFICIENCY

    output_json = args.output_json or args.path.with_name(f"{args.path.stem}_energy.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.output_states is not None:
        args.output_states.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_states, trajectory.states)
    if args.output_controls is not None:
        args.output_controls.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_controls, **trajectory.controls)

    print(json.dumps(summary, indent=2))
    print(f"Saved energy summary to {output_json}")


if __name__ == "__main__":
    main()
