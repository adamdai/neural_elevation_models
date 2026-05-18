"""Export a dense AirSim reference trajectory NPZ from a saved AirSim path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nemo.airsim_trajectory import generate_airsim_reference_trajectory
from nemo.physical_objective_planner import VehicleModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Input AirSim path .npy with shape (N, 3).")
    parser.add_argument("--output", type=Path, default=None, help="Output reference trajectory .npz.")
    parser.add_argument("--sample-spacing", type=float, default=2.0)
    parser.add_argument("--nominal-speed", type=float, default=6.0)
    parser.add_argument("--max-lateral-accel", type=float, default=2.0)
    parser.add_argument("--max-longitudinal-accel", type=float, default=1.5)
    parser.add_argument("--max-yaw-rate", type=float, default=0.6)
    parser.add_argument("--min-speed", type=float, default=0.5)
    parser.add_argument("--slope-speed-scale", type=float, default=1.5)
    args = parser.parse_args()

    path = np.load(args.path).astype(np.float64)
    vehicle = VehicleModelConfig(
        nominal_speed_mps=float(args.nominal_speed),
        max_lateral_accel_mps2=float(args.max_lateral_accel),
        max_longitudinal_accel_mps2=float(args.max_longitudinal_accel),
        max_yaw_rate_radps=float(args.max_yaw_rate),
        min_speed_mps=float(args.min_speed),
        slope_speed_scale=float(args.slope_speed_scale),
    )
    trajectory = generate_airsim_reference_trajectory(
        path,
        vehicle=vehicle,
        sample_spacing_m=float(args.sample_spacing),
    )
    output = args.output or args.path.with_name(f"{args.path.stem}_trajectory.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **trajectory)

    positions = trajectory["positions"]
    speed = trajectory["speed"]
    yaw_rate = trajectory["yaw_rate"]
    curvature = trajectory["curvature"]
    t = trajectory["t"]
    print(f"Saved reference trajectory to {output}")
    print(f"Samples: {positions.shape[0]}")
    print(f"Duration: {float(t[-1]):.2f} s")
    print(f"Speed range: {float(np.min(speed)):.2f} to {float(np.max(speed)):.2f} m/s")
    print(f"Max yaw rate: {float(np.max(np.abs(yaw_rate))):.3f} rad/s")
    print(f"Max curvature: {float(np.max(np.abs(curvature))):.4f} 1/m")


if __name__ == "__main__":
    main()
