from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from nemo.terrain_dynamics import (
    TerrainModel,
    Trajectory,
    fit_smooth_path,
    generate_terrain_aware_trajectory,
    resample_waypoints_by_arclength,
)


@dataclass(frozen=True)
class EnergyTimeDiagnostics:
    """Physical path-cost diagnostics.

    Energy is reported in Joules. Time is reported in seconds. The quasi-static
    model evaluates each path segment using the terrain gradient at the segment
    midpoint and the segment tangent direction in the xy plane.
    """

    energy_joules: float
    time_seconds: float
    path_length_3d_m: float
    average_slope: float
    max_slope: float
    rolling_energy_joules: float
    slope_energy_joules: float
    recovered_energy_joules: float


@dataclass(frozen=True)
class PathOptimizationConfig:
    """Configuration for terrain-aware control-point path optimization."""

    num_control_points: int = 8
    num_iters: int = 200
    sample_spacing: float = 0.25
    nominal_speed: float = 0.2
    max_slope_rad: float = float(np.deg2rad(15.0))
    max_roll_rad: float = float(np.deg2rad(15.0))
    max_curvature: float = 1.0
    mass_kg: float = 25.0
    gravity_mps2: float = 1.62
    rolling_resistance_coeff: float = 0.05
    drivetrain_efficiency: float = 0.75
    nominal_speed_mps: float = 0.2
    w_energy: float = 1.0
    w_time: float = 0.0
    w_smooth: float = 1.0
    allow_regen: bool = False
    regen_efficiency: float = 0.3
    w_length: float = 0.0
    w_slope: float = 10.0
    w_roll: float = 10.0
    w_curvature: float = 1.0
    w_yaw_rate: float = 0.5
    w_accel: float = 0.1
    w_anchor: float = 0.05
    bounds_margin: float = 0.0
    optimizer: Literal["scipy", "torch"] = "scipy"
    verbose: bool = False


@dataclass(frozen=True)
class PathOptimizationResult:
    initial_trajectory: Trajectory
    optimized_trajectory: Trajectory
    initial_cost: float
    optimized_cost: float
    cost_history: list[float]
    control_points_initial: np.ndarray
    control_points_optimized: np.ndarray
    success: bool
    message: str


class TerrainAwarePathOptimizer:
    """Optimize a small set of 2D control points through terrain-aware rollout."""

    def __init__(
        self,
        terrain_model: TerrainModel,
        config: PathOptimizationConfig | None = None,
    ) -> None:
        self.terrain_model = terrain_model
        self.config = config or PathOptimizationConfig()

    def optimize(self, initial_waypoints_xy: np.ndarray) -> PathOptimizationResult:
        """Optimize intermediate 2D control points while preserving start/goal."""
        waypoints = _validate_path(initial_waypoints_xy)
        start = waypoints[0].copy()
        goal = waypoints[-1].copy()
        control_initial = self.initialize_control_points(waypoints)
        initial_control_waypoints = self.reconstruct_waypoints_from_control_points(
            start,
            control_initial,
            goal,
        )
        reference_path_xy = fit_smooth_path(
            waypoints,
            sample_spacing=self.config.sample_spacing,
            smooth_path=True,
        )
        initial_trajectory = self._trajectory_from_waypoints(initial_control_waypoints)
        initial_cost = self.compute_cost(initial_trajectory, reference_path_xy=reference_path_xy)

        if control_initial.size == 0:
            return PathOptimizationResult(
                initial_trajectory=initial_trajectory,
                optimized_trajectory=initial_trajectory,
                initial_cost=initial_cost,
                optimized_cost=initial_cost,
                cost_history=[initial_cost],
                control_points_initial=control_initial,
                control_points_optimized=control_initial,
                success=True,
                message="No intermediate control points to optimize.",
            )

        if self.config.optimizer == "torch":
            return self._optimize_torch_placeholder(
                start,
                goal,
                control_initial,
                initial_trajectory,
                initial_cost,
                reference_path_xy,
            )
        if self.config.optimizer != "scipy":
            raise ValueError(f"Unsupported optimizer '{self.config.optimizer}'.")

        return self._optimize_scipy(
            start,
            goal,
            control_initial,
            initial_trajectory,
            initial_cost,
            reference_path_xy,
        )

    def compute_cost(
        self,
        trajectory: Trajectory,
        reference_path_xy: np.ndarray | None = None,
    ) -> float:
        """Compute scalar soft-constrained terrain trajectory cost."""
        diagnostics = self.compute_diagnostics(trajectory)
        return float(diagnostics["total_cost"])

    def compute_diagnostics(
        self,
        trajectory: Trajectory,
        reference_path_xy: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Return objective value and interpretable physical diagnostics."""
        cfg = self.config
        positions = np.asarray(trajectory.positions, dtype=np.float64)
        xy = positions[:, :2]
        energy_time = compute_energy_time_diagnostics(
            trajectory,
            mass_kg=cfg.mass_kg,
            gravity_mps2=cfg.gravity_mps2,
            rolling_resistance_coeff=cfg.rolling_resistance_coeff,
            drivetrain_efficiency=cfg.drivetrain_efficiency,
            nominal_speed_mps=cfg.nominal_speed_mps,
            allow_regen=cfg.allow_regen,
            regen_efficiency=cfg.regen_efficiency,
        )

        slope = _slope_angle_from_trajectory(trajectory)
        slope_cost = relu_squared_violation(slope, cfg.max_slope_rad)
        roll_cost = relu_squared_violation(np.abs(trajectory.rolls), cfg.max_roll_rad)

        curvature = planar_curvature(xy)
        curvature_cost = float(np.mean(curvature**2))
        curvature_cost += relu_squared_violation(np.abs(curvature), cfg.max_curvature)

        yaw_rate = np.asarray(trajectory.yaw_rates, dtype=np.float64)
        yaw_rate_cost = float(np.nanmean(yaw_rate**2))

        accel = _trajectory_accel(trajectory)
        accel_cost = float(np.nanmean(accel**2))

        anchor_cost = 0.0
        if reference_path_xy is not None and cfg.w_anchor != 0.0:
            dist = nearest_path_distance(xy, reference_path_xy)
            anchor_cost = float(np.nanmean(dist**2))

        smooth_cost = curvature_cost + yaw_rate_cost + accel_cost
        total_cost = (
            float(cfg.w_energy) * energy_time.energy_joules
            + float(cfg.w_time) * energy_time.time_seconds
            + float(cfg.w_smooth) * smooth_cost
            + float(cfg.w_length) * energy_time.path_length_3d_m
            + float(cfg.w_slope) * slope_cost
            + float(cfg.w_roll) * roll_cost
            + float(cfg.w_curvature) * curvature_cost
            + float(cfg.w_yaw_rate) * yaw_rate_cost
            + float(cfg.w_accel) * accel_cost
            + float(cfg.w_anchor) * anchor_cost
        )
        if not np.isfinite(total_cost):
            total_cost = float("inf")
        return {
            "total_cost": float(total_cost),
            "energy_joules": energy_time.energy_joules,
            "time_seconds": energy_time.time_seconds,
            "path_length_3d_m": energy_time.path_length_3d_m,
            "average_slope": energy_time.average_slope,
            "max_slope": energy_time.max_slope,
            "rolling_energy_joules": energy_time.rolling_energy_joules,
            "slope_energy_joules": energy_time.slope_energy_joules,
            "recovered_energy_joules": energy_time.recovered_energy_joules,
            "slope_violation_cost": float(slope_cost),
            "roll_violation_cost": float(roll_cost),
            "curvature_cost": float(curvature_cost),
            "yaw_rate_cost": float(yaw_rate_cost),
            "accel_cost": float(accel_cost),
            "smooth_cost": float(smooth_cost),
            "anchor_cost": float(anchor_cost),
        }

    def initialize_control_points(self, initial_waypoints_xy: np.ndarray) -> np.ndarray:
        """Return num_control_points intermediate points sampled from input path."""
        waypoints = _validate_path(initial_waypoints_xy)
        n_ctrl = max(int(self.config.num_control_points), 0)
        if n_ctrl == 0:
            return np.empty((0, 2), dtype=np.float64)
        sampled = resample_waypoints_by_arclength(waypoints, sample_spacing=1.0)
        cumulative = _cumulative_lengths(sampled)
        total = float(cumulative[-1])
        if total <= 1e-12:
            return np.repeat(waypoints[:1], n_ctrl, axis=0)
        targets = np.linspace(0.0, total, n_ctrl + 2)[1:-1]
        x = np.interp(targets, cumulative, sampled[:, 0])
        y = np.interp(targets, cumulative, sampled[:, 1])
        return np.column_stack([x, y]).astype(np.float64)

    def reconstruct_waypoints_from_control_points(
        self,
        start: np.ndarray,
        control_points: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Assemble fixed endpoints and optimized intermediate control points."""
        controls = np.asarray(control_points, dtype=np.float64).reshape(-1, 2)
        return np.vstack([np.asarray(start, dtype=np.float64), controls, np.asarray(goal, dtype=np.float64)])

    def _optimize_scipy(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        control_initial: np.ndarray,
        initial_trajectory: Trajectory,
        initial_cost: float,
        reference_path_xy: np.ndarray,
    ) -> PathOptimizationResult:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ImportError("TerrainAwarePathOptimizer with optimizer='scipy' requires scipy.") from exc

        cost_history: list[float] = []
        best_cost = float(initial_cost)
        best_control = control_initial.copy()
        best_trajectory = initial_trajectory

        def objective(flat_control_points: np.ndarray) -> float:
            nonlocal best_cost, best_control, best_trajectory
            controls = np.asarray(flat_control_points, dtype=np.float64).reshape(-1, 2)
            waypoints = self.reconstruct_waypoints_from_control_points(start, controls, goal)
            try:
                trajectory = self._trajectory_from_waypoints(waypoints)
                cost = self.compute_cost(trajectory, reference_path_xy=reference_path_xy)
            except Exception:
                cost = float("inf")
                trajectory = None
            if np.isfinite(cost) and cost < best_cost:
                best_cost = float(cost)
                best_control = controls.copy()
                best_trajectory = trajectory  # type: ignore[assignment]
            cost_history.append(float(cost))
            if self.config.verbose:
                print(f"[terrain_path_optimizer] eval={len(cost_history)} cost={cost:.6f}")
            return float(cost)

        result = minimize(
            objective,
            control_initial.reshape(-1),
            method="L-BFGS-B",
            bounds=self._control_bounds(control_initial.shape[0]),
            options={"maxiter": int(self.config.num_iters), "disp": bool(self.config.verbose)},
        )

        final_control = np.asarray(result.x, dtype=np.float64).reshape(-1, 2)
        final_waypoints = self.reconstruct_waypoints_from_control_points(start, final_control, goal)
        try:
            final_trajectory = self._trajectory_from_waypoints(final_waypoints)
            final_cost = self.compute_cost(final_trajectory, reference_path_xy=reference_path_xy)
            if np.isfinite(final_cost) and final_cost < best_cost:
                best_cost = float(final_cost)
                best_control = final_control.copy()
                best_trajectory = final_trajectory
        except Exception:
            pass

        success = bool(result.success) or bool(best_cost < initial_cost)
        return PathOptimizationResult(
            initial_trajectory=initial_trajectory,
            optimized_trajectory=best_trajectory,
            initial_cost=float(initial_cost),
            optimized_cost=float(best_cost),
            cost_history=cost_history,
            control_points_initial=control_initial,
            control_points_optimized=best_control,
            success=success,
            message=str(result.message),
        )

    def _optimize_torch_placeholder(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        control_initial: np.ndarray,
        initial_trajectory: Trajectory,
        initial_cost: float,
        reference_path_xy: np.ndarray,
    ) -> PathOptimizationResult:
        # TODO: Implement a differentiable Adam loop once terrain trajectory
        # generation has a torch-native path and no numpy/scipy interpolation.
        return self._optimize_scipy(
            start,
            goal,
            control_initial,
            initial_trajectory,
            initial_cost,
            reference_path_xy,
        )

    def _trajectory_from_waypoints(self, waypoints_xy: np.ndarray) -> Trajectory:
        return generate_terrain_aware_trajectory(
            waypoints_xy,
            self.terrain_model,
            nominal_speed=self.config.nominal_speed_mps,
            sample_spacing=self.config.sample_spacing,
            smooth_path=True,
            compute_controls=True,
        )

    def _control_bounds(self, num_controls: int):
        bounds = _terrain_xy_bounds(self.terrain_model, margin=float(self.config.bounds_margin))
        if bounds is None:
            return None
        (x_min, x_max), (y_min, y_max) = bounds
        return [(x_min, x_max), (y_min, y_max)] * int(num_controls)


def planar_curvature(xy: np.ndarray) -> np.ndarray:
    """Estimate signed planar curvature along sampled xy points."""
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected xy to have shape (N, 2).")
    if points.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=np.float64)
    s = _cumulative_lengths(points)
    if s[-1] <= 1e-12:
        return np.zeros(points.shape[0], dtype=np.float64)
    dx = np.gradient(points[:, 0], s, edge_order=1)
    dy = np.gradient(points[:, 1], s, edge_order=1)
    ddx = np.gradient(dx, s, edge_order=1)
    ddy = np.gradient(dy, s, edge_order=1)
    denom = np.clip((dx**2 + dy**2) ** 1.5, 1e-12, None)
    return (dx * ddy - dy * ddx) / denom


def unwrap_angle(angle: np.ndarray) -> np.ndarray:
    """Unwrap an angle sequence in radians."""
    return np.unwrap(np.asarray(angle, dtype=np.float64))


def relu_squared_violation(values: np.ndarray, limit: float) -> float:
    """Mean squared positive violation over an absolute or one-sided limit."""
    values = np.asarray(values, dtype=np.float64)
    violation = np.maximum(values - float(limit), 0.0)
    return float(np.nanmean(violation**2))


def nearest_path_distance(points_xy: np.ndarray, reference_xy: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest point on a reference polyline."""
    points = np.asarray(points_xy, dtype=np.float64)
    reference = _validate_path(reference_xy)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected points_xy to have shape (N, 2).")
    if reference.shape[0] == 1:
        return np.linalg.norm(points - reference[0], axis=1)

    out = np.full(points.shape[0], np.inf, dtype=np.float64)
    a = reference[:-1]
    b = reference[1:]
    ab = b - a
    ab_norm2 = np.sum(ab * ab, axis=1)
    for i, point in enumerate(points):
        ap = point - a
        t = np.divide(
            np.sum(ap * ab, axis=1),
            ab_norm2,
            out=np.zeros_like(ab_norm2),
            where=ab_norm2 > 1e-12,
        )
        t = np.clip(t, 0.0, 1.0)
        closest = a + t[:, None] * ab
        out[i] = float(np.min(np.linalg.norm(point - closest, axis=1)))
    return out


def compute_energy_time_diagnostics(
    trajectory: Trajectory,
    *,
    mass_kg: float,
    gravity_mps2: float = 1.62,
    rolling_resistance_coeff: float = 0.05,
    drivetrain_efficiency: float = 0.75,
    nominal_speed_mps: float = 0.2,
    allow_regen: bool = False,
    regen_efficiency: float = 0.3,
) -> EnergyTimeDiagnostics:
    """Compute quasi-static traversal energy and time.

    Segment force model, in SI units:
        F_drive = m g (c_r cos(phi) + sin(phi))
        dE = max(F_drive, 0) / eta * ds

    Directional slope s = dz / ds_xy is estimated from the trajectory terrain
    gradient at segment midpoints and the xy segment tangent. Then:
        sin(phi) = s / sqrt(1 + s^2)
        cos(phi) = 1 / sqrt(1 + s^2)
        ds = ds_xy * sqrt(1 + s^2)
    """
    xy = np.asarray(trajectory.positions[:, :2], dtype=np.float64)
    if xy.shape[0] < 2:
        return EnergyTimeDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    dxy = np.diff(xy, axis=0)
    ds_xy = np.linalg.norm(dxy, axis=1)
    valid = ds_xy > 1e-12
    tangent = np.zeros_like(dxy)
    tangent[valid] = dxy[valid] / ds_xy[valid, None]

    grad = np.asarray(trajectory.terrain_info.get("gradient"), dtype=np.float64)
    if grad.shape[0] == xy.shape[0]:
        grad_mid = 0.5 * (grad[:-1] + grad[1:])
    else:
        grad_mid = np.zeros_like(dxy)

    directional_slope = np.sum(grad_mid * tangent, axis=1)
    directional_slope = np.where(valid & np.isfinite(directional_slope), directional_slope, 0.0)
    slope_scale = np.sqrt(1.0 + directional_slope**2)
    sin_phi = directional_slope / slope_scale
    cos_phi = 1.0 / slope_scale
    ds = ds_xy * slope_scale

    mass = float(mass_kg)
    gravity = float(gravity_mps2)
    rolling = float(rolling_resistance_coeff)
    eta = max(float(drivetrain_efficiency), 1e-12)
    regen_eta = max(float(regen_efficiency), 0.0)

    rolling_force = mass * gravity * rolling * cos_phi
    slope_force = mass * gravity * sin_phi
    drive_force = rolling_force + slope_force
    positive_force = np.maximum(drive_force, 0.0)
    positive_energy = positive_force / eta * ds

    recovered_energy = np.zeros_like(positive_energy)
    if allow_regen:
        recovered_energy = np.maximum(-drive_force, 0.0) * regen_eta * ds
    energy = positive_energy - recovered_energy

    rolling_energy = np.sum(rolling_force / eta * ds)
    slope_energy = np.sum(np.maximum(slope_force, 0.0) / eta * ds)
    total_energy = float(np.sum(energy))
    total_time = float(np.sum(ds) / max(float(nominal_speed_mps), 1e-12))
    slope_angle = np.abs(np.arctan(directional_slope))
    return EnergyTimeDiagnostics(
        energy_joules=total_energy,
        time_seconds=total_time,
        path_length_3d_m=float(np.sum(ds)),
        average_slope=float(np.mean(slope_angle)) if slope_angle.size else 0.0,
        max_slope=float(np.max(slope_angle)) if slope_angle.size else 0.0,
        rolling_energy_joules=float(rolling_energy),
        slope_energy_joules=float(slope_energy),
        recovered_energy_joules=float(np.sum(recovered_energy)),
    )


def _trajectory_accel(trajectory: Trajectory) -> np.ndarray:
    if "accel_cmd" in trajectory.controls:
        return np.asarray(trajectory.controls["accel_cmd"], dtype=np.float64)
    t = np.asarray(trajectory.t, dtype=np.float64)
    v = np.asarray(trajectory.velocities, dtype=np.float64)
    if v.shape[0] < 2 or np.allclose(t, t[0]):
        return np.zeros_like(v)
    return np.gradient(v, t, edge_order=1)


def _slope_angle_from_trajectory(trajectory: Trajectory) -> np.ndarray:
    if "slope_magnitude" in trajectory.terrain_info:
        slope = np.asarray(trajectory.terrain_info["slope_magnitude"], dtype=np.float64)
        return np.arctan(slope)
    pitch_roll = np.hypot(trajectory.pitches, trajectory.rolls)
    return np.asarray(pitch_roll, dtype=np.float64)


def _path_length_3d(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return 0.0
    return float(np.nansum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def _terrain_xy_bounds(
    terrain_model: TerrainModel,
    *,
    margin: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if hasattr(terrain_model, "x_axis") and hasattr(terrain_model, "y_axis"):
        x_axis = np.asarray(getattr(terrain_model, "x_axis"), dtype=np.float64)
        y_axis = np.asarray(getattr(terrain_model, "y_axis"), dtype=np.float64)
        return (
            (float(np.nanmin(x_axis)) + margin, float(np.nanmax(x_axis)) - margin),
            (float(np.nanmin(y_axis)) + margin, float(np.nanmax(y_axis)) - margin),
        )
    bounds = getattr(terrain_model, "bounds", None)
    if bounds is not None:
        if hasattr(bounds, "x") and hasattr(bounds, "y"):
            return (
                (float(bounds.x[0]) + margin, float(bounds.x[1]) - margin),
                (float(bounds.y[0]) + margin, float(bounds.y[1]) - margin),
            )
        try:
            return (
                (float(bounds[0][0]) + margin, float(bounds[0][1]) - margin),
                (float(bounds[1][0]) + margin, float(bounds[1][1]) - margin),
            )
        except Exception:
            return None
    return None


def _validate_path(path_xy: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("Expected path xy array with shape (N, 2).")
    if path.shape[0] < 2:
        raise ValueError("Path must contain at least start and goal points.")
    return path


def _cumulative_lengths(path_xy: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _demo() -> None:
    from nemo.terrain_dynamics import DEMTerrainModel

    x_axis = np.linspace(-8.0, 8.0, 161)
    y_axis = np.linspace(-5.0, 5.0, 101)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    zz = 0.2 * np.sin(0.5 * xx) + 0.1 * np.cos(0.5 * yy)
    terrain = DEMTerrainModel(xx, yy, zz)

    initial = np.array(
        [
            [-7.0, -3.0],
            [-4.5, 2.0],
            [-2.0, -1.5],
            [1.0, 1.7],
            [4.0, -1.0],
            [7.0, 3.0],
        ],
        dtype=np.float64,
    )
    cfg = PathOptimizationConfig(
        num_control_points=6,
        num_iters=60,
        sample_spacing=0.35,
        nominal_speed=0.2,
        verbose=False,
    )
    result = TerrainAwarePathOptimizer(terrain, cfg).optimize(initial)
    print(f"initial cost: {result.initial_cost:.6f}")
    print(f"optimized cost: {result.optimized_cost:.6f}")
    print("states", result.optimized_trajectory.states.shape)
    print("controls", {key: value.shape for key, value in result.optimized_trajectory.controls.items()})
    print("first optimized positions")
    print(np.array2string(result.optimized_trajectory.positions[:5], precision=4))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure()
    plt.plot(result.initial_trajectory.positions[:, 0], result.initial_trajectory.positions[:, 1], "--", label="initial")
    plt.plot(result.optimized_trajectory.positions[:, 0], result.optimized_trajectory.positions[:, 1], label="optimized")
    plt.axis("equal")
    plt.legend()
    plt.title("Terrain-aware path optimization")
    plt.show()


if __name__ == "__main__":
    _demo()
