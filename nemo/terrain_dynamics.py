from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class TerrainQuery:
    """Terrain quantities evaluated at one or more world-frame xy points."""

    xy: np.ndarray
    height: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    normal: np.ndarray
    slope_magnitude: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    """Nominal terrain-aware rover trajectory.

    State columns are [x, y, z, yaw, pitch, roll, v, yaw_rate].
    The pitch convention is positive for uphill motion along the rover forward
    axis. The roll convention is positive when terrain height increases toward
    the rover's left lateral axis.
    """

    t: np.ndarray
    positions: np.ndarray
    yaws: np.ndarray
    pitches: np.ndarray
    rolls: np.ndarray
    velocities: np.ndarray
    yaw_rates: np.ndarray
    states: np.ndarray
    controls: dict[str, np.ndarray]
    terrain_info: dict[str, np.ndarray]


class TerrainModel(ABC):
    """Abstract terrain surface z = h(x, y)."""

    @abstractmethod
    def height(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Return terrain height at x, y."""

    @abstractmethod
    def gradient(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Return [dh/dx, dh/dy] at x, y with shape (..., 2)."""

    @abstractmethod
    def hessian(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Return Hessian matrices with shape (..., 2, 2)."""

    def normal(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Return normalized terrain normal [-h_x, -h_y, 1]."""
        grad = self.gradient(x, y)
        normal = np.concatenate([-grad, np.ones((*grad.shape[:-1], 1), dtype=grad.dtype)], axis=-1)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        return normal / np.clip(norm, 1e-12, None)

    def query(self, x: ArrayLike, y: ArrayLike) -> TerrainQuery:
        xy = _stack_xy(x, y)
        grad = self.gradient(x, y)
        return TerrainQuery(
            xy=xy,
            height=np.asarray(self.height(x, y), dtype=np.float64),
            gradient=grad,
            hessian=self.hessian(x, y),
            normal=self.normal(x, y),
            slope_magnitude=np.linalg.norm(grad, axis=-1),
        )


class DEMTerrainModel(TerrainModel):
    """Continuous DEM terrain model backed by regular-grid interpolation."""

    def __init__(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        *,
        grad_x: np.ndarray | None = None,
        grad_y: np.ndarray | None = None,
        hess_xx: np.ndarray | None = None,
        hess_xy: np.ndarray | None = None,
        hess_yy: np.ndarray | None = None,
        bounds_error: bool = False,
        fill_value: float = np.nan,
    ) -> None:
        self.x_axis, self.y_axis, self.z_grid, flip_x, flip_y = _regular_axes_and_grid(x_grid, y_grid, z_grid)
        self.bounds_error = bool(bounds_error)
        self.fill_value = float(fill_value)

        if grad_x is None or grad_y is None:
            gy, gx = np.gradient(self.z_grid, self.y_axis, self.x_axis, edge_order=1)
            grad_x = gx if grad_x is None else grad_x
            grad_y = gy if grad_y is None else grad_y

        self.grad_x = _orient_like_grid(np.asarray(grad_x, dtype=np.float64), flip_x=flip_x, flip_y=flip_y)
        self.grad_y = _orient_like_grid(np.asarray(grad_y, dtype=np.float64), flip_x=flip_x, flip_y=flip_y)

        if hess_xx is None or hess_xy is None or hess_yy is None:
            gyx, gxx = np.gradient(self.grad_x, self.y_axis, self.x_axis, edge_order=1)
            gyy, gxy = np.gradient(self.grad_y, self.y_axis, self.x_axis, edge_order=1)
            hess_xx = gxx if hess_xx is None else hess_xx
            hess_xy = 0.5 * (gyx + gxy) if hess_xy is None else hess_xy
            hess_yy = gyy if hess_yy is None else hess_yy

        self.hess_xx = _orient_like_grid(np.asarray(hess_xx, dtype=np.float64), flip_x=flip_x, flip_y=flip_y)
        self.hess_xy = _orient_like_grid(np.asarray(hess_xy, dtype=np.float64), flip_x=flip_x, flip_y=flip_y)
        self.hess_yy = _orient_like_grid(np.asarray(hess_yy, dtype=np.float64), flip_x=flip_x, flip_y=flip_y)

        self._height_interp = _make_interpolator(self.y_axis, self.x_axis, self.z_grid, bounds_error, fill_value)
        self._grad_x_interp = _make_interpolator(self.y_axis, self.x_axis, self.grad_x, bounds_error, fill_value)
        self._grad_y_interp = _make_interpolator(self.y_axis, self.x_axis, self.grad_y, bounds_error, fill_value)
        self._hess_xx_interp = _make_interpolator(self.y_axis, self.x_axis, self.hess_xx, bounds_error, fill_value)
        self._hess_xy_interp = _make_interpolator(self.y_axis, self.x_axis, self.hess_xy, bounds_error, fill_value)
        self._hess_yy_interp = _make_interpolator(self.y_axis, self.x_axis, self.hess_yy, bounds_error, fill_value)

    def height(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        return _restore_query_shape(self._height_interp(_points_yx(x, y)), x, y)

    def gradient(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        gx = _restore_query_shape(self._grad_x_interp(_points_yx(x, y)), x, y)
        gy = _restore_query_shape(self._grad_y_interp(_points_yx(x, y)), x, y)
        return np.stack([gx, gy], axis=-1)

    def hessian(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        hxx = _restore_query_shape(self._hess_xx_interp(_points_yx(x, y)), x, y)
        hxy = _restore_query_shape(self._hess_xy_interp(_points_yx(x, y)), x, y)
        hyy = _restore_query_shape(self._hess_yy_interp(_points_yx(x, y)), x, y)
        return np.stack(
            [
                np.stack([hxx, hxy], axis=-1),
                np.stack([hxy, hyy], axis=-1),
            ],
            axis=-2,
        )


class NEMOTerrainModel(TerrainModel):
    """Terrain model wrapper for differentiable NEMo-style height callables."""

    def __init__(
        self,
        model: Any,
        *,
        use_autograd: bool = True,
        finite_difference_eps: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.use_autograd = bool(use_autograd)
        self.finite_difference_eps = float(finite_difference_eps)
        self._torch = _optional_torch()
        if device is None and self._torch is not None:
            try:
                first_param = next(model.parameters())
                device = str(first_param.device)
            except Exception:
                device = "cpu"
        self.device = device or "cpu"

    def height(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        if self._torch is not None and self._is_torch_model():
            xy, shape = self._torch_xy(x, y, requires_grad=False)
            with self._torch.no_grad():
                z = self._call_torch_height(xy).reshape(-1)
            return z.detach().cpu().numpy().reshape(shape)
        return _call_numpy_height(self.model, x, y)

    def gradient(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        if self.use_autograd and self._torch is not None and self._is_torch_model():
            try:
                xy, shape = self._torch_xy(x, y, requires_grad=True)
                z = self._call_torch_height(xy).reshape(-1)
                grad = self._torch.autograd.grad(z.sum(), xy, create_graph=False)[0]
                return grad.detach().cpu().numpy().reshape((*shape, 2))
            except Exception:
                pass
        return _finite_difference_gradient(self.height, x, y, self.finite_difference_eps)

    def hessian(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        if self.use_autograd and self._torch is not None and self._is_torch_model():
            try:
                xy, shape = self._torch_xy(x, y, requires_grad=True)
                z = self._call_torch_height(xy).reshape(-1)
                grad = self._torch.autograd.grad(z.sum(), xy, create_graph=True)[0]
                cols = []
                for i in range(2):
                    second = self._torch.autograd.grad(grad[:, i].sum(), xy, retain_graph=True)[0]
                    cols.append(second)
                hess = self._torch.stack(cols, dim=-1)
                return hess.detach().cpu().numpy().reshape((*shape, 2, 2))
            except Exception:
                pass
        return _finite_difference_hessian(self.height, x, y, self.finite_difference_eps)

    def _is_torch_model(self) -> bool:
        return hasattr(self.model, "h") or callable(self.model)

    def _torch_xy(self, x: ArrayLike, y: ArrayLike, *, requires_grad: bool):
        torch = self._torch
        assert torch is not None
        xy_np = _stack_xy(x, y).astype(np.float32)
        shape = xy_np.shape[:-1]
        xy = torch.as_tensor(xy_np.reshape(-1, 2), dtype=torch.float32, device=self.device)
        xy.requires_grad_(requires_grad)
        return xy, shape

    def _call_torch_height(self, xy: Any) -> Any:
        if hasattr(self.model, "h"):
            return self.model.h(xy)
        return self.model(xy)


def generate_terrain_aware_trajectory(
    waypoints_xy: np.ndarray,
    terrain_model: TerrainModel,
    *,
    nominal_speed: float = 0.2,
    sample_spacing: float = 0.25,
    smooth_path: bool = True,
    compute_controls: bool = True,
    rover_mass: float | None = None,
    gravity: float = 1.62,
    reduce_speed_on_slope: bool = False,
    slope_speed_scale: float = 1.0,
    reduce_speed_on_curvature: bool = False,
    curvature_speed_scale: float = 1.0,
) -> Trajectory:
    """Generate a nominal terrain-aware trajectory from planar waypoints."""
    path_xy = fit_smooth_path(
        waypoints_xy,
        sample_spacing=sample_spacing,
        smooth_path=smooth_path,
    )
    if path_xy.shape[0] < 2:
        raise ValueError("Trajectory generation requires at least two sampled path points.")

    x = path_xy[:, 0]
    y = path_xy[:, 1]
    z = np.asarray(terrain_model.height(x, y), dtype=np.float64)
    grad = np.asarray(terrain_model.gradient(x, y), dtype=np.float64)
    hess = np.asarray(terrain_model.hessian(x, y), dtype=np.float64)
    normal = np.asarray(terrain_model.normal(x, y), dtype=np.float64)

    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    yaw = compute_yaw_from_path(path_xy)
    pitch, roll, along_slope, cross_slope = compute_attitude_from_gradient(grad, yaw)
    slope_magnitude = np.linalg.norm(grad, axis=-1)
    curvature = _path_curvature(path_xy, s)

    speed = np.full(path_xy.shape[0], float(nominal_speed), dtype=np.float64)
    if reduce_speed_on_slope:
        speed = speed / (1.0 + float(slope_speed_scale) * slope_magnitude)
    if reduce_speed_on_curvature:
        speed = speed / (1.0 + float(curvature_speed_scale) * np.abs(curvature))
    speed = np.clip(speed, 1e-8, None)

    t = _time_from_path_lengths(segment_lengths, speed)
    yaw_rate = _time_derivative(np.unwrap(yaw), t)
    accel = _time_derivative(speed, t)

    positions = np.column_stack([x, y, z])
    states = np.column_stack([x, y, z, yaw, pitch, roll, speed, yaw_rate])
    controls: dict[str, np.ndarray] = {}
    if compute_controls:
        controls["v_cmd"] = speed.copy()
        controls["yaw_rate_cmd"] = yaw_rate.copy()
        controls["accel_cmd"] = accel.copy()
        if rover_mass is not None:
            slope_angle = np.arctan(along_slope)
            force_gravity = float(rover_mass) * float(gravity) * np.sin(slope_angle)
            force_inertial = float(rover_mass) * accel
            controls["force_cmd"] = force_inertial + force_gravity

    terrain_info = {
        "height": z,
        "gradient": grad,
        "hessian": hess,
        "normal": normal,
        "slope_magnitude": slope_magnitude,
        "along_track_slope": along_slope,
        "cross_track_slope": cross_slope,
        "curvature": curvature,
        "arc_length": s,
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


def fit_smooth_path(
    waypoints_xy: np.ndarray,
    *,
    sample_spacing: float,
    smooth_path: bool = True,
) -> np.ndarray:
    """Fit/interpolate a path and resample by approximate arc length."""
    waypoints = _validate_waypoints(waypoints_xy)
    if not smooth_path or waypoints.shape[0] < 3:
        return resample_waypoints_by_arclength(waypoints, sample_spacing)
    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        return resample_waypoints_by_arclength(waypoints, sample_spacing)

    cumulative = _cumulative_lengths(waypoints)
    if cumulative[-1] <= 1e-12:
        return waypoints[:1].copy()
    k = min(3, waypoints.shape[0] - 1)
    try:
        tck, _ = splprep(
            [waypoints[:, 0], waypoints[:, 1]],
            u=cumulative / cumulative[-1],
            s=0.0,
            k=k,
        )
    except Exception:
        return resample_waypoints_by_arclength(waypoints, sample_spacing)

    dense_count = max(int(np.ceil(cumulative[-1] / max(float(sample_spacing), 1e-8))) * 10, 100)
    u_dense = np.linspace(0.0, 1.0, dense_count)
    dense = np.column_stack(splev(u_dense, tck)).astype(np.float64)
    return resample_waypoints_by_arclength(dense, sample_spacing)


def resample_waypoints_by_arclength(waypoints_xy: np.ndarray, sample_spacing: float) -> np.ndarray:
    """Piecewise-linear waypoint resampling by approximate arc length."""
    waypoints = _validate_waypoints(waypoints_xy)
    cumulative = _cumulative_lengths(waypoints)
    total = float(cumulative[-1])
    if total <= 1e-12:
        return waypoints[:1].copy()
    spacing = max(float(sample_spacing), 1e-8)
    count = max(int(np.ceil(total / spacing)) + 1, 2)
    target = np.linspace(0.0, total, count)
    x = np.interp(target, cumulative, waypoints[:, 0])
    y = np.interp(target, cumulative, waypoints[:, 1])
    return np.column_stack([x, y]).astype(np.float64)


def compute_yaw_from_path(path_xy: np.ndarray) -> np.ndarray:
    """Compute unwrapped heading yaw from sampled path tangents."""
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("Expected path_xy to have shape (N, 2).")
    if path.shape[0] < 2:
        return np.zeros(path.shape[0], dtype=np.float64)
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    return np.unwrap(np.arctan2(dy, dx))


def compute_attitude_from_gradient(
    gradient: np.ndarray,
    yaw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute terrain-aligned pitch and roll from gradient in body axes."""
    grad = np.asarray(gradient, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    forward = np.column_stack([np.cos(yaw), np.sin(yaw)])
    lateral_left = np.column_stack([-np.sin(yaw), np.cos(yaw)])
    along_slope = np.sum(grad * forward, axis=-1)
    cross_slope = np.sum(grad * lateral_left, axis=-1)
    pitch = np.arctan(along_slope)
    roll = np.arctan(cross_slope)
    return pitch, roll, along_slope, cross_slope


def finite_difference_derivatives(
    height_fn: Callable[[ArrayLike, ArrayLike], np.ndarray],
    x: ArrayLike,
    y: ArrayLike,
    *,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-difference gradient and Hessian for a height function."""
    return (
        _finite_difference_gradient(height_fn, x, y, eps),
        _finite_difference_hessian(height_fn, x, y, eps),
    )


def _make_interpolator(y_axis, x_axis, grid, bounds_error, fill_value):
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as exc:
        raise ImportError("DEMTerrainModel requires scipy for RegularGridInterpolator.") from exc
    return RegularGridInterpolator(
        (np.asarray(y_axis, dtype=np.float64), np.asarray(x_axis, dtype=np.float64)),
        np.asarray(grid, dtype=np.float64),
        bounds_error=bounds_error,
        fill_value=fill_value,
    )


def _regular_axes_and_grid(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray):
    z = np.asarray(z_grid, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z_grid must be 2D.")
    if x.ndim == 1 and y.ndim == 1:
        x_axis = x.copy()
        y_axis = y.copy()
    elif x.ndim == 2 and y.ndim == 2:
        if x.shape != z.shape or y.shape != z.shape:
            raise ValueError("2D x_grid and y_grid must match z_grid shape.")
        x_axis = x[0, :].copy()
        y_axis = y[:, 0].copy()
    else:
        raise ValueError("x_grid and y_grid must both be 1D axes or 2D mesh grids.")
    if x_axis.size != z.shape[1] or y_axis.size != z.shape[0]:
        raise ValueError("Grid axis lengths must match z_grid shape.")
    flip_x = bool(x_axis[0] > x_axis[-1])
    flip_y = bool(y_axis[0] > y_axis[-1])
    if flip_x:
        x_axis = x_axis[::-1]
        z = z[:, ::-1]
    if flip_y:
        y_axis = y_axis[::-1]
        z = z[::-1, :]
    return x_axis, y_axis, z, flip_x, flip_y


def _orient_like_grid(grid: np.ndarray, *, flip_x: bool, flip_y: bool) -> np.ndarray:
    if grid.ndim != 2:
        raise ValueError("Derivative grids must be 2D.")
    if flip_x:
        grid = grid[:, ::-1]
    if flip_y:
        grid = grid[::-1, :]
    return grid


def _points_yx(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape.")
    return np.column_stack([y_arr.reshape(-1), x_arr.reshape(-1)])


def _restore_query_shape(values: np.ndarray, x: ArrayLike, y: ArrayLike) -> np.ndarray:
    shape = np.broadcast_shapes(np.asarray(x).shape, np.asarray(y).shape)
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _stack_xy(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape.")
    return np.stack([x_arr, y_arr], axis=-1)


def _call_numpy_height(model: Any, x: ArrayLike, y: ArrayLike) -> np.ndarray:
    xy = _stack_xy(x, y)
    if hasattr(model, "h"):
        out = model.h(xy.reshape(-1, 2))
    elif callable(model):
        try:
            out = model(xy.reshape(-1, 2))
        except TypeError:
            out = model(np.asarray(x), np.asarray(y))
    else:
        raise TypeError("NEMOTerrainModel model must be callable or expose .h(xy).")
    return np.asarray(out, dtype=np.float64).reshape(np.asarray(x).shape)


def _finite_difference_gradient(height_fn, x: ArrayLike, y: ArrayLike, eps: float) -> np.ndarray:
    eps = float(eps)
    hx = (height_fn(np.asarray(x) + eps, y) - height_fn(np.asarray(x) - eps, y)) / (2.0 * eps)
    hy = (height_fn(x, np.asarray(y) + eps) - height_fn(x, np.asarray(y) - eps)) / (2.0 * eps)
    return np.stack([hx, hy], axis=-1).astype(np.float64)


def _finite_difference_hessian(height_fn, x: ArrayLike, y: ArrayLike, eps: float) -> np.ndarray:
    eps = float(eps)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z = height_fn(x_arr, y_arr)
    hxx = (height_fn(x_arr + eps, y_arr) - 2.0 * z + height_fn(x_arr - eps, y_arr)) / (eps**2)
    hyy = (height_fn(x_arr, y_arr + eps) - 2.0 * z + height_fn(x_arr, y_arr - eps)) / (eps**2)
    hxy = (
        height_fn(x_arr + eps, y_arr + eps)
        - height_fn(x_arr + eps, y_arr - eps)
        - height_fn(x_arr - eps, y_arr + eps)
        + height_fn(x_arr - eps, y_arr - eps)
    ) / (4.0 * eps**2)
    return np.stack(
        [
            np.stack([hxx, hxy], axis=-1),
            np.stack([hxy, hyy], axis=-1),
        ],
        axis=-2,
    ).astype(np.float64)


def _optional_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _validate_waypoints(waypoints_xy: np.ndarray) -> np.ndarray:
    waypoints = np.asarray(waypoints_xy, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("Expected waypoints_xy to have shape (N, 2).")
    if waypoints.shape[0] < 2:
        raise ValueError("At least two waypoints are required.")
    return waypoints


def _cumulative_lengths(path_xy: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _time_from_path_lengths(segment_lengths: np.ndarray, speed: np.ndarray) -> np.ndarray:
    if segment_lengths.size == 0:
        return np.zeros(speed.shape[0], dtype=np.float64)
    segment_speed = 0.5 * (speed[:-1] + speed[1:])
    dt = segment_lengths / np.clip(segment_speed, 1e-8, None)
    return np.concatenate([[0.0], np.cumsum(dt)])


def _time_derivative(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values)
    if np.allclose(t, t[0]):
        return np.zeros_like(values)
    return np.gradient(values, t, edge_order=1)


def _path_curvature(path_xy: np.ndarray, s: np.ndarray) -> np.ndarray:
    if path_xy.shape[0] < 3 or s[-1] <= 1e-12:
        return np.zeros(path_xy.shape[0], dtype=np.float64)
    dx = np.gradient(path_xy[:, 0], s, edge_order=1)
    dy = np.gradient(path_xy[:, 1], s, edge_order=1)
    ddx = np.gradient(dx, s, edge_order=1)
    ddy = np.gradient(dy, s, edge_order=1)
    denom = np.clip((dx**2 + dy**2) ** 1.5, 1e-12, None)
    return (dx * ddy - dy * ddx) / denom


def _demo() -> None:
    x_axis = np.linspace(-4.0, 4.0, 81)
    y_axis = np.linspace(-3.0, 3.0, 61)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    zz = 0.1 * np.sin(xx) + 0.05 * np.cos(yy)
    terrain = DEMTerrainModel(xx, yy, zz)
    waypoints = np.array([[-3.5, -2.5], [-1.0, -0.3], [1.0, 0.7], [3.5, 2.2]])
    trajectory = generate_terrain_aware_trajectory(
        waypoints,
        terrain,
        nominal_speed=0.2,
        sample_spacing=0.25,
        rover_mass=25.0,
    )
    print("states", trajectory.states.shape)
    print("positions", trajectory.positions.shape)
    print("controls", {key: value.shape for key, value in trajectory.controls.items()})
    print("first state", np.array2string(trajectory.states[0], precision=4))
    print("last state", np.array2string(trajectory.states[-1], precision=4))


if __name__ == "__main__":
    _demo()
