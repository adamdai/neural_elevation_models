from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm


@dataclass(frozen=True)
class TerrainAwarePlannerConfig:
    """Configuration for differentiable terrain-aware NEMo path planning.

    Costs are optionally normalized by their initial-path values. With
    normalization enabled, the weights are intended to be interpretable and live
    in [0, 1].
    """

    num_control_points: int = 12
    num_samples: int = 240
    num_iters: int = 1500
    lr: float = 1e-2
    nominal_speed: float = 1.0
    wheelbase_m: float = 1.0
    max_steering_rad: float = 0.6
    max_throttle_accel: float = 2.0
    gravity: float = 1.62
    uphill_limit_rad: float = float(np.deg2rad(15.0))
    downhill_limit_rad: float = float(np.deg2rad(15.0))
    crosstrack_limit_rad: float = float(np.deg2rad(15.0))
    w_length: float = 0.25
    w_uphill: float = 0.25
    w_downhill: float = 0.25
    w_crosstrack: float = 0.25
    w_throttle: float = 0.25
    w_steering: float = 0.25
    w_smoothness: float = 0.01
    gradient_mode: Literal["finite_difference", "autograd"] = "finite_difference"
    finite_difference_eps: float | None = None
    bounds_margin: float = 0.0
    normalize_costs: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class TerrainAwarePlanDiagnostics:
    total_cost: float
    length_cost: float
    uphill_cost: float
    downhill_cost: float
    crosstrack_cost: float
    throttle_cost: float
    steering_cost: float
    smoothness_cost: float
    length_3d_m: float
    mean_slope_deg: float
    max_slope_deg: float
    mean_uphill_deg: float
    max_uphill_deg: float
    mean_downhill_deg: float
    max_downhill_deg: float
    mean_crosstrack_deg: float
    max_crosstrack_deg: float
    mean_abs_throttle: float
    max_abs_throttle: float
    mean_abs_steering_deg: float
    max_abs_steering_deg: float


@dataclass(frozen=True)
class TerrainAwarePlanResult:
    initial_path_xy: np.ndarray
    optimized_path_xy: np.ndarray
    initial_path_xyz: np.ndarray
    optimized_path_xyz: np.ndarray
    control_points_initial: np.ndarray
    control_points_optimized: np.ndarray
    initial_diagnostics: TerrainAwarePlanDiagnostics
    optimized_diagnostics: TerrainAwarePlanDiagnostics
    cost_history: list[float]
    trajectory: dict[str, np.ndarray]


def optimize_terrain_aware_path(
    nemo: Any,
    initial_path_xy: np.ndarray,
    config: TerrainAwarePlannerConfig | None = None,
    initial_control_points: np.ndarray | None = None,
) -> TerrainAwarePlanResult:
    """Optimize a smooth 2D spline path over a NEMo height field.

    Steering is derived from path curvature with a bicycle-model approximation:
    ``steering = atan(wheelbase * curvature)``.

    Throttle is a simplified terrain-aware proxy, not a full vehicle dynamics
    model. It estimates required longitudinal acceleration from gravity along
    the path using ``a_required = gravity * along_track_slope``.
    """
    cfg = config or TerrainAwarePlannerConfig()
    seed = validate_path(initial_path_xy)
    if initial_control_points is None:
        controls_np = initialize_control_points(seed, cfg.num_control_points)
    else:
        controls_np = validate_path(initial_control_points)
        if controls_np.shape[0] < 2:
            raise ValueError("Expected at least two initial control points.")
        controls_np[0] = seed[0]
        controls_np[-1] = seed[-1]
    basis_np = bspline_basis(
        num_control_points=controls_np.shape[0],
        num_samples=max(int(cfg.num_samples), 2),
        degree=min(3, controls_np.shape[0] - 1),
    )

    device = nemo.device
    basis = torch.as_tensor(basis_np, dtype=torch.float32, device=device)
    controls_initial = torch.as_tensor(controls_np, dtype=torch.float32, device=device)
    start = controls_initial[0].detach().clone()
    goal = controls_initial[-1].detach().clone()
    interior = torch.nn.Parameter(controls_initial[1:-1].detach().clone())
    optimizer = torch.optim.Adam([interior], lr=float(cfg.lr))

    initial_path = sample_path(controls_initial, basis)
    initial_trajectory = trajectory_from_path(nemo, initial_path, cfg)
    initial_raw_costs = compute_raw_costs(initial_trajectory, controls_initial, cfg)
    normalizers = _cost_normalizers(initial_raw_costs, enabled=cfg.normalize_costs)
    initial_total = _weighted_total(initial_raw_costs, normalizers, cfg)
    initial_terms = {**initial_raw_costs, "total": initial_total}

    best_total = float(initial_total.detach().cpu().item())
    best_interior = interior.detach().clone()
    best_terms = {name: value.detach().clone() for name, value in initial_terms.items()}
    history: list[float] = []

    x_min = float(nemo.field.bounds[0][0]) + float(cfg.bounds_margin)
    x_max = float(nemo.field.bounds[0][1]) - float(cfg.bounds_margin)
    y_min = float(nemo.field.bounds[1][0]) + float(cfg.bounds_margin)
    y_max = float(nemo.field.bounds[1][1]) - float(cfg.bounds_margin)

    pbar = tqdm(range(int(cfg.num_iters)), desc="Optimizing path", disable=not cfg.verbose)
    for idx in pbar:
        optimizer.zero_grad(set_to_none=True)
        controls = torch.cat([start[None], interior, goal[None]], dim=0)
        path = sample_path(controls, basis)
        traj = trajectory_from_path(nemo, path, cfg)
        raw_costs = compute_raw_costs(traj, controls, cfg)
        total = _weighted_total(raw_costs, normalizers, cfg)
        total.backward()
        optimizer.step()

        with torch.no_grad():
            interior[:, 0].clamp_(x_min, x_max)
            interior[:, 1].clamp_(y_min, y_max)
            candidate_controls = torch.cat([start[None], interior, goal[None]], dim=0)
            candidate_path = sample_path(candidate_controls, basis)
            candidate_traj = trajectory_from_path(nemo, candidate_path, cfg)
            candidate_raw = compute_raw_costs(candidate_traj, candidate_controls, cfg)
            candidate_total = _weighted_total(candidate_raw, normalizers, cfg)
            candidate_value = float(candidate_total.detach().cpu().item())
            if candidate_value < best_total:
                best_total = candidate_value
                best_interior = interior.detach().clone()
                best_terms = {
                    **{name: value.detach().clone() for name, value in candidate_raw.items()}
                }
                best_terms["total"] = candidate_total.detach().clone()
        history.append(candidate_value)
        if cfg.verbose:
            pbar.set_postfix(cost=f"{candidate_value:.4f}", best=f"{best_total:.4f}")

    best_controls = torch.cat([start[None], best_interior, goal[None]], dim=0)
    optimized_path = sample_path(best_controls, basis)
    optimized_trajectory = trajectory_from_path(nemo, optimized_path, cfg)
    optimized_raw = compute_raw_costs(optimized_trajectory, best_controls, cfg)
    optimized_total = _weighted_total(optimized_raw, normalizers, cfg)
    optimized_terms = {**optimized_raw, "total": optimized_total}

    initial_path_np = initial_path.detach().cpu().numpy().astype(np.float32)
    optimized_path_np = optimized_path.detach().cpu().numpy().astype(np.float32)
    return TerrainAwarePlanResult(
        initial_path_xy=initial_path_np,
        optimized_path_xy=optimized_path_np,
        initial_path_xyz=path_xyz(nemo, initial_path_np),
        optimized_path_xyz=path_xyz(nemo, optimized_path_np),
        control_points_initial=controls_np.astype(np.float32),
        control_points_optimized=best_controls.detach().cpu().numpy().astype(np.float32),
        initial_diagnostics=make_diagnostics(initial_trajectory, initial_terms),
        optimized_diagnostics=make_diagnostics(optimized_trajectory, optimized_terms),
        cost_history=history,
        trajectory=_trajectory_to_numpy(optimized_trajectory),
    )


def validate_path(path_xy: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError("Expected initial_path_xy with shape (N, 2), N >= 2.")
    if not np.all(np.isfinite(path)):
        raise ValueError("Path contains non-finite values.")
    return path


def cumulative_lengths(path_xy: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def initialize_control_points(initial_path_xy: np.ndarray, num_control_points: int) -> np.ndarray:
    """Arc-length sample interior controls plus fixed start and goal."""
    path = validate_path(initial_path_xy)
    total_controls = max(int(num_control_points), 0) + 2
    cumulative = cumulative_lengths(path)
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(path[:1], total_controls, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, total_controls)
    x = np.interp(targets, cumulative, path[:, 0])
    y = np.interp(targets, cumulative, path[:, 1])
    controls = np.column_stack([x, y]).astype(np.float32)
    controls[0] = path[0]
    controls[-1] = path[-1]
    return controls


def bspline_basis(num_control_points: int, num_samples: int, degree: int = 3) -> np.ndarray:
    """Return a clamped B-spline basis matrix with endpoint interpolation."""
    if num_control_points < 2:
        raise ValueError("At least two control points are required.")
    degree = int(min(max(degree, 1), num_control_points - 1))
    knot_count = num_control_points + degree + 1
    knots = np.zeros(knot_count, dtype=np.float64)
    knots[degree : num_control_points + 1] = np.linspace(0.0, 1.0, num_control_points - degree + 1)
    knots[num_control_points + 1 :] = 1.0
    u = np.linspace(0.0, 1.0, int(num_samples), dtype=np.float64)
    basis = np.column_stack(
        [_basis_function(i, degree, u, knots) for i in range(num_control_points)]
    )
    basis[0, :] = 0.0
    basis[0, 0] = 1.0
    basis[-1, :] = 0.0
    basis[-1, -1] = 1.0
    row_sum = basis.sum(axis=1, keepdims=True)
    return (basis / np.clip(row_sum, 1e-12, None)).astype(np.float32)


def sample_path(control_points: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    path = basis @ control_points
    path = path.clone()
    path[0] = control_points[0]
    path[-1] = control_points[-1]
    return path


def finite_difference_grad(
    nemo: Any, xy: torch.Tensor, config: TerrainAwarePlannerConfig
) -> torch.Tensor:
    if config.finite_difference_eps is None:
        x_range = float(nemo.field.bounds[0][1] - nemo.field.bounds[0][0])
        y_range = float(nemo.field.bounds[1][1] - nemo.field.bounds[1][0])
        eps = max(min(x_range, y_range) / 512.0, 1e-3)
    else:
        eps = float(config.finite_difference_eps)
    dx = xy.new_tensor([eps, 0.0])
    dy = xy.new_tensor([0.0, eps])
    gx = (nemo.field.h(xy + dx).squeeze(-1) - nemo.field.h(xy - dx).squeeze(-1)) / (2.0 * eps)
    gy = (nemo.field.h(xy + dy).squeeze(-1) - nemo.field.h(xy - dy).squeeze(-1)) / (2.0 * eps)
    return torch.stack([gx, gy], dim=-1)


def autograd_grad(nemo: Any, xy: torch.Tensor) -> torch.Tensor:
    """Return dh/dx, dh/dy while keeping gradients connected to the path tensor."""
    with torch.enable_grad():
        create_graph = bool(xy.requires_grad)
        if create_graph:
            xy_query = xy
        else:
            xy_query = xy.clone().detach().requires_grad_(True)
        z = nemo.field.h(xy_query).squeeze(-1)
        return torch.autograd.grad(
            outputs=z.sum(),
            inputs=xy_query,
            create_graph=create_graph,
        )[0]


def terrain_grad(nemo: Any, xy: torch.Tensor, config: TerrainAwarePlannerConfig) -> torch.Tensor:
    if config.gradient_mode == "finite_difference":
        return finite_difference_grad(nemo, xy, config)
    if config.gradient_mode == "autograd":
        return autograd_grad(nemo, xy)
    raise ValueError(
        "Unsupported terrain gradient mode "
        f"{config.gradient_mode!r}; expected 'finite_difference' or 'autograd'."
    )


def trajectory_from_path(
    nemo: Any, path_xy: torch.Tensor, config: TerrainAwarePlannerConfig
) -> dict[str, torch.Tensor]:
    """Build a differentiable terrain-aware trajectory dictionary."""
    z = nemo.field.h(path_xy).squeeze(-1)
    grad = terrain_grad(nemo, path_xy, config)

    deriv = _gradient_by_index(path_xy)
    second = _gradient_by_index(deriv)
    dx = deriv[:, 0]
    dy = deriv[:, 1]
    ddx = second[:, 0]
    ddy = second[:, 1]
    speed_xy = torch.linalg.norm(deriv, dim=-1).clamp_min(1e-8)
    tangent_xy = deriv / speed_xy[:, None]
    lateral_xy = torch.stack([-tangent_xy[:, 1], tangent_xy[:, 0]], dim=-1)

    yaw = torch.atan2(dy, dx)
    denom = (dx.pow(2) + dy.pow(2)).clamp_min(1e-8).pow(1.5)
    curvature = (dx * ddy - dy * ddx) / denom

    slope_mag = torch.atan(torch.linalg.norm(grad, dim=-1))
    along_track_slope = torch.sum(grad * tangent_xy, dim=-1)
    cross_track_slope = torch.sum(grad * lateral_xy, dim=-1)
    along_track_slope_rad = torch.atan(along_track_slope)
    cross_track_slope_rad = torch.atan(cross_track_slope)

    speed = torch.full_like(z, float(config.nominal_speed))
    steering = torch.atan(float(config.wheelbase_m) * curvature)
    a_required = float(config.gravity) * along_track_slope
    throttle = a_required / max(float(config.max_throttle_accel), 1e-8)

    dxy = path_xy[1:] - path_xy[:-1]
    dz = z[1:] - z[:-1]
    ds_3d = torch.sqrt(torch.sum(dxy.pow(2), dim=-1) + dz.pow(2)).clamp_min(1e-8)
    length_3d = torch.sum(ds_3d)

    return {
        "x": path_xy[:, 0],
        "y": path_xy[:, 1],
        "z": z,
        "yaw": yaw,
        "curvature": curvature,
        "slope_mag": slope_mag,
        "along_track_slope": along_track_slope,
        "cross_track_slope": cross_track_slope,
        "along_track_slope_rad": along_track_slope_rad,
        "cross_track_slope_rad": cross_track_slope_rad,
        "speed": speed,
        "steering": steering,
        "throttle": throttle,
        "path_xy": path_xy,
        "grad": grad,
        "length_3d": length_3d,
    }


def compute_raw_costs(
    trajectory: dict[str, torch.Tensor],
    control_points: torch.Tensor,
    config: TerrainAwarePlannerConfig,
) -> dict[str, torch.Tensor]:
    """Compute unnormalized differentiable objective terms."""
    uphill_limit = max(float(config.uphill_limit_rad), 1e-8)
    downhill_limit = max(float(config.downhill_limit_rad), 1e-8)
    crosstrack_limit = max(float(config.crosstrack_limit_rad), 1e-8)
    steering_limit = max(float(config.max_steering_rad), 1e-8)

    along_track_slope_rad = trajectory["along_track_slope_rad"]
    cross_track_slope_rad = trajectory["cross_track_slope_rad"]
    throttle = trajectory["throttle"]
    steering = trajectory["steering"]

    uphill_rad = torch.relu(along_track_slope_rad)
    downhill_rad = torch.relu(-along_track_slope_rad)
    crosstrack_rad = torch.abs(cross_track_slope_rad)

    length_cost = trajectory["length_3d"]

    def directional_slope_cost(rad: torch.Tensor, limit: float) -> torch.Tensor:
        cost = rad.div(limit).pow(2).mean()
        cost = cost + torch.relu(rad - limit).pow(2).div(limit**2).mean()
        return cost

    uphill_cost = directional_slope_cost(uphill_rad, uphill_limit)
    downhill_cost = directional_slope_cost(downhill_rad, downhill_limit)
    crosstrack_cost = directional_slope_cost(crosstrack_rad, crosstrack_limit)

    throttle_cost = throttle.pow(2).mean() + torch.relu(torch.abs(throttle) - 1.0).pow(2).mean()
    steering_cost = steering.div(steering_limit).pow(2).mean()
    steering_cost = (
        steering_cost
        + torch.relu(torch.abs(steering) - steering_limit).pow(2).div(steering_limit**2).mean()
    )

    path_xy = trajectory["path_xy"]
    path_second = path_xy[2:] - 2.0 * path_xy[1:-1] + path_xy[:-2]
    control_second = control_points[2:] - 2.0 * control_points[1:-1] + control_points[:-2]
    smooth_parts = []
    if path_second.numel():
        smooth_parts.append(path_second.pow(2).sum(dim=-1).mean())
    if control_second.numel():
        smooth_parts.append(control_second.pow(2).sum(dim=-1).mean())
    smoothness_cost = sum(smooth_parts) if smooth_parts else path_xy.new_tensor(0.0)

    return {
        "length_cost": length_cost,
        "uphill_cost": uphill_cost,
        "downhill_cost": downhill_cost,
        "crosstrack_cost": crosstrack_cost,
        "throttle_cost": throttle_cost,
        "steering_cost": steering_cost,
        "smoothness_cost": smoothness_cost,
    }


def make_diagnostics(
    trajectory: dict[str, torch.Tensor],
    terms: dict[str, torch.Tensor],
) -> TerrainAwarePlanDiagnostics:
    def scalar(name: str) -> float:
        return float(terms[name].detach().cpu().item())

    slope_deg = torch.rad2deg(trajectory["slope_mag"])
    uphill_deg = torch.rad2deg(torch.relu(trajectory["along_track_slope_rad"]))
    downhill_deg = torch.rad2deg(torch.relu(-trajectory["along_track_slope_rad"]))
    crosstrack_deg = torch.rad2deg(torch.abs(trajectory["cross_track_slope_rad"]))
    steering_deg = torch.rad2deg(torch.abs(trajectory["steering"]))
    throttle_abs = torch.abs(trajectory["throttle"])

    return TerrainAwarePlanDiagnostics(
        total_cost=scalar("total"),
        length_cost=scalar("length_cost"),
        uphill_cost=scalar("uphill_cost"),
        downhill_cost=scalar("downhill_cost"),
        crosstrack_cost=scalar("crosstrack_cost"),
        throttle_cost=scalar("throttle_cost"),
        steering_cost=scalar("steering_cost"),
        smoothness_cost=scalar("smoothness_cost"),
        length_3d_m=float(trajectory["length_3d"].detach().cpu().item()),
        mean_slope_deg=float(torch.mean(slope_deg).detach().cpu().item()),
        max_slope_deg=float(torch.max(slope_deg).detach().cpu().item()),
        mean_uphill_deg=float(torch.mean(uphill_deg).detach().cpu().item()),
        max_uphill_deg=float(torch.max(uphill_deg).detach().cpu().item()),
        mean_downhill_deg=float(torch.mean(downhill_deg).detach().cpu().item()),
        max_downhill_deg=float(torch.max(downhill_deg).detach().cpu().item()),
        mean_crosstrack_deg=float(torch.mean(crosstrack_deg).detach().cpu().item()),
        max_crosstrack_deg=float(torch.max(crosstrack_deg).detach().cpu().item()),
        mean_abs_throttle=float(torch.mean(throttle_abs).detach().cpu().item()),
        max_abs_throttle=float(torch.max(throttle_abs).detach().cpu().item()),
        mean_abs_steering_deg=float(torch.mean(steering_deg).detach().cpu().item()),
        max_abs_steering_deg=float(torch.max(steering_deg).detach().cpu().item()),
    )


def path_xyz(nemo: Any, path_xy: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xy = torch.as_tensor(path_xy, dtype=torch.float32, device=nemo.device)
        z = nemo.field.h(xy).detach().cpu().numpy().reshape(-1)
    return np.column_stack([path_xy, z]).astype(np.float32)


def _cost_normalizers(
    raw_costs: dict[str, torch.Tensor], *, enabled: bool
) -> dict[str, torch.Tensor]:
    normalizers: dict[str, torch.Tensor] = {}
    for name, value in raw_costs.items():
        if enabled:
            normalizers[name] = torch.clamp(torch.abs(value.detach()), min=1e-8)
        else:
            normalizers[name] = value.detach().new_tensor(1.0)
    return normalizers


def _weighted_total(
    raw_costs: dict[str, torch.Tensor],
    normalizers: dict[str, torch.Tensor],
    config: TerrainAwarePlannerConfig,
) -> torch.Tensor:
    return (
        float(config.w_length) * raw_costs["length_cost"] / normalizers["length_cost"]
        + float(config.w_uphill) * raw_costs["uphill_cost"] / normalizers["uphill_cost"]
        + float(config.w_downhill) * raw_costs["downhill_cost"] / normalizers["downhill_cost"]
        + float(config.w_crosstrack) * raw_costs["crosstrack_cost"] / normalizers["crosstrack_cost"]
        + float(config.w_throttle) * raw_costs["throttle_cost"] / normalizers["throttle_cost"]
        + float(config.w_steering) * raw_costs["steering_cost"] / normalizers["steering_cost"]
        + float(config.w_smoothness) * raw_costs["smoothness_cost"] / normalizers["smoothness_cost"]
    )


def _trajectory_to_numpy(trajectory: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    names = [
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
    ]
    return {name: trajectory[name].detach().cpu().numpy().astype(np.float32) for name in names}


def _gradient_by_index(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return torch.zeros_like(values)
    grad = torch.empty_like(values)
    grad[0] = values[1] - values[0]
    grad[-1] = values[-1] - values[-2]
    if values.shape[0] > 2:
        grad[1:-1] = 0.5 * (values[2:] - values[:-2])
    return grad


def _basis_function(i: int, degree: int, u: np.ndarray, knots: np.ndarray) -> np.ndarray:
    if degree == 0:
        out = ((u >= knots[i]) & (u < knots[i + 1])).astype(np.float64)
        if knots[i + 1] == 1.0:
            out[u == 1.0] = 1.0
        return out
    left_den = knots[i + degree] - knots[i]
    right_den = knots[i + degree + 1] - knots[i + 1]
    left: np.ndarray | float = 0.0
    right: np.ndarray | float = 0.0
    if left_den > 0.0:
        left = ((u - knots[i]) / left_den) * _basis_function(i, degree - 1, u, knots)
    if right_den > 0.0:
        right = ((knots[i + degree + 1] - u) / right_den) * _basis_function(
            i + 1, degree - 1, u, knots
        )
    return left + right


class _GaussianHillField(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bounds = ((-2.0, 2.0), (-1.5, 1.5))
        self.register_parameter("_dummy", torch.nn.Parameter(torch.empty(0), requires_grad=False))

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        r2 = xy[:, 0:1].pow(2) + xy[:, 1:2].pow(2)
        return 2.0 * torch.exp(-0.5 * r2 / (0.33**2))


class _FakeNemo:
    def __init__(self) -> None:
        self.field = _GaussianHillField()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


def _main() -> None:
    nemo = _FakeNemo()
    x = np.linspace(-1.65, 1.65, 80, dtype=np.float32)
    y = (0.15 * np.sin(np.linspace(0.0, np.pi, x.size))).astype(np.float32)
    initial = np.column_stack([x, y]).astype(np.float32)
    initial[0, 1] = 0.0
    initial[-1, 1] = 0.0
    cfg = TerrainAwarePlannerConfig(
        num_control_points=12,
        num_samples=220,
        num_iters=900,
        lr=3e-2,
        w_length=0.02,
        w_uphill=0.80,
        w_downhill=0.80,
        w_crosstrack=0.10,
        w_throttle=0.35,
        w_steering=0.1,
        w_smoothness=0.01,
        verbose=True,
    )
    result = optimize_terrain_aware_path(nemo, initial, cfg)
    print("initial:", result.initial_diagnostics)
    print("optimized:", result.optimized_diagnostics)
    max_lateral = float(np.max(np.abs(result.optimized_path_xy[:, 1])))
    assert result.optimized_diagnostics.total_cost < result.initial_diagnostics.total_cost
    # Using uphill_cost as a proxy for the old slope_cost check
    assert result.optimized_diagnostics.uphill_cost < result.initial_diagnostics.uphill_cost
    assert result.optimized_diagnostics.throttle_cost < result.initial_diagnostics.throttle_cost
    assert max_lateral > 0.3
    print(f"Sanity check passed. max lateral deviation: {max_lateral:.3f} m")


if __name__ == "__main__":
    _main()
