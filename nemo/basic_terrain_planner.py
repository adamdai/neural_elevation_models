from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nemo.nemo import Nemo


@dataclass(frozen=True)
class BasicTerrainPlannerConfig:
    """Simple differentiable terrain planner for sanity tests.

    The planner optimizes a small set of 2D B-spline control points. The cost is
    deliberately basic and interpretable: path length, terrain slope exposure,
    and planar control effort. It is meant as a clean baseline, not a full rover
    dynamics model.
    """

    num_control_points: int = 20
    num_samples: int = 180
    num_iters: int = 1200
    lr: float = 2e-2
    w_length: float = 0.0
    w_slope: float = 0.0
    w_control: float = 0.0
    w_control_points: float = 0.0
    finite_difference_eps: float | None = None
    bounds_margin: float = 0.0
    verbose: bool = False


@dataclass(frozen=True)
class BasicTerrainPlanDiagnostics:
    total_cost: float
    length_3d_m: float
    slope_cost: float
    mean_slope: float
    max_slope: float
    control_effort: float
    control_point_effort: float
    max_curvature_proxy: float


@dataclass(frozen=True)
class BasicTerrainPlanResult:
    initial_path_xy: np.ndarray
    optimized_path_xy: np.ndarray
    control_points_initial: np.ndarray
    control_points_optimized: np.ndarray
    initial_diagnostics: BasicTerrainPlanDiagnostics
    optimized_diagnostics: BasicTerrainPlanDiagnostics
    cost_history: list[float]


def optimize_basic_terrain_path(
    nemo: Nemo,
    initial_path_xy: np.ndarray,
    config: BasicTerrainPlannerConfig | None = None,
) -> BasicTerrainPlanResult:
    """Optimize a smooth 2D path over a NEMo height field."""
    cfg = config or BasicTerrainPlannerConfig()
    seed = _validate_path(initial_path_xy)
    control_initial = _initialize_control_points(seed, cfg.num_control_points)
    basis = _bspline_basis(
        num_control_points=control_initial.shape[0],
        num_samples=max(int(cfg.num_samples), 2),
        degree=min(3, control_initial.shape[0] - 1),
    )

    device = nemo.device
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    control_t = torch.as_tensor(control_initial, dtype=torch.float32, device=device)
    start = control_t[0].detach().clone()
    goal = control_t[-1].detach().clone()
    interior = torch.nn.Parameter(control_t[1:-1].detach().clone())
    optimizer = torch.optim.Adam([interior], lr=float(cfg.lr))

    initial_path_t = _sample_path(control_t, basis_t)
    initial_cost, initial_terms = _basic_objective(nemo, initial_path_t, control_t, cfg)
    best_total = float(initial_cost.detach().cpu().item())
    best_terms = {name: value.detach().clone() for name, value in initial_terms.items()}
    best_interior = interior.detach().clone()
    history: list[float] = []
    x_min, x_max = (
        float(nemo.field.bounds[0][0]) + cfg.bounds_margin,
        float(nemo.field.bounds[0][1]) - cfg.bounds_margin,
    )
    y_min, y_max = (
        float(nemo.field.bounds[1][0]) + cfg.bounds_margin,
        float(nemo.field.bounds[1][1]) - cfg.bounds_margin,
    )

    for idx in range(int(cfg.num_iters)):
        optimizer.zero_grad(set_to_none=True)
        controls = torch.cat([start[None], interior, goal[None]], dim=0)
        path = _sample_path(controls, basis_t)
        total, _ = _basic_objective(nemo, path, controls, cfg)
        total.backward()
        optimizer.step()
        with torch.no_grad():
            interior[:, 0].clamp_(x_min, x_max)
            interior[:, 1].clamp_(y_min, y_max)
            candidate_controls = torch.cat([start[None], interior, goal[None]], dim=0)
            candidate_path = _sample_path(candidate_controls, basis_t)
            candidate_total, candidate_terms = _basic_objective(
                nemo, candidate_path, candidate_controls, cfg
            )
            candidate_value = float(candidate_total.detach().cpu().item())
            if candidate_value < best_total:
                best_total = candidate_value
                best_terms = {
                    name: value.detach().clone() for name, value in candidate_terms.items()
                }
                best_interior = interior.detach().clone()
        history.append(candidate_value)
        if cfg.verbose and (idx + 1) % 100 == 0:
            print(f"iter {idx + 1:04d}: cost={candidate_value:.4f}")

    optimized_controls = torch.cat([start[None], best_interior, goal[None]], dim=0)
    optimized_path = _sample_path(optimized_controls, basis_t)
    return BasicTerrainPlanResult(
        initial_path_xy=initial_path_t.detach().cpu().numpy().astype(np.float32),
        optimized_path_xy=optimized_path.detach().cpu().numpy().astype(np.float32),
        control_points_initial=control_initial.astype(np.float32),
        control_points_optimized=optimized_controls.detach().cpu().numpy().astype(np.float32),
        initial_diagnostics=_diagnostics_from_terms(initial_terms),
        optimized_diagnostics=_diagnostics_from_terms(best_terms),
        cost_history=history,
    )


def path_xyz(nemo: Nemo, path_xy: np.ndarray) -> np.ndarray:
    """Lift an optimized path onto the terrain surface."""
    with torch.no_grad():
        xy = torch.as_tensor(path_xy, dtype=torch.float32, device=nemo.device)
        z = nemo.h(xy).detach().cpu().numpy().reshape(-1)
    return np.column_stack([path_xy, z]).astype(np.float32)


def _basic_objective(
    nemo: Nemo,
    path_xy: torch.Tensor,
    control_points: torch.Tensor,
    cfg: BasicTerrainPlannerConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    z = nemo.field.h(path_xy).squeeze(-1)
    dxy = path_xy[1:] - path_xy[:-1]
    dz = z[1:] - z[:-1]
    ds_xy = torch.linalg.norm(dxy, dim=-1).clamp_min(1e-8)
    ds_3d = torch.sqrt(ds_xy.pow(2) + dz.pow(2)).clamp_min(1e-8)
    length_3d = ds_3d.sum()

    grad = _finite_difference_grad(nemo, path_xy, cfg)
    slope_mag = torch.linalg.norm(grad, dim=-1)
    segment_slope = 0.5 * (slope_mag[1:] + slope_mag[:-1])
    slope_cost = torch.sum(segment_slope.pow(2) * ds_xy) / torch.sum(ds_xy).clamp_min(1e-8)

    second = path_xy[2:] - 2.0 * path_xy[1:-1] + path_xy[:-2]
    first = 0.5 * (path_xy[2:] - path_xy[:-2])
    first_norm = torch.linalg.norm(first, dim=-1).clamp_min(1e-5)
    curvature_proxy = torch.linalg.norm(second, dim=-1) / first_norm.pow(2)
    control_effort = (
        curvature_proxy.pow(2).mean() if curvature_proxy.numel() else path_xy.new_tensor(0.0)
    )

    control_second = control_points[2:] - 2.0 * control_points[1:-1] + control_points[:-2]
    control_point_effort = (
        control_second.pow(2).sum(dim=-1).mean()
        if control_second.numel()
        else path_xy.new_tensor(0.0)
    )

    total = (
        float(cfg.w_length) * length_3d
        + float(cfg.w_slope) * slope_cost
        + float(cfg.w_control) * control_effort
        + float(cfg.w_control_points) * control_point_effort
    )
    terms = {
        "total": total,
        "length_3d": length_3d,
        "slope_cost": slope_cost,
        "mean_slope": slope_mag.mean(),
        "max_slope": slope_mag.max(),
        "control_effort": control_effort,
        "control_point_effort": control_point_effort,
        "max_curvature_proxy": curvature_proxy.max()
        if curvature_proxy.numel()
        else path_xy.new_tensor(0.0),
    }
    return total, terms


def _sample_path(control_points: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    path = basis @ control_points
    path = path.clone()
    path[0] = control_points[0]
    path[-1] = control_points[-1]
    return path


def _finite_difference_grad(
    nemo: Nemo,
    xy: torch.Tensor,
    cfg: BasicTerrainPlannerConfig,
) -> torch.Tensor:
    if cfg.finite_difference_eps is None:
        x_range = float(nemo.field.bounds[0][1] - nemo.field.bounds[0][0])
        y_range = float(nemo.field.bounds[1][1] - nemo.field.bounds[1][0])
        eps = max(min(x_range, y_range) / 512.0, 1e-3)
    else:
        eps = float(cfg.finite_difference_eps)
    dx = xy.new_tensor([eps, 0.0])
    dy = xy.new_tensor([0.0, eps])
    gx = (nemo.field.h(xy + dx).squeeze(-1) - nemo.field.h(xy - dx).squeeze(-1)) / (2.0 * eps)
    gy = (nemo.field.h(xy + dy).squeeze(-1) - nemo.field.h(xy - dy).squeeze(-1)) / (2.0 * eps)
    return torch.stack([gx, gy], dim=-1)


def _diagnostics_from_terms(terms: dict[str, torch.Tensor]) -> BasicTerrainPlanDiagnostics:
    def scalar(name: str) -> float:
        return float(terms[name].detach().cpu().item())

    return BasicTerrainPlanDiagnostics(
        total_cost=scalar("total"),
        length_3d_m=scalar("length_3d"),
        slope_cost=scalar("slope_cost"),
        mean_slope=scalar("mean_slope"),
        max_slope=scalar("max_slope"),
        control_effort=scalar("control_effort"),
        control_point_effort=scalar("control_point_effort"),
        max_curvature_proxy=scalar("max_curvature_proxy"),
    )


def _initialize_control_points(path_xy: np.ndarray, num_control_points: int) -> np.ndarray:
    n_interior = max(int(num_control_points), 0)
    total_controls = n_interior + 2
    cumulative = _cumulative_lengths(path_xy)
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(path_xy[:1], total_controls, axis=0)
    targets = np.linspace(0.0, total, total_controls)
    x = np.interp(targets, cumulative, path_xy[:, 0])
    y = np.interp(targets, cumulative, path_xy[:, 1])
    controls = np.column_stack([x, y]).astype(np.float32)
    controls[0] = path_xy[0]
    controls[-1] = path_xy[-1]
    return controls


def _bspline_basis(num_control_points: int, num_samples: int, degree: int = 3) -> np.ndarray:
    if num_control_points < 2:
        raise ValueError("At least two control points are required.")
    degree = int(min(max(degree, 1), num_control_points - 1))
    knot_count = num_control_points + degree + 1
    knots = np.zeros(knot_count, dtype=np.float64)
    knots[degree : num_control_points + 1] = np.linspace(0.0, 1.0, num_control_points - degree + 1)
    knots[num_control_points + 1 :] = 1.0
    u = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    basis = np.column_stack(
        [_basis_function(i, degree, u, knots) for i in range(num_control_points)]
    )
    basis[0, :] = 0.0
    basis[0, 0] = 1.0
    basis[-1, :] = 0.0
    basis[-1, -1] = 1.0
    row_sum = basis.sum(axis=1, keepdims=True)
    return (basis / np.clip(row_sum, 1e-12, None)).astype(np.float32)


def _basis_function(i: int, degree: int, u: np.ndarray, knots: np.ndarray) -> np.ndarray:
    if degree == 0:
        out = ((u >= knots[i]) & (u < knots[i + 1])).astype(np.float64)
        if knots[i + 1] == 1.0:
            out[u == 1.0] = 1.0
        return out
    left_den = knots[i + degree] - knots[i]
    right_den = knots[i + degree + 1] - knots[i + 1]
    left = 0.0
    right = 0.0
    if left_den > 0.0:
        left = ((u - knots[i]) / left_den) * _basis_function(i, degree - 1, u, knots)
    if right_den > 0.0:
        right = ((knots[i + degree + 1] - u) / right_den) * _basis_function(
            i + 1, degree - 1, u, knots
        )
    return left + right


def _cumulative_lengths(path_xy: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _validate_path(path_xy: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError("Expected path_xy with shape (N, 2), N >= 2.")
    if not np.all(np.isfinite(path)):
        raise ValueError("Path contains non-finite values.")
    return path
