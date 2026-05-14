from __future__ import annotations

from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import hypot

import numpy as np
import torch

from nemo.nemo import Nemo


@dataclass(frozen=True)
class PathPlanningConfig:
    astar_grid_resolution_x: int = 128
    astar_grid_resolution_y: int = 128
    astar_height_weight: float = 1.0
    astar_slope_weight: float = 1.5
    astar_step_weight: float = 1.0
    buffer_fraction: float = 0.2
    num_waypoints: int = 48
    optimize_iterations: int = 250
    optimize_lr: float = 2e-2
    terrain_height_weight: float = 1.0
    terrain_slope_weight: float = 0.5
    flatness_weight: float = 1.0
    smoothness_weight: float = 0.15
    length_weight: float = 0.05
    gravity: float = 9.81
    dt: float = 1.0
    batch_size: int = 65536


@dataclass(frozen=True)
class PathPlanningResult:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    initial_path_xy: np.ndarray
    optimized_path_xy: np.ndarray
    initial_path_xyz: np.ndarray
    optimized_path_xyz: np.ndarray
    astar_path_xy: np.ndarray
    cost_history: list[float]
    initial_objective: float
    final_objective: float

    def to_payload(self) -> dict[str, object]:
        return {
            "start_xy": list(self.start_xy),
            "goal_xy": list(self.goal_xy),
            "initial_path_xy": self.initial_path_xy.tolist(),
            "optimized_path_xy": self.optimized_path_xy.tolist(),
            "initial_path_xyz": self.initial_path_xyz.tolist(),
            "optimized_path_xyz": self.optimized_path_xyz.tolist(),
            "astar_path_xy": self.astar_path_xy.tolist(),
            "cost_history": [float(v) for v in self.cost_history],
            "initial_objective": float(self.initial_objective),
            "final_objective": float(self.final_objective),
        }


def buffered_corner_points(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    buffer_fraction: float = 0.2,
) -> tuple[tuple[float, float], tuple[float, float]]:
    buffer_fraction = float(np.clip(buffer_fraction, 0.0, 0.49))
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    x_margin = buffer_fraction * (x_max - x_min)
    y_margin = buffer_fraction * (y_max - y_min)
    start = (float(x_min + x_margin), float(y_min + y_margin))
    goal = (float(x_max - x_margin), float(y_max - y_margin))
    return start, goal


def grid_axes(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    resolution_x: int,
    resolution_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_axis = np.linspace(bounds[0][0], bounds[0][1], int(resolution_x), dtype=np.float32)
    y_axis = np.linspace(bounds[1][0], bounds[1][1], int(resolution_y), dtype=np.float32)
    return x_axis, y_axis


def sample_height_and_gradient_grid(
    nemo: Nemo,
    *,
    resolution_x: int,
    resolution_y: int,
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_axis, y_axis = grid_axes(
        nemo.field.bounds,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
    )
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    heights: list[np.ndarray] = []
    grads: list[np.ndarray] = []
    device = nemo.device
    for start in range(0, len(xy), int(batch_size)):
        batch = torch.as_tensor(xy[start : start + int(batch_size)], dtype=torch.float32, device=device)
        heights.append(nemo.h(batch).detach().cpu().numpy())
        grads.append(nemo.grad(batch).detach().cpu().numpy())

    z = np.concatenate(heights, axis=0).reshape(yy.shape)
    grad = np.concatenate(grads, axis=0).reshape(*yy.shape, 2)
    gx = grad[..., 0]
    gy = grad[..., 1]
    return xx, yy, z, gx, gy


def build_astar_cost_grid(
    z: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    height_weight: float = 1.0,
    slope_weight: float = 1.5,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)

    height_range = float(np.nanmax(z) - np.nanmin(z))
    slope = np.sqrt(gx**2 + gy**2)
    slope_range = float(np.nanmax(slope))

    if not np.isfinite(height_range) or height_range <= 1e-8:
        normalized_height = np.zeros_like(z, dtype=np.float32)
    else:
        normalized_height = (z - float(np.nanmin(z))) / height_range

    if not np.isfinite(slope_range) or slope_range <= 1e-8:
        normalized_slope = np.zeros_like(slope, dtype=np.float32)
    else:
        normalized_slope = slope / slope_range

    cost = 1.0 + float(height_weight) * normalized_height + float(slope_weight) * normalized_slope
    return np.clip(cost, 1e-4, None).astype(np.float32)


def astar_grid_path(
    cost_grid: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    *,
    step_weight: float = 1.0,
) -> np.ndarray:
    cost_grid = np.asarray(cost_grid, dtype=np.float32)
    height, width = cost_grid.shape
    start = (int(start_rc[0]), int(start_rc[1]))
    goal = (int(goal_rc[0]), int(goal_rc[1]))

    if not _rc_in_bounds(start, height, width):
        raise ValueError("Start index is out of bounds.")
    if not _rc_in_bounds(goal, height, width):
        raise ValueError("Goal index is out of bounds.")

    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heappush(open_heap, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, current = heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            return _reconstruct_rc_path(came_from, current)

        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            if not _rc_in_bounds(neighbor, height, width):
                continue
            step_length = hypot(float(dr), float(dc))
            transition_cost = 0.5 * (float(cost_grid[current]) + float(cost_grid[neighbor]))
            tentative = g_score[current] + float(step_weight) * step_length * transition_cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + _heuristic(neighbor, goal)
                heappush(open_heap, (priority, neighbor))

    raise RuntimeError("A* failed to find a path between the requested points.")


def resample_polyline(path_xy: np.ndarray, num_points: int) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError("Expected path_xy to have shape (N, 2).")
    if len(path_xy) < 2:
        raise ValueError("Need at least two points to resample a polyline.")

    num_points = max(int(num_points), 2)
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-8:
        return np.repeat(path_xy[:1], num_points, axis=0)

    target = np.linspace(0.0, total_length, num_points, dtype=np.float32)
    x = np.interp(target, cumulative, path_xy[:, 0]).astype(np.float32)
    y = np.interp(target, cumulative, path_xy[:, 1]).astype(np.float32)
    return np.column_stack([x, y]).astype(np.float32)


def path_xy_to_xyz(nemo: Nemo, path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    device = nemo.device
    xyz_batches: list[np.ndarray] = []
    for start in range(0, len(path_xy), 65536):
        batch = torch.as_tensor(path_xy[start : start + 65536], dtype=torch.float32, device=device)
        z = nemo.h(batch).detach().cpu().numpy()
        xyz_batches.append(np.concatenate([path_xy[start : start + 65536], z], axis=1))
    return np.concatenate(xyz_batches, axis=0).astype(np.float32)


def path_objective(
    nemo: Nemo,
    path_xy: torch.Tensor,
    *,
    terrain_height_weight: float = 1.0,
    terrain_slope_weight: float = 0.5,
    flatness_weight: float = 1.0,
    smoothness_weight: float = 0.15,
    length_weight: float = 0.05,
    gravity: float = 9.81,
    dt: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if path_xy.ndim != 2 or path_xy.shape[-1] != 2:
        raise ValueError("Expected path_xy to have shape (N, 2).")
    if path_xy.shape[0] < 3:
        raise ValueError("Path optimization requires at least three waypoints.")

    xy = path_xy
    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    heights, grad = _finite_difference_height_and_grad(
        nemo,
        xy,
        eps=_finite_difference_eps(nemo),
    )
    controls = differential_flatness_controls(xy, grad=grad, gravity=gravity, dt=dt)

    segment_lengths = torch.linalg.norm(xy[1:] - xy[:-1], dim=-1)
    curvature = xy[2:] - 2.0 * xy[1:-1] + xy[:-2]

    metrics = {
        "terrain_height": heights[1:-1].mean(),
        "terrain_slope": torch.linalg.norm(grad, dim=-1)[1:-1].mean(),
        "flatness": controls[1:-1].pow(2).sum(dim=-1).mean(),
        "smoothness": curvature.pow(2).sum(dim=-1).mean(),
        "length": segment_lengths.mean(),
    }
    loss = (
        float(terrain_height_weight) * metrics["terrain_height"]
        + float(terrain_slope_weight) * metrics["terrain_slope"]
        + float(flatness_weight) * metrics["flatness"]
        + float(smoothness_weight) * metrics["smoothness"]
        + float(length_weight) * metrics["length"]
    )
    return loss, metrics


def differential_flatness_controls(
    path_xy: torch.Tensor,
    *,
    grad: torch.Tensor | None = None,
    gravity: float = 9.81,
    dt: float = 1.0,
) -> torch.Tensor:
    if path_xy.ndim != 2 or path_xy.shape[-1] != 2:
        raise ValueError("Expected path_xy to have shape (N, 2).")
    if path_xy.shape[0] < 3:
        raise ValueError("Need at least three waypoints to estimate differential flatness.")

    x = path_xy[:, 0]
    y = path_xy[:, 1]

    xdot = torch.cat([((x[1] - x[0]) / dt).unsqueeze(0), torch.diff(x) / dt])
    ydot = torch.cat([((y[1] - y[0]) / dt).unsqueeze(0), torch.diff(y) / dt])
    xddot = torch.cat([((xdot[1] - xdot[0]) / dt).unsqueeze(0), torch.diff(xdot) / dt])
    yddot = torch.cat([((ydot[1] - ydot[0]) / dt).unsqueeze(0), torch.diff(ydot) / dt])

    v = torch.sqrt(xdot**2 + ydot**2).clamp_min(1e-4)
    theta = torch.atan2(ydot, xdot)

    if grad is None:
        raise ValueError("grad must be provided by path_objective for differentiable planning.")
    psi = torch.atan2(grad[:, 1], grad[:, 0])
    alpha = torch.atan(torch.linalg.norm(grad, dim=-1))

    phi = alpha * torch.cos(theta - psi)
    g_eff = float(gravity) * torch.sin(phi)

    j_inv = torch.stack(
        [
            v * torch.cos(theta),
            v * torch.sin(theta),
            -torch.sin(theta),
            torch.cos(theta),
        ],
        dim=-1,
    ).reshape(-1, 2, 2) / v.view(-1, 1, 1)
    b = torch.stack(
        [
            xddot + g_eff * torch.cos(theta),
            yddot + g_eff * torch.sin(theta),
        ],
        dim=-1,
    )
    return torch.bmm(j_inv, b.unsqueeze(-1)).squeeze(-1)


def _finite_difference_eps(nemo: Nemo) -> float:
    x_bounds = float(nemo.field.bounds[0][1] - nemo.field.bounds[0][0])
    y_bounds = float(nemo.field.bounds[1][1] - nemo.field.bounds[1][0])
    return max(min(x_bounds, y_bounds) / 512.0, 1e-3)


def _finite_difference_height_and_grad(
    nemo: Nemo,
    xy: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError("Expected xy to have shape (N, 2).")

    offset_x = xy.new_tensor([float(eps), 0.0])
    offset_y = xy.new_tensor([0.0, float(eps)])

    z_center = nemo.field.h(xy).squeeze(-1)
    z_x_plus = nemo.field.h(xy + offset_x).squeeze(-1)
    z_x_minus = nemo.field.h(xy - offset_x).squeeze(-1)
    z_y_plus = nemo.field.h(xy + offset_y).squeeze(-1)
    z_y_minus = nemo.field.h(xy - offset_y).squeeze(-1)

    grad_x = (z_x_plus - z_x_minus) / (2.0 * float(eps))
    grad_y = (z_y_plus - z_y_minus) / (2.0 * float(eps))
    grad = torch.stack([grad_x, grad_y], dim=-1)
    return z_center, grad


def optimize_flat_path(
    nemo: Nemo,
    path_xy: np.ndarray,
    *,
    iterations: int = 250,
    lr: float = 2e-2,
    terrain_height_weight: float = 1.0,
    terrain_slope_weight: float = 0.5,
    flatness_weight: float = 1.0,
    smoothness_weight: float = 0.15,
    length_weight: float = 0.05,
    gravity: float = 9.81,
    dt: float = 1.0,
) -> tuple[np.ndarray, list[float], float, float]:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError("Expected path_xy to have shape (N, 2).")
    if len(path_xy) < 3:
        raise ValueError("Need at least three points to optimize a path.")

    device = nemo.device
    start = torch.as_tensor(path_xy[0], dtype=torch.float32, device=device)
    goal = torch.as_tensor(path_xy[-1], dtype=torch.float32, device=device)
    interior = torch.nn.Parameter(
        torch.as_tensor(path_xy[1:-1], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([interior], lr=float(lr))

    cost_history: list[float] = []
    initial_path = torch.as_tensor(path_xy, dtype=torch.float32, device=device)
    initial_loss, _ = path_objective(
        nemo,
        initial_path,
        terrain_height_weight=terrain_height_weight,
        terrain_slope_weight=terrain_slope_weight,
        flatness_weight=flatness_weight,
        smoothness_weight=smoothness_weight,
        length_weight=length_weight,
        gravity=gravity,
        dt=dt,
    )
    initial_objective = float(initial_loss.detach().cpu().item())

    x_min, x_max = float(nemo.field.bounds[0][0]), float(nemo.field.bounds[0][1])
    y_min, y_max = float(nemo.field.bounds[1][0]), float(nemo.field.bounds[1][1])

    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        path = torch.cat([start[None], interior, goal[None]], dim=0)
        loss, _ = path_objective(
            nemo,
            path,
            terrain_height_weight=terrain_height_weight,
            terrain_slope_weight=terrain_slope_weight,
            flatness_weight=flatness_weight,
            smoothness_weight=smoothness_weight,
            length_weight=length_weight,
            gravity=gravity,
            dt=dt,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            interior[:, 0].clamp_(x_min, x_max)
            interior[:, 1].clamp_(y_min, y_max)
        cost_history.append(float(loss.detach().cpu().item()))

    optimized_path = torch.cat([start[None], interior.detach(), goal[None]], dim=0)
    optimized_loss, _ = path_objective(
        nemo,
        optimized_path,
        terrain_height_weight=terrain_height_weight,
        terrain_slope_weight=terrain_slope_weight,
        flatness_weight=flatness_weight,
        smoothness_weight=smoothness_weight,
        length_weight=length_weight,
        gravity=gravity,
        dt=dt,
    )
    final_objective = float(optimized_loss.detach().cpu().item())
    return optimized_path.detach().cpu().numpy().astype(np.float32), cost_history, initial_objective, final_objective


def plan_path(
    nemo: Nemo,
    *,
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    config: PathPlanningConfig | None = None,
) -> PathPlanningResult:
    cfg = config or PathPlanningConfig()
    if start_xy is None or goal_xy is None:
        default_start, default_goal = buffered_corner_points(
            nemo.field.bounds,
            buffer_fraction=cfg.buffer_fraction,
        )
        start_xy = start_xy or default_start
        goal_xy = goal_xy or default_goal

    xx, yy, z, gx, gy = sample_height_and_gradient_grid(
        nemo,
        resolution_x=cfg.astar_grid_resolution_x,
        resolution_y=cfg.astar_grid_resolution_y,
        batch_size=cfg.batch_size,
    )
    cost_grid = build_astar_cost_grid(
        z,
        gx,
        gy,
        height_weight=cfg.astar_height_weight,
        slope_weight=cfg.astar_slope_weight,
    )

    start_rc = _xy_to_rc(start_xy, xx[0, :], yy[:, 0])
    goal_rc = _xy_to_rc(goal_xy, xx[0, :], yy[:, 0])
    astar_rc = astar_grid_path(cost_grid, start_rc, goal_rc, step_weight=cfg.astar_step_weight)

    astar_path_xy = np.column_stack([xx[astar_rc[:, 0], astar_rc[:, 1]], yy[astar_rc[:, 0], astar_rc[:, 1]]]).astype(
        np.float32
    )
    astar_path_xy[0] = np.asarray(start_xy, dtype=np.float32)
    astar_path_xy[-1] = np.asarray(goal_xy, dtype=np.float32)

    initial_path_xy = resample_polyline(astar_path_xy, cfg.num_waypoints)
    initial_path_xy[0] = np.asarray(start_xy, dtype=np.float32)
    initial_path_xy[-1] = np.asarray(goal_xy, dtype=np.float32)

    optimized_path_xy, cost_history, initial_objective, final_objective = optimize_flat_path(
        nemo,
        initial_path_xy,
        iterations=cfg.optimize_iterations,
        lr=cfg.optimize_lr,
        terrain_height_weight=cfg.terrain_height_weight,
        terrain_slope_weight=cfg.terrain_slope_weight,
        flatness_weight=cfg.flatness_weight,
        smoothness_weight=cfg.smoothness_weight,
        length_weight=cfg.length_weight,
        gravity=cfg.gravity,
        dt=cfg.dt,
    )
    optimized_path_xy[0] = np.asarray(start_xy, dtype=np.float32)
    optimized_path_xy[-1] = np.asarray(goal_xy, dtype=np.float32)

    initial_path_xyz = path_xy_to_xyz(nemo, initial_path_xy)
    optimized_path_xyz = path_xy_to_xyz(nemo, optimized_path_xy)

    return PathPlanningResult(
        start_xy=(float(start_xy[0]), float(start_xy[1])),
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        initial_path_xy=initial_path_xy,
        optimized_path_xy=optimized_path_xy,
        initial_path_xyz=initial_path_xyz,
        optimized_path_xyz=optimized_path_xyz,
        astar_path_xy=astar_path_xy,
        cost_history=cost_history,
        initial_objective=initial_objective,
        final_objective=final_objective,
    )


def compute_path_metrics(
    nemo: Nemo,
    path_xy: np.ndarray,
    *,
    terrain_height_weight: float = 1.0,
    terrain_slope_weight: float = 0.5,
    flatness_weight: float = 1.0,
    smoothness_weight: float = 0.15,
    length_weight: float = 0.05,
    gravity: float = 9.81,
    dt: float = 1.0,
) -> dict[str, float]:
    path = torch.as_tensor(path_xy, dtype=torch.float32, device=nemo.device)
    loss, metrics = path_objective(
        nemo,
        path,
        terrain_height_weight=terrain_height_weight,
        terrain_slope_weight=terrain_slope_weight,
        flatness_weight=flatness_weight,
        smoothness_weight=smoothness_weight,
        length_weight=length_weight,
        gravity=gravity,
        dt=dt,
    )
    payload = {name: float(value.detach().cpu().item()) for name, value in metrics.items()}
    payload["objective"] = float(loss.detach().cpu().item())
    payload["path_length"] = float(torch.linalg.norm(path[1:] - path[:-1], dim=-1).sum().detach().cpu().item())
    return payload


def _rc_in_bounds(rc: tuple[int, int], height: int, width: int) -> bool:
    return 0 <= rc[0] < height and 0 <= rc[1] < width


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _reconstruct_rc_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> np.ndarray:
    path: list[tuple[int, int]] = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def _xy_to_rc(
    xy: tuple[float, float],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    c = int(np.clip(np.argmin(np.abs(x_axis - x)), 0, len(x_axis) - 1))
    r = int(np.clip(np.argmin(np.abs(y_axis - y)), 0, len(y_axis) - 1))
    return r, c


def path_summary(result: PathPlanningResult) -> dict[str, object]:
    return {
        "start_xy": list(result.start_xy),
        "goal_xy": list(result.goal_xy),
        "initial_objective": result.initial_objective,
        "final_objective": result.final_objective,
        "cost_history": [float(v) for v in result.cost_history],
        "initial_path_length": float(np.sum(np.linalg.norm(np.diff(result.initial_path_xy, axis=0), axis=1))),
        "optimized_path_length": float(np.sum(np.linalg.norm(np.diff(result.optimized_path_xy, axis=0), axis=1))),
    }


def config_payload(config: PathPlanningConfig) -> dict[str, object]:
    return asdict(config)
