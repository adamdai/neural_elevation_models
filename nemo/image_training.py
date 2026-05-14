from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nemo.dem import CameraIntrinsics
from nemo.height_field import HeightField


@dataclass
class SampledRenderResult:
    rgb: Tensor
    hit_mask: Tensor
    xy_hit: Tensor
    depth: Tensor


@dataclass
class HorizonRenderResult:
    rows: Tensor
    hit_mask: Tensor
    columns: Tensor


def generate_camera_rays_for_pixels(
    intrinsics: CameraIntrinsics,
    world_T_camera,
    uv: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    world_T_camera_t = torch.as_tensor(world_T_camera, dtype=torch.float32, device=device)
    Rwc = world_T_camera_t[:3, :3]
    twc = world_T_camera_t[:3, 3]
    x = (uv[:, 0] + 0.5 - float(intrinsics.cx)) / float(intrinsics.fx)
    y = (uv[:, 1] + 0.5 - float(intrinsics.cy)) / float(intrinsics.fy)
    dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    dirs_cam = dirs_cam / torch.linalg.norm(dirs_cam, dim=-1, keepdim=True).clamp_min(1e-8)
    dirs_world = dirs_cam @ Rwc.T
    dirs_world = dirs_world / torch.linalg.norm(dirs_world, dim=-1, keepdim=True).clamp_min(1e-8)
    origins_world = twc.expand_as(dirs_world)
    return origins_world, dirs_world


def render_color_samples(
    field: HeightField,
    intrinsics: CameraIntrinsics,
    world_T_camera,
    uv: Tensor,
    *,
    t_near: float = 1.0,
    t_far: float | None = None,
    num_bracket_samples: int = 32,
    num_bisection_steps: int = 6,
    num_newton_steps: int = 1,
    background_color: tuple[float, float, float] = (0.5, 0.7, 0.9),
    differentiable_geometry: bool = False,
) -> SampledRenderResult:
    if not field.has_color():
        raise ValueError("Field must define color() for image supervision.")
    device = next(field.parameters(), torch.empty(0, device=torch.device("cpu"))).device
    uv = uv.to(device=device, dtype=torch.float32)
    origins, dirs = generate_camera_rays_for_pixels(intrinsics, world_T_camera, uv, device=device)
    if t_far is None:
        xy_span = max(
            float(field.bounds[0][1] - field.bounds[0][0]),
            float(field.bounds[1][1] - field.bounds[1][0]),
        )
        t_far = max(2.0 * xy_span, 100.0)

    interval = _xy_intersection_interval(
        origins,
        dirs,
        x_min=float(field.bounds[0][0]),
        x_max=float(field.bounds[0][1]),
        y_min=float(field.bounds[1][0]),
        y_max=float(field.bounds[1][1]),
        t_near=float(t_near),
        t_far=float(t_far),
    )
    n = origins.shape[0]
    hit_mask = torch.zeros((n,), dtype=torch.bool, device=device)
    xy_hit = torch.full((n, 2), float("nan"), dtype=torch.float32, device=device)
    depth = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    rgb = torch.empty((n, 3), dtype=torch.float32, device=device)
    rgb[:] = rgb.new_tensor(background_color)
    if interval is None:
        return SampledRenderResult(rgb=rgb, hit_mask=hit_mask, xy_hit=xy_hit, depth=depth)

    valid_mask, t_enter, t_exit = interval
    if not torch.any(valid_mask):
        return SampledRenderResult(rgb=rgb, hit_mask=hit_mask, xy_hit=xy_hit, depth=depth)

    active_origins = origins[valid_mask]
    active_dirs = dirs[valid_mask]
    active_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    sample_grid = torch.linspace(0.0, 1.0, int(num_bracket_samples), dtype=torch.float32, device=device)
    t_samples = t_enter[:, None] + (t_exit - t_enter)[:, None] * sample_grid[None, :]
    xyz_samples = active_origins[:, None, :] + t_samples[..., None] * active_dirs[:, None, :]
    flat_xy = xyz_samples[..., :2].reshape(-1, 2)

    with torch.no_grad():
        heights = field.h(flat_xy).reshape(xyz_samples.shape[:2])
    f_values = xyz_samples[..., 2] - heights
    sign_change = (f_values[:, :-1] > 0.0) & (f_values[:, 1:] <= 0.0)
    has_hit = sign_change.any(dim=1)
    if not torch.any(has_hit):
        return SampledRenderResult(rgb=rgb, hit_mask=hit_mask, xy_hit=xy_hit, depth=depth)

    first_idx = torch.argmax(sign_change.to(torch.int64), dim=1)
    hit_active_idx = torch.nonzero(has_hit, as_tuple=False).squeeze(-1)
    edge_idx = first_idx[hit_active_idx]
    hit_origins = active_origins[hit_active_idx]
    hit_dirs = active_dirs[hit_active_idx]
    hit_indices = active_indices[hit_active_idx]
    t_lower = t_samples[hit_active_idx, edge_idx]
    t_upper = t_samples[hit_active_idx, edge_idx + 1]

    for _ in range(int(num_bisection_steps)):
        t_mid = 0.5 * (t_lower + t_upper)
        xyz_mid = hit_origins + t_mid[:, None] * hit_dirs
        with torch.no_grad():
            h_mid = field.h(xyz_mid[:, :2]).squeeze(-1)
        f_mid = xyz_mid[:, 2] - h_mid
        move_lower = f_mid > 0.0
        t_lower = torch.where(move_lower, t_mid, t_lower)
        t_upper = torch.where(move_lower, t_upper, t_mid)

    t_hit = 0.5 * (t_lower + t_upper)
    if differentiable_geometry:
        t_hit = _refine_hit_t(
            field,
            hit_origins,
            hit_dirs,
            t_hit,
            t_lower,
            t_upper,
            num_newton_steps=int(num_newton_steps),
            create_graph=True,
        )
    else:
        t_hit = _refine_hit_t(
            field,
            hit_origins,
            hit_dirs,
            t_hit,
            t_lower,
            t_upper,
            num_newton_steps=int(num_newton_steps),
            create_graph=False,
        )

    hit_xy = hit_origins[:, :2] + t_hit[:, None] * hit_dirs[:, :2]
    hit_rgb = field.color(hit_xy).to(dtype=rgb.dtype)
    hit_mask[hit_indices] = True
    xy_hit[hit_indices] = hit_xy
    depth[hit_indices] = t_hit
    rgb[hit_indices] = hit_rgb
    return SampledRenderResult(rgb=rgb, hit_mask=hit_mask, xy_hit=xy_hit, depth=depth)


def render_horizon_samples(
    field: HeightField,
    intrinsics: CameraIntrinsics,
    world_T_camera,
    columns: Tensor | None = None,
    *,
    t_near: float = 1.0,
    t_far: float | None = None,
    num_bracket_samples: int = 32,
    num_bisection_steps: int = 6,
    num_newton_steps: int = 1,
    row_step: float = 8.0,
    row_bisection_steps: int = 6,
    horizon_refine_steps: int = 2,
    horizon_softmin_beta: float = 32.0,
    horizon_clearance_samples: int = 48,
    differentiable: bool = True,
) -> HorizonRenderResult:
    device = next(field.parameters(), torch.empty(0, device=torch.device("cpu"))).device
    if columns is None:
        columns = torch.arange(intrinsics.width, dtype=torch.float32, device=device)
    else:
        columns = columns.to(device=device, dtype=torch.float32).reshape(-1)
    num_columns = int(columns.numel())
    rows = torch.full((num_columns,), -1.0, dtype=torch.float32, device=device)
    hit_mask = torch.zeros((num_columns,), dtype=torch.bool, device=device)
    if num_columns == 0:
        return HorizonRenderResult(rows=rows, hit_mask=hit_mask, columns=columns)

    top_rows = torch.zeros_like(columns)
    bottom_rows = torch.full_like(columns, float(intrinsics.height - 1))
    top_hits = _render_hit_mask(
        field,
        intrinsics,
        world_T_camera,
        torch.stack([columns, top_rows], dim=-1),
        t_near=t_near,
        t_far=t_far,
        num_bracket_samples=num_bracket_samples,
        num_bisection_steps=num_bisection_steps,
        num_newton_steps=num_newton_steps,
    )
    rows[top_hits] = 0.0
    hit_mask |= top_hits

    unresolved_mask = ~top_hits
    if not torch.any(unresolved_mask):
        return HorizonRenderResult(rows=rows, hit_mask=hit_mask, columns=columns)

    bottom_hits = _render_hit_mask(
        field,
        intrinsics,
        world_T_camera,
        torch.stack([columns[unresolved_mask], bottom_rows[unresolved_mask]], dim=-1),
        t_near=t_near,
        t_far=t_far,
        num_bracket_samples=num_bracket_samples,
        num_bisection_steps=num_bisection_steps,
        num_newton_steps=num_newton_steps,
    )
    active_indices = torch.nonzero(unresolved_mask, as_tuple=False).squeeze(-1)
    hittable_indices = active_indices[bottom_hits]
    if hittable_indices.numel() == 0:
        return HorizonRenderResult(rows=rows, hit_mask=hit_mask, columns=columns)

    lower = torch.zeros((hittable_indices.numel(),), dtype=torch.float32, device=device)
    upper = torch.full_like(lower, float(intrinsics.height - 1))
    active_columns = columns[hittable_indices]
    stride = max(float(row_step), 1.0)
    previous_rows = torch.zeros_like(lower)
    bracket_found = torch.zeros((hittable_indices.numel(),), dtype=torch.bool, device=device)
    probe_row = stride
    while probe_row < float(intrinsics.height) and not torch.all(bracket_found):
        unresolved = ~bracket_found
        probe_rows = torch.full((int(unresolved.sum().item()),), probe_row, dtype=torch.float32, device=device)
        probe_hits = _render_hit_mask(
            field,
            intrinsics,
            world_T_camera,
            torch.stack([active_columns[unresolved], probe_rows], dim=-1),
            t_near=t_near,
            t_far=t_far,
            num_bracket_samples=num_bracket_samples,
            num_bisection_steps=num_bisection_steps,
            num_newton_steps=num_newton_steps,
        )
        unresolved_idx = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        if torch.any(probe_hits):
            hit_idx = unresolved_idx[probe_hits]
            lower[hit_idx] = previous_rows[hit_idx]
            upper[hit_idx] = probe_row
            bracket_found[hit_idx] = True
        if torch.any(~probe_hits):
            miss_idx = unresolved_idx[~probe_hits]
            previous_rows[miss_idx] = probe_row
        probe_row += stride

    unresolved = ~bracket_found
    if torch.any(unresolved):
        lower[unresolved] = previous_rows[unresolved]
        upper[unresolved] = float(intrinsics.height - 1)

    lower_rows = lower
    upper_rows = upper
    for _ in range(int(row_bisection_steps)):
        mid_rows = 0.5 * (lower_rows + upper_rows)
        mid_hits = _render_hit_mask(
            field,
            intrinsics,
            world_T_camera,
            torch.stack([active_columns, mid_rows], dim=-1),
            t_near=t_near,
            t_far=t_far,
            num_bracket_samples=num_bracket_samples,
            num_bisection_steps=num_bisection_steps,
            num_newton_steps=num_newton_steps,
        )
        upper_rows = torch.where(mid_hits, mid_rows, upper_rows)
        lower_rows = torch.where(mid_hits, lower_rows, mid_rows)

    horizon_rows = upper_rows
    if differentiable:
        horizon_rows = _refine_horizon_rows(
            field,
            intrinsics,
            world_T_camera,
            active_columns,
            lower_rows,
            upper_rows,
            t_near=t_near,
            t_far=t_far,
            steps=int(horizon_refine_steps),
            softmin_beta=float(horizon_softmin_beta),
            num_samples=int(horizon_clearance_samples),
        )
    rows[hittable_indices] = horizon_rows
    hit_mask[hittable_indices] = True
    return HorizonRenderResult(rows=rows, hit_mask=hit_mask, columns=columns)


def _refine_hit_t(
    field: HeightField,
    hit_origins: Tensor,
    hit_dirs: Tensor,
    t_hit: Tensor,
    t_lower: Tensor,
    t_upper: Tensor,
    *,
    num_newton_steps: int,
    create_graph: bool,
) -> Tensor:
    for _ in range(int(num_newton_steps)):
        hit_xy = hit_origins[:, :2] + t_hit[:, None] * hit_dirs[:, :2]
        z_ray = hit_origins[:, 2] + t_hit * hit_dirs[:, 2]
        h_hit, grad_hit = field.h_and_grad(hit_xy, create_graph=create_graph)
        h_hit = h_hit.squeeze(-1)
        f_hit = z_ray - h_hit
        f_prime = hit_dirs[:, 2] - torch.sum(grad_hit * hit_dirs[:, :2], dim=-1)
        safe_f_prime = torch.where(
            torch.abs(f_prime) < 1e-6,
            torch.sign(f_prime + 1e-6) * 1e-6,
            f_prime,
        )
        t_hit = torch.clamp(t_hit - f_hit / safe_f_prime, min=t_lower, max=t_upper)
    return t_hit


def _render_hit_mask(
    field: HeightField,
    intrinsics: CameraIntrinsics,
    world_T_camera,
    uv: Tensor,
    *,
    t_near: float,
    t_far: float | None,
    num_bracket_samples: int,
    num_bisection_steps: int,
    num_newton_steps: int,
) -> Tensor:
    device = uv.device
    origins, dirs = generate_camera_rays_for_pixels(intrinsics, world_T_camera, uv, device=device)
    if t_far is None:
        xy_span = max(
            float(field.bounds[0][1] - field.bounds[0][0]),
            float(field.bounds[1][1] - field.bounds[1][0]),
        )
        t_far = max(2.0 * xy_span, 100.0)
    interval = _xy_intersection_interval(
        origins,
        dirs,
        x_min=float(field.bounds[0][0]),
        x_max=float(field.bounds[0][1]),
        y_min=float(field.bounds[1][0]),
        y_max=float(field.bounds[1][1]),
        t_near=float(t_near),
        t_far=float(t_far),
    )
    hit_mask = torch.zeros((uv.shape[0],), dtype=torch.bool, device=device)
    if interval is None:
        return hit_mask
    valid_mask, t_enter, t_exit = interval
    if not torch.any(valid_mask):
        return hit_mask

    active_origins = origins[valid_mask]
    active_dirs = dirs[valid_mask]
    sample_grid = torch.linspace(0.0, 1.0, int(num_bracket_samples), dtype=torch.float32, device=device)
    t_samples = t_enter[:, None] + (t_exit - t_enter)[:, None] * sample_grid[None, :]
    xyz_samples = active_origins[:, None, :] + t_samples[..., None] * active_dirs[:, None, :]
    flat_xy = xyz_samples[..., :2].reshape(-1, 2)
    with torch.no_grad():
        heights = field.h(flat_xy).reshape(xyz_samples.shape[:2])
    f_values = xyz_samples[..., 2] - heights
    sign_change = (f_values[:, :-1] > 0.0) & (f_values[:, 1:] <= 0.0)
    has_hit = sign_change.any(dim=1)
    hit_mask[torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)[has_hit]] = True
    return hit_mask


def _refine_horizon_rows(
    field: HeightField,
    intrinsics: CameraIntrinsics,
    world_T_camera,
    columns: Tensor,
    lower_rows: Tensor,
    upper_rows: Tensor,
    *,
    t_near: float,
    t_far: float | None,
    steps: int,
    softmin_beta: float,
    num_samples: int,
) -> Tensor:
    rows = 0.5 * (lower_rows + upper_rows)
    if steps <= 0 or rows.numel() == 0:
        return rows
    for _ in range(int(steps)):
        rows = rows.clone().requires_grad_(True)
        clearance = _horizon_clearance(
            field,
            intrinsics,
            world_T_camera,
            columns,
            rows,
            t_near=t_near,
            t_far=t_far,
            softmin_beta=softmin_beta,
            num_samples=num_samples,
        )
        grad_rows = torch.autograd.grad(clearance.sum(), rows, create_graph=True)[0]
        safe_grad = torch.where(
            torch.abs(grad_rows) < 1e-6,
            torch.sign(grad_rows + 1e-6) * 1e-6,
            grad_rows,
        )
        rows = torch.clamp(rows - clearance / safe_grad, min=lower_rows, max=upper_rows)
    return rows


def _horizon_clearance(
    field: HeightField,
    intrinsics: CameraIntrinsics,
    world_T_camera,
    columns: Tensor,
    rows: Tensor,
    *,
    t_near: float,
    t_far: float | None,
    softmin_beta: float,
    num_samples: int,
) -> Tensor:
    uv = torch.stack([columns, rows], dim=-1)
    device = uv.device
    origins, dirs = generate_camera_rays_for_pixels(intrinsics, world_T_camera, uv, device=device)
    if t_far is None:
        xy_span = max(
            float(field.bounds[0][1] - field.bounds[0][0]),
            float(field.bounds[1][1] - field.bounds[1][0]),
        )
        t_far = max(2.0 * xy_span, 100.0)
    interval = _xy_intersection_interval(
        origins,
        dirs,
        x_min=float(field.bounds[0][0]),
        x_max=float(field.bounds[0][1]),
        y_min=float(field.bounds[1][0]),
        y_max=float(field.bounds[1][1]),
        t_near=float(t_near),
        t_far=float(t_far),
    )
    if interval is None:
        return torch.full_like(rows, float("inf"))
    valid_mask, t_enter, t_exit = interval
    clearance = torch.full_like(rows, float("inf"))
    if not torch.any(valid_mask):
        return clearance

    active_origins = origins[valid_mask]
    active_dirs = dirs[valid_mask]
    sample_grid = torch.linspace(0.0, 1.0, int(num_samples), dtype=torch.float32, device=device)
    t_samples = t_enter[:, None] + (t_exit - t_enter)[:, None] * sample_grid[None, :]
    xyz_samples = active_origins[:, None, :] + t_samples[..., None] * active_dirs[:, None, :]
    f_values = xyz_samples[..., 2] - field.h(xyz_samples[..., :2].reshape(-1, 2)).reshape(xyz_samples.shape[:2])
    softmin = -torch.logsumexp(-float(softmin_beta) * f_values, dim=1) / float(softmin_beta)
    clearance[torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)] = softmin
    return clearance


def _xy_intersection_interval(
    origins: Tensor,
    dirs: Tensor,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    t_near: float,
    t_far: float,
) -> tuple[Tensor, Tensor, Tensor] | None:
    bounds_min = origins.new_tensor([x_min, y_min])
    bounds_max = origins.new_tensor([x_max, y_max])
    origins_xy = origins[:, :2]
    dirs_xy = dirs[:, :2]
    parallel = torch.abs(dirs_xy) < 1e-8
    outside_parallel = parallel & ((origins_xy < bounds_min) | (origins_xy > bounds_max))
    valid = ~torch.any(outside_parallel, dim=-1)
    if not torch.any(valid):
        return None
    safe_dirs = torch.where(parallel, torch.ones_like(dirs_xy), dirs_xy)
    t0 = (bounds_min - origins_xy) / safe_dirs
    t1 = (bounds_max - origins_xy) / safe_dirs
    t_min_axis = torch.minimum(t0, t1)
    t_max_axis = torch.maximum(t0, t1)
    t_min_axis = torch.where(parallel, torch.full_like(t_min_axis, float("-inf")), t_min_axis)
    t_max_axis = torch.where(parallel, torch.full_like(t_max_axis, float("inf")), t_max_axis)
    t_enter = torch.maximum(t_min_axis.max(dim=-1).values, origins.new_full((origins.shape[0],), float(t_near)))
    t_exit = torch.minimum(t_max_axis.min(dim=-1).values, origins.new_full((origins.shape[0],), float(t_far)))
    valid = valid & (t_exit > t_enter)
    if not torch.any(valid):
        return None
    return valid, t_enter[valid], t_exit[valid]
