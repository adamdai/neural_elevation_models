from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from nemo.dem import CameraIntrinsics
from nemo.height_field import HeightField


@dataclass
class RenderResult:
    depth: np.ndarray
    hit_mask: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    rgb: np.ndarray | None = None


def look_at_pose(
    eye: np.ndarray,
    target: np.ndarray,
    *,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up_hint = np.asarray([0.0, 0.0, 1.0] if world_up is None else world_up, dtype=np.float32)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-8)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = -up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def generate_camera_rays(
    intrinsics: CameraIntrinsics,
    world_T_camera: np.ndarray,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    u = torch.arange(intrinsics.width, dtype=torch.float32, device=device)
    v = torch.arange(intrinsics.height, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    x = (uu + 0.5 - float(intrinsics.cx)) / float(intrinsics.fx)
    y = (vv + 0.5 - float(intrinsics.cy)) / float(intrinsics.fy)
    dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    dirs_cam = dirs_cam / torch.linalg.norm(dirs_cam, dim=-1, keepdim=True).clamp_min(1e-8)

    Rwc = torch.as_tensor(world_T_camera[:3, :3], dtype=torch.float32, device=device)
    twc = torch.as_tensor(world_T_camera[:3, 3], dtype=torch.float32, device=device)
    dirs_world = dirs_cam.reshape(-1, 3) @ Rwc.T
    dirs_world = dirs_world / torch.linalg.norm(dirs_world, dim=-1, keepdim=True).clamp_min(1e-8)
    origins_world = twc.expand_as(dirs_world)
    return origins_world, dirs_world


def render_height_field(
    field: HeightField,
    intrinsics: CameraIntrinsics,
    world_T_camera: np.ndarray,
    *,
    t_near: float = 1.0,
    t_far: float | None = None,
    num_bracket_samples: int = 64,
    num_bisection_steps: int = 12,
    num_newton_steps: int = 2,
    ray_batch_size: int = 16384,
) -> RenderResult:
    device = next(field.parameters(), torch.empty(0, device=torch.device("cpu"))).device
    origins, dirs = generate_camera_rays(intrinsics, world_T_camera, device=device)
    if t_far is None:
        xy_span = max(
            float(field.bounds[0][1] - field.bounds[0][0]),
            float(field.bounds[1][1] - field.bounds[1][0]),
        )
        t_far = max(2.0 * xy_span, 100.0)

    depth = torch.full((origins.shape[0],), float("nan"), dtype=torch.float32, device=device)
    hit_mask = torch.zeros((origins.shape[0],), dtype=torch.bool, device=device)
    points = torch.full((origins.shape[0], 3), float("nan"), dtype=torch.float32, device=device)
    normals = torch.full((origins.shape[0], 3), float("nan"), dtype=torch.float32, device=device)
    colors = None
    if field.has_color():
        colors = torch.full((origins.shape[0], 3), float("nan"), dtype=torch.float32, device=device)

    sample_grid = torch.linspace(0.0, 1.0, int(num_bracket_samples), dtype=torch.float32, device=device)
    x_min, x_max = float(field.bounds[0][0]), float(field.bounds[0][1])
    y_min, y_max = float(field.bounds[1][0]), float(field.bounds[1][1])

    for start in range(0, origins.shape[0], ray_batch_size):
        stop = min(start + ray_batch_size, origins.shape[0])
        batch_origins = origins[start:stop]
        batch_dirs = dirs[start:stop]
        interval = _xy_intersection_interval(
            batch_origins,
            batch_dirs,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            t_near=float(t_near),
            t_far=float(t_far),
        )
        if interval is None:
            continue
        interval_mask, t_enter, t_exit = interval
        if not torch.any(interval_mask):
            continue

        active_origins = batch_origins[interval_mask]
        active_dirs = batch_dirs[interval_mask]
        active_batch_indices = torch.nonzero(interval_mask, as_tuple=False).squeeze(-1)
        t_samples = t_enter[:, None] + (t_exit - t_enter)[:, None] * sample_grid[None, :]

        xyz_samples = active_origins[:, None, :] + t_samples[..., None] * active_dirs[:, None, :]
        xy_samples = xyz_samples[..., :2]
        z_ray = xyz_samples[..., 2]

        flat_xy = xy_samples.reshape(-1, 2)
        in_bounds = field.normalizer.contains(flat_xy).reshape(xy_samples.shape[:2])
        with torch.no_grad():
            heights = field.h(flat_xy).reshape(xy_samples.shape[:2])
        f_values = z_ray - heights

        lower_valid = in_bounds[:, :-1]
        upper_valid = in_bounds[:, 1:]
        sign_change = lower_valid & upper_valid & (f_values[:, :-1] > 0.0) & (f_values[:, 1:] <= 0.0)
        has_hit = sign_change.any(dim=1)
        if not torch.any(has_hit):
            continue

        first_idx = torch.argmax(sign_change.to(torch.int64), dim=1)
        hit_idx = torch.nonzero(has_hit, as_tuple=False).squeeze(-1)
        edge_idx = first_idx[hit_idx]

        t_lower = t_samples[hit_idx, edge_idx]
        t_upper = t_samples[hit_idx, edge_idx + 1]
        active_origins = active_origins[hit_idx]
        active_dirs = active_dirs[hit_idx]
        hit_batch_indices = active_batch_indices[hit_idx]

        for _ in range(int(num_bisection_steps)):
            t_mid = 0.5 * (t_lower + t_upper)
            xyz_mid = active_origins + t_mid[:, None] * active_dirs
            with torch.no_grad():
                h_mid = field.h(xyz_mid[:, :2]).squeeze(-1)
            f_mid = xyz_mid[:, 2] - h_mid
            move_lower = f_mid > 0.0
            t_lower = torch.where(move_lower, t_mid, t_lower)
            t_upper = torch.where(move_lower, t_upper, t_mid)

        t_hit = 0.5 * (t_lower + t_upper)
        h_hit = None
        grad_hit = None
        for _ in range(int(num_newton_steps)):
            xy_hit = active_origins[:, :2] + t_hit[:, None] * active_dirs[:, :2]
            z_ray_hit = active_origins[:, 2] + t_hit * active_dirs[:, 2]
            h_hit_full, grad_hit = field.h_and_grad(xy_hit, create_graph=False)
            h_hit = h_hit_full.squeeze(-1)
            f_hit = z_ray_hit - h_hit
            f_prime = active_dirs[:, 2] - torch.sum(grad_hit * active_dirs[:, :2], dim=-1)
            sign = torch.where(f_prime >= 0.0, 1.0, -1.0).to(dtype=f_prime.dtype, device=f_prime.device)
            safe_f_prime = torch.where(
                torch.abs(f_prime) < 1e-6,
                sign * 1e-6,
                f_prime,
            )
            update = f_hit / safe_f_prime
            t_hit = torch.clamp(t_hit - update, min=t_lower, max=t_upper)

        xyz_hit = active_origins + t_hit[:, None] * active_dirs
        if h_hit is None or grad_hit is None:
            _, grad_hit = field.h_and_grad(xyz_hit[:, :2], create_graph=False)
        else:
            xy_hit = active_origins[:, :2] + t_hit[:, None] * active_dirs[:, :2]
            _, grad_hit = field.h_and_grad(xy_hit, create_graph=False)
        normal_hit = torch.cat(
            [-grad_hit, torch.ones((grad_hit.shape[0], 1), dtype=grad_hit.dtype, device=device)], dim=-1
        )
        normal_hit = normal_hit / torch.linalg.norm(normal_hit, dim=-1, keepdim=True).clamp_min(1e-8)
        color_hit = None
        if colors is not None:
            with torch.no_grad():
                color_hit = field.color(xyz_hit[:, :2]).to(dtype=colors.dtype, device=colors.device)

        batch_hit_indices = hit_batch_indices + start
        depth[batch_hit_indices] = t_hit
        hit_mask[batch_hit_indices] = True
        points[batch_hit_indices] = xyz_hit
        normals[batch_hit_indices] = normal_hit
        if color_hit is not None:
            colors[batch_hit_indices] = color_hit

    h = intrinsics.height
    w = intrinsics.width
    return RenderResult(
        depth=depth.reshape(h, w).detach().cpu().numpy(),
        hit_mask=hit_mask.reshape(h, w).detach().cpu().numpy(),
        points=points.reshape(h, w, 3).detach().cpu().numpy(),
        normals=normals.reshape(h, w, 3).detach().cpu().numpy(),
        rgb=colors.reshape(h, w, 3).detach().cpu().numpy() if colors is not None else None,
    )


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
