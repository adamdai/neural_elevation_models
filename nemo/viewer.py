from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh

from nemo.dem import CameraIntrinsics
from nemo.nemo import Nemo
from nemo.particle_sim import sample_height_field_grid
from nemo.rendering import RenderResult, look_at_pose


@dataclass(frozen=True)
class CameraViewState:
    position: np.ndarray
    look_at: np.ndarray
    up_direction: np.ndarray
    fov: float
    image_width: int
    image_height: int


def intrinsics_from_view_state(view: CameraViewState, *, resolution_scale: float = 1.0) -> CameraIntrinsics:
    scale = float(np.clip(resolution_scale, 0.05, 1.0))
    width = max(int(round(int(view.image_width) * scale)), 32)
    height = max(int(round(int(view.image_height) * scale)), 32)
    fy = 0.5 * float(height) / max(np.tan(0.5 * float(view.fov)), 1e-6)
    fx = fy
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=0.5 * float(width),
        cy=0.5 * float(height),
    )


def pose_from_view_state(view: CameraViewState) -> np.ndarray:
    return look_at_pose(
        np.asarray(view.position, dtype=np.float32),
        np.asarray(view.look_at, dtype=np.float32),
        world_up=np.asarray(view.up_direction, dtype=np.float32),
    )


def render_rgb(
    nemo: Nemo,
    view: CameraViewState,
    *,
    resolution_scale: float = 0.5,
    t_near: float = 1.0,
    t_far: float | None = None,
    num_bracket_samples: int = 96,
    num_bisection_steps: int = 14,
    num_newton_steps: int = 2,
    ray_batch_size: int = 8192,
    light_direction: tuple[float, float, float] = (-0.35, -0.25, 0.9),
    ambient: float = 0.35,
) -> tuple[np.ndarray, RenderResult]:
    intrinsics = intrinsics_from_view_state(view, resolution_scale=resolution_scale)
    world_T_camera = pose_from_view_state(view)
    render = nemo.render_view(
        intrinsics,
        world_T_camera,
        t_near=t_near,
        t_far=t_far,
        num_bracket_samples=num_bracket_samples,
        num_bisection_steps=num_bisection_steps,
        num_newton_steps=num_newton_steps,
        ray_batch_size=ray_batch_size,
    )
    rgb = shade_render(
        render,
        light_direction=light_direction,
        ambient=ambient,
        z_bounds=estimate_z_bounds(nemo),
    )
    return rgb, render


def estimate_z_bounds(
    nemo: Nemo,
    *,
    resolution_x: int = 96,
    resolution_y: int = 96,
    batch_size: int = 65536,
) -> tuple[float, float]:
    _, _, zz = sample_height_field_grid(
        nemo,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        batch_size=batch_size,
    )
    finite = np.isfinite(zz)
    if not np.any(finite):
        return 0.0, 1.0
    z = zz[finite]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max - z_min < 1e-6:
        z_max = z_min + 1.0
    return z_min, z_max


def default_camera(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    z_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = z_bounds
    center = np.array(
        [
            0.5 * (x_min + x_max),
            0.5 * (y_min + y_max),
            0.5 * (z_min + z_max),
        ],
        dtype=np.float32,
    )
    xy_span = max(x_max - x_min, y_max - y_min, 1e-3)
    z_span = max(z_max - z_min, 1e-3)
    eye = center + np.array(
        [-0.9 * xy_span, -0.9 * xy_span, 0.75 * xy_span + 1.5 * z_span],
        dtype=np.float32,
    )
    return eye, center


def preview_point_cloud(
    nemo: Nemo,
    *,
    resolution_x: int = 180,
    resolution_y: int = 180,
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy, zz = sample_height_field_grid(
        nemo,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        batch_size=batch_size,
    )
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    if nemo.field.has_color():
        colors = _sample_field_colors(nemo, xx, yy, batch_size=batch_size)
    else:
        colors = _height_colors(zz, estimate_z_bounds(nemo)).reshape(-1, 3)
    return points, colors


def sample_height_field_mesh(
    nemo: Nemo,
    *,
    resolution_x: int = 220,
    resolution_y: int = 220,
    batch_size: int = 65536,
) -> trimesh.Trimesh:
    xx, yy, zz = sample_height_field_grid(
        nemo,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        batch_size=batch_size,
    )
    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    if nemo.field.has_color():
        colors = _sample_field_colors(nemo, xx, yy, batch_size=batch_size)
    else:
        colors = _height_colors(zz, estimate_z_bounds(nemo)).reshape(-1, 3)
    vertex_colors = np.clip(255.0 * colors, 0.0, 255.0).astype(np.uint8)

    h, w = zz.shape
    quads_r, quads_c = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing="ij")
    i0 = quads_r * w + quads_c
    i1 = i0 + 1
    i2 = i0 + w
    i3 = i2 + 1
    faces = np.stack(
        [
            np.stack([i0, i1, i2], axis=-1),
            np.stack([i1, i3, i2], axis=-1),
        ],
        axis=0,
    ).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)


def shade_render(
    render: RenderResult,
    *,
    light_direction: tuple[float, float, float] = (-0.35, -0.25, 0.9),
    ambient: float = 0.35,
    z_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    h, w = render.depth.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[...] = np.array([0.78, 0.86, 0.96], dtype=np.float32)

    mask = np.asarray(render.hit_mask, dtype=bool)
    if not np.any(mask):
        return (255.0 * image).astype(np.uint8)

    points = np.asarray(render.points, dtype=np.float32)
    normals = np.asarray(render.normals, dtype=np.float32)

    if z_bounds is None:
        z_hit = points[..., 2][mask]
        z_min = float(np.min(z_hit))
        z_max = float(np.max(z_hit))
    else:
        z_min, z_max = z_bounds
    if z_max - z_min < 1e-6:
        z_max = z_min + 1.0

    if render.rgb is not None:
        base = np.clip(np.asarray(render.rgb, dtype=np.float32), 0.0, 1.0)
    else:
        base = _height_colors(points[..., 2], (z_min, z_max))
    light = np.asarray(light_direction, dtype=np.float32)
    light /= max(float(np.linalg.norm(light)), 1e-8)
    diffuse = np.clip(np.sum(normals * light[None, None, :], axis=-1), 0.0, 1.0)
    shading = np.clip(float(ambient) + (1.0 - float(ambient)) * diffuse, 0.0, 1.0)
    shaded = np.clip(base * shading[..., None], 0.0, 1.0)
    image[mask] = shaded[mask]

    depth = np.asarray(render.depth, dtype=np.float32)
    finite = np.isfinite(depth) & mask
    if np.any(finite):
        depth_norm = np.zeros_like(depth)
        d = depth[finite]
        d_min = float(np.min(d))
        d_max = float(np.max(d))
        if d_max - d_min > 1e-6:
            depth_norm[finite] = (d - d_min) / (d_max - d_min)
        haze = np.clip(0.08 + 0.18 * depth_norm, 0.0, 0.28)
        sky = np.array([0.82, 0.9, 0.98], dtype=np.float32)
        image[finite] = (1.0 - haze[finite, None]) * image[finite] + haze[finite, None] * sky

    return np.clip(255.0 * image, 0.0, 255.0).astype(np.uint8)


def _height_colors(z: np.ndarray, z_bounds: tuple[float, float]) -> np.ndarray:
    z_min, z_max = z_bounds
    t = np.clip((np.asarray(z, dtype=np.float32) - float(z_min)) / max(float(z_max - z_min), 1e-6), 0.0, 1.0)
    low = np.array([0.16, 0.35, 0.22], dtype=np.float32)
    mid = np.array([0.58, 0.49, 0.30], dtype=np.float32)
    high = np.array([0.85, 0.81, 0.74], dtype=np.float32)
    snow = np.array([0.96, 0.96, 0.97], dtype=np.float32)

    first = np.clip(t / 0.55, 0.0, 1.0)
    second = np.clip((t - 0.55) / 0.3, 0.0, 1.0)
    third = np.clip((t - 0.85) / 0.15, 0.0, 1.0)

    color = low + (mid - low) * first[..., None]
    color = color + (high - color) * second[..., None]
    color = color + (snow - color) * third[..., None]
    return np.clip(color, 0.0, 1.0)


def _sample_field_colors(
    nemo: Nemo,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    preds: list[np.ndarray] = []
    device = nemo.device
    for start in range(0, len(xy), batch_size):
        batch = torch.as_tensor(xy[start : start + batch_size], dtype=torch.float32, device=device)
        pred = nemo.field.color(batch).detach().cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0).reshape(xx.shape[0] * xx.shape[1], 3)
