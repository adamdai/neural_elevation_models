from __future__ import annotations

import numpy as np

from nemo import Nemo
from nemo.dem import CameraIntrinsics
from nemo.rendering import render_height_field
from nemo.viewer import CameraViewState, intrinsics_from_view_state, shade_render
from nemo.rendering import RenderResult


def test_intrinsics_from_view_state_scales_resolution() -> None:
    view = CameraViewState(
        position=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        look_at=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        up_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        fov=np.deg2rad(60.0),
        image_width=800,
        image_height=600,
    )
    intrinsics = intrinsics_from_view_state(view, resolution_scale=0.5)
    assert intrinsics.width == 400
    assert intrinsics.height == 300
    assert intrinsics.fx > 0.0
    assert intrinsics.fy > 0.0


def test_shade_render_returns_uint8_image() -> None:
    render = RenderResult(
        depth=np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
        hit_mask=np.array([[True, False], [True, True]]),
        points=np.array(
            [
                [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        normals=np.array(
            [
                [[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    )
    rgb = shade_render(render, z_bounds=(0.0, 2.0))
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8


def test_render_rgb_produces_image() -> None:
    nemo = Nemo.smooth_grid(bounds=((-1.0, 1.0), (-1.0, 1.0)))
    view = CameraViewState(
        position=np.array([-2.0, -2.0, 3.0], dtype=np.float32),
        look_at=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        up_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        fov=np.deg2rad(55.0),
        image_width=160,
        image_height=120,
    )
    from nemo.viewer import render_rgb

    rgb, render = render_rgb(
        nemo,
        view,
        resolution_scale=0.25,
        num_bracket_samples=24,
        num_bisection_steps=4,
        num_newton_steps=1,
        ray_batch_size=4096,
    )
    assert rgb.ndim == 3
    assert rgb.shape[-1] == 3
    assert render.depth.shape == rgb.shape[:2]


def test_render_height_field_smoke() -> None:
    nemo = Nemo.smooth_grid(bounds=((-1.0, 1.0), (-1.0, 1.0)))
    intrinsics = CameraIntrinsics(width=32, height=24, fx=28.0, fy=28.0, cx=16.0, cy=12.0)
    world_T_camera = np.eye(4, dtype=np.float32)
    world_T_camera[:3, 3] = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    render = render_height_field(
        nemo.field,
        intrinsics,
        world_T_camera,
        t_near=0.1,
        t_far=4.0,
        num_bracket_samples=16,
        num_bisection_steps=4,
        num_newton_steps=1,
        ray_batch_size=2048,
    )
    assert render.depth.shape == (24, 32)
    assert render.hit_mask.shape == (24, 32)
