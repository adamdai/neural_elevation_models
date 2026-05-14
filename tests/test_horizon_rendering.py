from __future__ import annotations

import numpy as np
import torch

from nemo import CameraIntrinsics, HeightField, look_at_pose, render_horizon_samples
from nemo.image_training import _render_hit_mask


class FlatHeightField(HeightField):
    def __init__(
        self,
        *,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        height: float,
    ) -> None:
        super().__init__(bounds)
        self.height = torch.nn.Parameter(torch.tensor([height], dtype=torch.float32))

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        return self.height.expand(xy.shape[0], 1)


def _make_camera() -> tuple[CameraIntrinsics, np.ndarray]:
    intrinsics = CameraIntrinsics(
        width=9,
        height=9,
        fx=8.0,
        fy=8.0,
        cx=4.5,
        cy=4.5,
    )
    pose = look_at_pose(
        np.array([0.0, -2.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    return intrinsics, pose


def test_horizon_rows_bracket_hit_transition() -> None:
    field = FlatHeightField(bounds=((-50.0, 50.0), (-50.0, 50.0)), height=0.0)
    intrinsics, pose = _make_camera()
    columns = torch.tensor([0.0, 4.0, 8.0], dtype=torch.float32)

    result = render_horizon_samples(
        field,
        intrinsics,
        pose,
        columns,
        t_near=0.01,
        t_far=200.0,
        row_step=2.0,
        row_bisection_steps=8,
        differentiable=False,
    )

    assert torch.all(result.hit_mask)
    assert torch.all(result.rows > 0.0)
    assert torch.all(result.rows < intrinsics.height - 1)

    rows_above = torch.clamp(result.rows - 0.05, min=0.0)
    rows_below = torch.clamp(result.rows + 0.05, max=float(intrinsics.height - 1))
    hits_above = _render_hit_mask(
        field,
        intrinsics,
        pose,
        torch.stack([columns, rows_above], dim=-1),
        t_near=0.01,
        t_far=200.0,
        num_bracket_samples=32,
        num_bisection_steps=6,
        num_newton_steps=1,
    )
    hits_below = _render_hit_mask(
        field,
        intrinsics,
        pose,
        torch.stack([columns, rows_below], dim=-1),
        t_near=0.01,
        t_far=200.0,
        num_bracket_samples=32,
        num_bisection_steps=6,
        num_newton_steps=1,
    )
    assert not torch.any(hits_above)
    assert torch.all(hits_below)


def test_horizon_rows_backpropagate_to_geometry() -> None:
    field = FlatHeightField(bounds=((-50.0, 50.0), (-50.0, 50.0)), height=0.0)
    intrinsics, pose = _make_camera()
    columns = torch.tensor([4.0], dtype=torch.float32)

    result = render_horizon_samples(
        field,
        intrinsics,
        pose,
        columns,
        t_near=0.01,
        t_far=200.0,
        row_step=2.0,
        row_bisection_steps=8,
        horizon_refine_steps=2,
        differentiable=True,
    )
    loss = result.rows.sum()
    loss.backward()

    assert field.height.grad is not None
    assert torch.isfinite(field.height.grad).all()
    assert torch.any(torch.abs(field.height.grad) > 0.0)
