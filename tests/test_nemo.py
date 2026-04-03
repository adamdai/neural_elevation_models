from __future__ import annotations

import torch

from nemo import Nemo, PlaneBaseline, TileConfig, TorchFitConfig
from nemo.models.residual_mlp import ResidualMLPHeightField
from nemo.models.smooth_grid import SmoothGridHeightField


def make_training_data(n: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    xy = torch.rand(n, 2) * 2.0 - 1.0
    z = (
        0.2 * xy[:, :1]
        - 0.1 * xy[:, 1:2]
        + 0.1 * torch.sin(3.0 * xy[:, :1]) * torch.cos(2.0 * xy[:, 1:2])
    )
    return xy, z


def test_residual_mlp_fit_reduces_error() -> None:
    xy, z = make_training_data()
    nemo = Nemo.residual_mlp(
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        baseline=PlaneBaseline(ax=0.2, ay=-0.1),
        hidden_dim=64,
        depth=3,
    )
    nemo.fit(xy, z, fit_config=TorchFitConfig(iterations=300, lr=2e-3))
    pred = nemo.h(xy)
    mse = torch.mean((pred - z) ** 2).item()
    assert mse < 5e-3


def test_gradient_matches_autograd() -> None:
    field = ResidualMLPHeightField(bounds=((-1.0, 1.0), (-1.0, 1.0)), hidden_dim=16, depth=2)
    xy = torch.tensor([[0.25, -0.5]], dtype=torch.float32, requires_grad=True)
    z = field.h(xy)
    expected = torch.autograd.grad(z.sum(), xy)[0]
    got = field.grad(xy.detach())
    assert torch.allclose(got, expected, atol=1e-5)


def test_tiled_field_can_fit_local_models() -> None:
    xy, z = make_training_data(2048)
    tile_config = TileConfig(
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        tile_size=(1.2, 1.2),
        overlap=(0.2, 0.2),
    )
    nemo = Nemo.tiled(
        tile_config=tile_config,
        field_factory=lambda bounds: ResidualMLPHeightField(
            bounds=bounds,
            baseline=PlaneBaseline(ax=0.2, ay=-0.1),
            hidden_dim=32,
            depth=2,
        ),
        fit_config=TorchFitConfig(iterations=150, lr=2e-3),
    )
    nemo.fit(xy, z)
    pred = nemo.h(xy[:128])
    mse = torch.mean((pred - z[:128]) ** 2).item()
    assert mse < 2e-2


def test_smooth_grid_siren_without_residual_matches_autograd_gradient() -> None:
    field = SmoothGridHeightField(
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        hidden_dim=16,
        depth=3,
        backbone_type="siren",
        residual_type="none",
    )
    xy = torch.tensor([[0.1, -0.2]], dtype=torch.float32, requires_grad=True)
    z = field.h(xy)
    expected = torch.autograd.grad(z.sum(), xy)[0]
    got = field.grad(xy.detach())
    assert torch.allclose(got, expected, atol=1e-5)
