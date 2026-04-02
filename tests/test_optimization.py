from __future__ import annotations

import pytest
import torch

from nemo import Nemo, PlaneBaseline, TorchFitConfig


def _terrain_height(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, :1]
    y = xy[:, 1:2]
    return 0.15 * x - 0.05 * y + 0.8 * (x - 0.25) ** 2 + 1.2 * (y + 0.2) ** 2


@pytest.fixture(scope="module")
def learned_basin() -> tuple[Nemo, torch.Tensor]:
    torch.manual_seed(0)
    xy = torch.rand(2048, 2) * 2.0 - 1.0
    z = _terrain_height(xy)

    nemo = Nemo.residual_mlp(
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        baseline=PlaneBaseline(ax=0.15, ay=-0.05),
        hidden_dim=64,
        depth=3,
    )
    nemo.fit(
        xy,
        z,
        fit_config=TorchFitConfig(iterations=200, lr=2e-3, early_stopping_patience=30),
    )

    optimum = torch.tensor([0.15625, -0.17916667], dtype=torch.float32)
    return nemo, optimum


def test_optimization_descends_to_basin_minimum(learned_basin: tuple[Nemo, torch.Tensor]) -> None:
    nemo, optimum = learned_basin
    start = torch.tensor([[0.8, 0.8]], dtype=torch.float32)
    xy_opt = start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([xy_opt], lr=0.05)

    for _ in range(120):
        optimizer.zero_grad()
        loss = nemo.field.h(xy_opt).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            xy_opt.clamp_(-1.0, 1.0)

    start_height = float(nemo.field.h(start).item())
    final_height = float(nemo.field.h(xy_opt).item())
    final_distance = float(torch.norm(xy_opt.detach().squeeze(0) - optimum).item())

    assert final_height < 0.1
    assert final_height < start_height - 1.0
    assert final_distance < 0.05


def test_optimization_finds_low_trajectory_through_basin(
    learned_basin: tuple[Nemo, torch.Tensor],
) -> None:
    nemo, optimum = learned_basin
    start = torch.tensor([0.85, 0.75], dtype=torch.float32)
    end = torch.tensor([0.05, -0.15], dtype=torch.float32)
    initial_path = torch.stack(
        [
            torch.linspace(start[0], end[0], 16),
            torch.linspace(start[1], end[1], 16),
        ],
        dim=-1,
    )
    interior = torch.nn.Parameter(initial_path[1:-1].clone())
    optimizer = torch.optim.Adam([interior], lr=0.05)

    for _ in range(250):
        optimizer.zero_grad()
        path = torch.cat([start[None], interior, end[None]], dim=0)
        heights = nemo.field.h(path).squeeze(-1)
        step_lengths = (path[1:] - path[:-1]).square().sum(dim=-1)
        loss = heights[1:-1].mean() + 0.2 * step_lengths.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            interior.clamp_(-1.0, 1.0)

    final_path = torch.cat([start[None], interior.detach(), end[None]], dim=0)
    initial_mean_height = float(nemo.field.h(initial_path).mean().item())
    final_mean_height = float(nemo.field.h(final_path).mean().item())
    initial_mean_distance = float(torch.norm(initial_path[1:-1] - optimum, dim=-1).mean().item())
    final_mean_distance = float(torch.norm(final_path[1:-1] - optimum, dim=-1).mean().item())

    assert final_mean_height < initial_mean_height - 0.3
    assert final_mean_distance < 0.1
    assert final_mean_distance < initial_mean_distance * 0.25
