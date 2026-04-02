from __future__ import annotations

import torch

from nemo import HeightField, Nemo
from nemo.particle_sim import ParticleSimConfig, simulate_particle


class BowlField(HeightField):
    def __init__(self) -> None:
        super().__init__(bounds=((-1.0, 1.0), (-1.0, 1.0)))

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, :1]
        y = xy[:, 1:2]
        return x.square() + 1.5 * y.square()


class PlaneField(HeightField):
    def __init__(self) -> None:
        super().__init__(bounds=((-1.0, 1.0), (-1.0, 1.0)))

    def h(self, xy: torch.Tensor) -> torch.Tensor:
        return xy[:, :1]


def test_particle_settles_in_bowl() -> None:
    nemo = Nemo(BowlField())
    result = simulate_particle(
        nemo,
        (0.8, 0.4),
        config=ParticleSimConfig(
            dt=0.02,
            max_steps=1200,
            gravity=9.81,
            spawn_height=0.1,
            surface_damping=2.5,
            static_friction_slope=0.02,
            settle_speed=0.025,
            settle_slope=0.015,
        ),
    )

    assert result.status == "settled"
    assert result.final_height < 0.02
    assert abs(result.final_xy[0]) < 0.08
    assert abs(result.final_xy[1]) < 0.08


def test_particle_rolls_off_plane() -> None:
    nemo = Nemo(PlaneField())
    result = simulate_particle(
        nemo,
        (0.7, 0.0),
        config=ParticleSimConfig(
            dt=0.02,
            max_steps=500,
            gravity=9.81,
            spawn_height=0.05,
            surface_damping=0.2,
            static_friction_slope=0.0,
            settle_speed=0.0,
            settle_slope=0.0,
        ),
    )

    assert result.status == "rolled_off"
    assert result.final_xy[0] <= -0.99
