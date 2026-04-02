from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from nemo.nemo import Nemo


@dataclass(frozen=True)
class ParticleSimConfig:
    dt: float = 0.02
    max_steps: int = 1500
    gravity: float = 9.81
    spawn_height: float = 0.1
    air_damping: float = 0.05
    surface_damping: float = 1.5
    static_friction_slope: float = 0.015
    settle_speed: float = 0.02
    settle_slope: float = 0.01
    downhill_scale: float = 1.0


@dataclass(frozen=True)
class ParticleSimulationResult:
    trajectory_xyz: np.ndarray
    status: str
    steps: int
    start_xy: tuple[float, float]
    final_xy: tuple[float, float]
    final_height: float

    def to_payload(self) -> dict[str, object]:
        return {
            "trajectory_xyz": self.trajectory_xyz.tolist(),
            "status": self.status,
            "steps": self.steps,
            "start_xy": list(self.start_xy),
            "final_xy": list(self.final_xy),
            "final_height": self.final_height,
        }


def sample_height_field_grid(
    nemo: Nemo,
    *,
    resolution_x: int = 160,
    resolution_y: int = 160,
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = nemo.field.bounds
    x_axis = np.linspace(bounds[0][0], bounds[0][1], int(resolution_x), dtype=np.float32)
    y_axis = np.linspace(bounds[1][0], bounds[1][1], int(resolution_y), dtype=np.float32)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    preds: list[np.ndarray] = []
    device = nemo.device
    for start in range(0, len(xy), batch_size):
        batch = torch.as_tensor(xy[start : start + batch_size], dtype=torch.float32, device=device)
        pred = nemo.h(batch).detach().cpu().numpy()
        preds.append(pred)
    zz = np.concatenate(preds, axis=0).reshape(yy.shape)
    return xx, yy, zz


def simulate_particle(
    nemo: Nemo,
    start_xy: tuple[float, float] | list[float] | np.ndarray,
    *,
    config: ParticleSimConfig | None = None,
) -> ParticleSimulationResult:
    cfg = config or ParticleSimConfig()
    device = nemo.device

    xy = torch.as_tensor(start_xy, dtype=torch.float32, device=device).reshape(2)
    bounds = nemo.field.bounds
    x_min, x_max = float(bounds[0][0]), float(bounds[0][1])
    y_min, y_max = float(bounds[1][0]), float(bounds[1][1])

    if not (x_min <= float(xy[0]) <= x_max and y_min <= float(xy[1]) <= y_max):
        raise ValueError("Particle spawn point must lie within the NeMO bounds.")

    surface_z = float(nemo.h(xy[None]).item())
    z = surface_z + float(cfg.spawn_height)
    vel_xy = torch.zeros(2, dtype=torch.float32, device=device)
    vel_z = 0.0
    on_surface = False

    trajectory: list[list[float]] = [[float(xy[0]), float(xy[1]), float(z)]]
    status = "max_steps"
    step_idx = 0

    for step_idx in range(1, int(cfg.max_steps) + 1):
        prev_xy = xy.clone()

        surface_z = float(nemo.h(xy[None]).item())

        if not on_surface:
            vel_z -= float(cfg.gravity) * float(cfg.dt)
            vel_z *= max(0.0, 1.0 - float(cfg.air_damping) * float(cfg.dt))
            z += vel_z * float(cfg.dt)
            if z <= surface_z:
                z = surface_z
                vel_z = 0.0
                on_surface = True
        else:
            grad = nemo.grad(xy[None]).squeeze(0)
            slope = float(torch.linalg.norm(grad).item())
            speed = float(torch.linalg.norm(vel_xy).item())
            if slope < float(cfg.static_friction_slope) and speed < float(cfg.settle_speed):
                vel_xy.zero_()
                status = "settled"
                trajectory.append([float(xy[0]), float(xy[1]), float(surface_z)])
                break

            accel_xy = -float(cfg.downhill_scale) * float(cfg.gravity) * grad / max(1.0 + slope * slope, 1e-6)
            vel_xy = vel_xy + accel_xy * float(cfg.dt)
            vel_xy = vel_xy * max(0.0, 1.0 - float(cfg.surface_damping) * float(cfg.dt))
            xy = xy + vel_xy * float(cfg.dt)

            rolled_off = (
                float(xy[0]) < x_min
                or float(xy[0]) > x_max
                or float(xy[1]) < y_min
                or float(xy[1]) > y_max
            )
            if rolled_off:
                xy = _clip_segment_to_bounds(prev_xy, xy, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                z = float(nemo.h(xy[None]).item())
                trajectory.append([float(xy[0]), float(xy[1]), float(z)])
                status = "rolled_off"
                break

            z = float(nemo.h(xy[None]).item())

            speed = float(torch.linalg.norm(vel_xy).item())
            if speed < float(cfg.settle_speed) and slope < float(cfg.settle_slope):
                vel_xy.zero_()
                status = "settled"
                trajectory.append([float(xy[0]), float(xy[1]), float(z)])
                break

        trajectory.append([float(xy[0]), float(xy[1]), float(z)])

    final_xy = (float(xy[0]), float(xy[1]))
    final_height = float(z)
    return ParticleSimulationResult(
        trajectory_xyz=np.asarray(trajectory, dtype=np.float32),
        status=status,
        steps=step_idx,
        start_xy=(float(trajectory[0][0]), float(trajectory[0][1])),
        final_xy=final_xy,
        final_height=final_height,
    )


def particle_config_payload(config: ParticleSimConfig) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(config).items()}


def _clip_segment_to_bounds(
    start_xy: torch.Tensor,
    end_xy: torch.Tensor,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    direction = end_xy - start_xy
    t_exit = 1.0

    if float(direction[0]) > 0.0:
        t_exit = min(t_exit, (x_max - float(start_xy[0])) / float(direction[0]))
    elif float(direction[0]) < 0.0:
        t_exit = min(t_exit, (x_min - float(start_xy[0])) / float(direction[0]))

    if float(direction[1]) > 0.0:
        t_exit = min(t_exit, (y_max - float(start_xy[1])) / float(direction[1]))
    elif float(direction[1]) < 0.0:
        t_exit = min(t_exit, (y_min - float(start_xy[1])) / float(direction[1]))

    t_exit = float(np.clip(t_exit, 0.0, 1.0))
    clipped = start_xy + t_exit * direction
    clipped[0] = clipped[0].clamp(x_min, x_max)
    clipped[1] = clipped[1].clamp(y_min, y_max)
    return clipped
