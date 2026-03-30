from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nemo.height_field import HeightField, as_targets, as_tensor


@dataclass
class TorchFitConfig:
    iterations: int = 1000
    lr: float = 1e-3
    batch_size: int | None = None
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    verbose: bool = False


class TorchHeightFieldFitter:
    """Generic supervised fitter for torch-based height fields."""

    def __init__(self, optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam) -> None:
        self.optimizer_cls = optimizer_cls

    def fit(
        self,
        field: HeightField,
        xy: Tensor,
        z: Tensor,
        *,
        config: TorchFitConfig | None = None,
    ) -> HeightField:
        config = config or TorchFitConfig()
        device = next(field.parameters(), torch.empty(0, device=torch.device("cpu"))).device
        xy = as_tensor(xy, device=device)
        z = as_targets(z, device=device)

        optimizer = self.optimizer_cls(
            field.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        n = xy.shape[0]
        batch_size = config.batch_size or n

        for step in range(config.iterations):
            if batch_size >= n:
                batch_xy = xy
                batch_z = z
            else:
                idx = torch.randint(0, n, (batch_size,), device=device)
                batch_xy = xy[idx]
                batch_z = z[idx]

            pred = field.h(batch_xy)
            loss = torch.nn.functional.mse_loss(pred, batch_z)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(field.parameters(), config.grad_clip_norm)
            optimizer.step()

            if config.verbose and (
                step == 0 or step == config.iterations - 1 or (step + 1) % 100 == 0
            ):
                print(f"[fit] step={step + 1} loss={loss.item():.6f}")

        return field
