from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    gradient_loss_weight: float = 0.0
    eval_every: int = 25
    early_stopping_patience: int | None = 20
    early_stopping_min_delta: float = 1e-5
    restore_best: bool = True
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
        grad_targets: Tensor | None = None,
        *,
        config: TorchFitConfig | None = None,
    ) -> HeightField:
        config = config or TorchFitConfig()
        device = next(field.parameters(), torch.empty(0, device=torch.device("cpu"))).device
        xy = as_tensor(xy, device=device)
        z = as_targets(z, device=device)
        if grad_targets is not None:
            grad_targets = torch.as_tensor(grad_targets, dtype=torch.float32, device=device)
            if grad_targets.ndim != 2 or grad_targets.shape[-1] != 2:
                raise ValueError("Expected grad_targets to have shape (N, 2).")
            if grad_targets.shape[0] != xy.shape[0]:
                raise ValueError("grad_targets must have the same number of samples as xy.")
        field.configure_output_normalization(xy, z)

        optimizer = self.optimizer_cls(
            field.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        n = xy.shape[0]
        batch_size = config.batch_size or n
        eval_every = max(int(config.eval_every), 1)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale_evals = 0

        def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
            target_cast = target.to(dtype=pred.dtype)
            return torch.nn.functional.mse_loss(pred, target_cast)

        def evaluate_full_loss() -> float:
            if grad_targets is None or float(config.gradient_loss_weight) <= 0.0:
                with torch.no_grad():
                    pred_full = field.h(xy)
                    target_full = z.to(dtype=pred_full.dtype)
                    return float(torch.nn.functional.mse_loss(pred_full, target_full).item())
            with torch.enable_grad():
                pred_full = field.h(xy)
                target_full = z.to(dtype=pred_full.dtype)
                loss_full = torch.nn.functional.mse_loss(pred_full, target_full)
                grad_pred_full = field.training_gradients(xy).detach()
                grad_target_full = grad_targets.to(dtype=grad_pred_full.dtype)
                grad_loss_full = torch.nn.functional.mse_loss(grad_pred_full, grad_target_full)
                total_full = loss_full + float(config.gradient_loss_weight) * grad_loss_full
                return float(total_full.item())

        for step in range(config.iterations):
            if batch_size >= n:
                batch_xy = xy
                batch_z = z
            else:
                idx = torch.randint(0, n, (batch_size,), device=device)
                batch_xy = xy[idx]
                batch_z = z[idx]

            pred = field.training_predictions(batch_xy)
            target = field.training_targets(batch_xy, batch_z)
            loss = mse_loss(pred, target)
            grad_loss = None
            if grad_targets is not None and float(config.gradient_loss_weight) > 0.0:
                batch_grad_target = grad_targets if batch_size >= n else grad_targets[idx]
                grad_pred = field.training_gradients(batch_xy)
                grad_loss = mse_loss(grad_pred, batch_grad_target)
                loss = loss + float(config.gradient_loss_weight) * grad_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(field.parameters(), config.grad_clip_norm)
            optimizer.step()

            should_eval = (
                step == 0
                or step == config.iterations - 1
                or (step + 1) % eval_every == 0
                or batch_size < n
            )
            full_loss = None
            if should_eval:
                full_loss = evaluate_full_loss()
                if full_loss < best_loss - float(config.early_stopping_min_delta):
                    best_loss = full_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in field.state_dict().items()
                    }
                    stale_evals = 0
                else:
                    stale_evals += 1

            if config.verbose and (
                step == 0 or step == config.iterations - 1 or (step + 1) % 100 == 0
            ):
                if full_loss is None:
                    print(f"[fit] step={step + 1} batch_loss={loss.item():.6f}")
                else:
                    suffix = ""
                    if grad_loss is not None:
                        suffix = f" grad_loss={grad_loss.item():.6f}"
                    print(
                        f"[fit] step={step + 1} batch_loss={loss.item():.6f} "
                        f"full_loss={full_loss:.6f}{suffix}"
                    )

            if (
                config.early_stopping_patience is not None
                and should_eval
                and stale_evals >= int(config.early_stopping_patience)
            ):
                if config.verbose:
                    print(
                        f"[fit] early stopping at step={step + 1} "
                        f"best_full_loss={best_loss:.6f}"
                    )
                break

        if config.restore_best and best_state is not None:
            field.load_state_dict(best_state)

        return field
