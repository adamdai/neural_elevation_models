from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from nemo import Nemo, TorchFitConfig
from nemo.image_training import render_color_samples
from nemo.io.nerfstudio_dataset import load_nerfstudio_dataset
from nemo.models.smooth_grid import SmoothGridHeightField


@dataclass
class TrainArgs:
    dataset_path: str = "data/images/RedRocks"
    output_dir: str = "output/train_redrocks"
    image_folder: str = "images_4"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    max_frames: int | None = 16
    eval_frame_index: int = 0
    fit_geometry: bool = True
    geometry_iterations: int = 2000
    geometry_lr: float = 1e-3
    geometry_batch_size: int = 32768
    geometry_max_points: int = 120000
    hidden_dim: int = 64
    depth: int = 4
    backbone_type: str = "mlp"
    residual_type: str = "grid"
    grid_resolution_x: int = 192
    grid_resolution_y: int = 192
    image_iterations: int = 1000
    image_lr: float = 1e-2
    color_weight_decay: float = 0.0
    freeze_geometry: bool = True
    image_loss_weight: float = 1.0
    geometry_loss_weight: float = 0.1
    differentiable_geometry_from_images: bool = False
    geometry_loss_weight_final: float | None = None
    geometry_smoothness_weight: float = 0.0
    smoothness_batch_size: int = 2048
    pixel_batch_size: int = 4096
    eval_every: int = 25
    eval_pixel_count: int = 8192
    frame_hit_eval_pixels: int = 2048
    min_train_frame_hit_fraction: float = 0.0
    render_preview_every: int = 100
    render_preview_max_dim: int = 256
    t_near: float = 0.1
    t_far: float | None = None
    num_bracket_samples: int = 32
    num_bisection_steps: int = 6
    num_newton_steps: int = 1


def main(args: TrainArgs) -> None:
    _seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_nerfstudio_dataset(
        args.dataset_path,
        image_folder=args.image_folder,
        max_frames=args.max_frames,
    )
    auto_t_far = _auto_t_far(dataset)
    resolved_t_far = float(args.t_far) if args.t_far is not None else auto_t_far
    print(
        f"[data] frames={len(dataset.frames)} sparse_points={len(dataset.sparse_points)} "
        f"bounds={dataset.bounds_xy} z_range={dataset.z_range} "
        f"t_far={resolved_t_far:.2f}"
    )

    nemo = Nemo.smooth_grid(
        bounds=dataset.bounds_xy,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        backbone_type=args.backbone_type,
        residual_type=args.residual_type,
        color_type="hashgrid",
        grid_resolution_x=args.grid_resolution_x,
        grid_resolution_y=args.grid_resolution_y,
    ).to(args.device)
    image_geom_grad_enabled = bool(args.differentiable_geometry_from_images) and _supports_image_geometry_gradients(nemo)
    if bool(args.differentiable_geometry_from_images) and not image_geom_grad_enabled:
        print("[train] image->geometry gradients disabled for this field; using joint color+geometry losses instead")

    if args.fit_geometry:
        _fit_geometry(nemo, dataset, args)

    geometry_xy_full, geometry_z_full = _prepare_geometry_targets(dataset, args, device=args.device)
    frame_hit_fractions = _evaluate_frame_hit_fractions(
        nemo,
        dataset.frames,
        args,
        resolved_t_far=resolved_t_far,
    )
    train_frames = [
        frame
        for frame, hit_fraction in zip(dataset.frames, frame_hit_fractions, strict=True)
        if hit_fraction >= float(args.min_train_frame_hit_fraction)
    ]
    if not train_frames:
        raise ValueError(
            f"No train frames satisfy min_train_frame_hit_fraction={args.min_train_frame_hit_fraction:.3f}"
        )
    print(
        f"[frames] using {len(train_frames)}/{len(dataset.frames)} frames for training "
        f"(min_hit={min(frame_hit_fractions):.3f} max_hit={max(frame_hit_fractions):.3f} "
        f"threshold={float(args.min_train_frame_hit_fraction):.3f})"
    )

    if args.freeze_geometry:
        for name, parameter in nemo.field.named_parameters():
            parameter.requires_grad_(name.startswith("color_decoder."))
    else:
        for parameter in nemo.field.parameters():
            parameter.requires_grad_(True)

    trainable = [parameter for parameter in nemo.field.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found for image training.")
    optimizer = torch.optim.Adam(
        trainable,
        lr=args.image_lr,
        weight_decay=args.color_weight_decay,
    )

    metrics: list[dict[str, float | int]] = []
    skipped_no_hit = 0
    train_start = time.perf_counter()
    for step in range(1, int(args.image_iterations) + 1):
        frame = train_frames[random.randrange(len(train_frames))]
        uv, target_rgb = _sample_pixels(frame.image, args.pixel_batch_size, device=args.device)
        render = render_color_samples(
            nemo.field,
            frame.intrinsics,
            frame.world_T_camera,
            uv,
            t_near=args.t_near,
            t_far=resolved_t_far,
            num_bracket_samples=args.num_bracket_samples,
            num_bisection_steps=args.num_bisection_steps,
            num_newton_steps=args.num_newton_steps,
            differentiable_geometry=(not args.freeze_geometry and image_geom_grad_enabled),
        )

        total_loss = None
        hit_mask = render.hit_mask
        if torch.any(hit_mask):
            pred = render.rgb[hit_mask]
            target = target_rgb[hit_mask]
            color_loss = torch.nn.functional.mse_loss(pred, target)
            psnr = _psnr(color_loss.detach())
            hit_fraction = float(hit_mask.float().mean().item())
            total_loss = float(args.image_loss_weight) * color_loss
        else:
            skipped_no_hit += 1
            if step == 1 or skipped_no_hit <= 5 or skipped_no_hit % 25 == 0:
                print(f"[train] step={step} skipped batch with no ray hits")
            color_loss = None
            psnr = float("nan")
            hit_fraction = 0.0

        geometry_loss = None
        geometry_weight = _scheduled_weight(
            step,
            int(args.image_iterations),
            start=float(args.geometry_loss_weight),
            end=(
                float(args.geometry_loss_weight_final)
                if args.geometry_loss_weight_final is not None
                else float(args.geometry_loss_weight)
            ),
        )
        if geometry_weight > 0.0 and geometry_xy_full.shape[0] > 0:
            idx = torch.randint(0, geometry_xy_full.shape[0], (int(args.geometry_batch_size),), device=args.device)
            geom_xy = geometry_xy_full[idx]
            geom_z = geometry_z_full[idx]
            geom_pred = nemo.field.h(geom_xy)
            geometry_loss = torch.nn.functional.mse_loss(geom_pred, geom_z)
            total_loss = (
                geometry_weight * geometry_loss
                if total_loss is None
                else total_loss + geometry_weight * geometry_loss
            )

        smoothness_loss = None
        if float(args.geometry_smoothness_weight) > 0.0 and not args.freeze_geometry:
            smooth_xy = _sample_xy_in_bounds(
                nemo,
                int(args.smoothness_batch_size),
                device=args.device,
            )
            smoothness_loss = _geometry_smoothness_loss(nemo, smooth_xy)
            total_loss = total_loss + float(args.geometry_smoothness_weight) * smoothness_loss

        if total_loss is None:
            continue

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % int(args.eval_every) == 0 or step == int(args.image_iterations):
            eval_metrics = _evaluate_color(
                nemo,
                dataset.frames[args.eval_frame_index % len(dataset.frames)],
                args,
                resolved_t_far=resolved_t_far,
            )
            record = {
                "step": step,
                "train_loss": float(total_loss.item()),
                "train_color_loss": float(color_loss.item()) if color_loss is not None else float("nan"),
                "train_geometry_loss": float(geometry_loss.item()) if geometry_loss is not None else float("nan"),
                "train_geometry_weight": float(geometry_weight),
                "train_smoothness_loss": float(smoothness_loss.item()) if smoothness_loss is not None else float("nan"),
                "train_psnr": float(psnr),
                "train_hit_fraction": hit_fraction,
                "skipped_no_hit": skipped_no_hit,
                **eval_metrics,
            }
            metrics.append(record)
            print(
                f"[train] step={step} "
                f"train_loss={record['train_loss']:.6f} "
                f"color_loss={record['train_color_loss']:.6f} "
                f"geom_loss={record['train_geometry_loss']:.6f} "
                f"geom_w={record['train_geometry_weight']:.4f} "
                f"smooth={record['train_smoothness_loss']:.6f} "
                f"train_psnr={record['train_psnr']:.2f} "
                f"train_hit={record['train_hit_fraction']:.3f} "
                f"eval_loss={record['eval_loss']:.6f} "
                f"eval_psnr={record['eval_psnr']:.2f} "
                f"eval_hit={record['eval_hit_fraction']:.3f}"
            )
        if step % int(args.render_preview_every) == 0 or step == int(args.image_iterations):
            _render_preview(
                nemo,
                dataset.frames[args.eval_frame_index % len(dataset.frames)],
                args,
                output_dir / f"preview_step_{step:05d}.png",
                resolved_t_far=resolved_t_far,
            )

    total_seconds = time.perf_counter() - train_start
    checkpoint_path = nemo.save_checkpoint(output_dir / "model.pt")
    summary = {
        "args": asdict(args),
        "resolved_t_far": resolved_t_far,
        "frame_hit_fractions": [
            {
                "frame_index": idx,
                "image_path": str(frame.image_path),
                "hit_fraction": float(hit_fraction),
                "used_for_training": bool(hit_fraction >= float(args.min_train_frame_hit_fraction)),
            }
            for idx, (frame, hit_fraction) in enumerate(zip(dataset.frames, frame_hit_fractions, strict=True))
        ],
        "skipped_no_hit": int(skipped_no_hit),
        "total_seconds": float(total_seconds),
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] seconds={total_seconds:.2f} checkpoint={checkpoint_path}")


def _fit_geometry(nemo: Nemo, dataset, args: TrainArgs) -> None:
    points = dataset.sparse_points
    x_min, x_max = dataset.bounds_xy[0]
    y_min, y_max = dataset.bounds_xy[1]
    in_bounds = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    points = points[in_bounds]
    if len(points) > int(args.geometry_max_points):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(points), size=int(args.geometry_max_points), replace=False)
        points = points[idx]
    xy = torch.from_numpy(points[:, :2].astype(np.float32)).to(args.device)
    z = torch.from_numpy(points[:, 2:3].astype(np.float32)).to(args.device)
    print(f"[geometry] fitting {len(points)} sparse points")
    nemo.fit(
        xy,
        z,
        fit_config=TorchFitConfig(
            iterations=int(args.geometry_iterations),
            lr=float(args.geometry_lr),
            batch_size=int(args.geometry_batch_size),
            eval_every=25,
            early_stopping_patience=None,
            restore_best=True,
            verbose=True,
        ),
    )


def _prepare_geometry_targets(dataset, args: TrainArgs, *, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    points = dataset.sparse_points
    x_min, x_max = dataset.bounds_xy[0]
    y_min, y_max = dataset.bounds_xy[1]
    in_bounds = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    points = points[in_bounds]
    if len(points) > int(args.geometry_max_points):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(points), size=int(args.geometry_max_points), replace=False)
        points = points[idx]
    xy = torch.from_numpy(points[:, :2].astype(np.float32)).to(device)
    z = torch.from_numpy(points[:, 2:3].astype(np.float32)).to(device)
    return xy, z


def _sample_pixels(
    image: torch.Tensor, batch_size: int, *, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image.shape[:2]
    idx = torch.randint(0, h * w, (int(batch_size),), device=device)
    v = idx // w
    u = idx % w
    uv = torch.stack([u, v], dim=-1).to(dtype=torch.float32)
    target = image.to(device=device, dtype=torch.float32).reshape(-1, 3)[idx]
    return uv, target


def _evaluate_color(
    nemo: Nemo,
    frame,
    args: TrainArgs,
    *,
    resolved_t_far: float,
) -> dict[str, float]:
    uv, target = _sample_pixels(frame.image, args.eval_pixel_count, device=args.device)
    with torch.no_grad():
        render = render_color_samples(
            nemo.field,
            frame.intrinsics,
            frame.world_T_camera,
            uv,
            t_near=args.t_near,
            t_far=resolved_t_far,
            num_bracket_samples=args.num_bracket_samples,
            num_bisection_steps=args.num_bisection_steps,
            num_newton_steps=args.num_newton_steps,
        )
        hit_mask = render.hit_mask
        if torch.any(hit_mask):
            loss = torch.nn.functional.mse_loss(render.rgb[hit_mask], target[hit_mask])
            hit_fraction = float(hit_mask.float().mean().item())
        else:
            loss = torch.nn.functional.mse_loss(render.rgb, target)
            hit_fraction = 0.0
    return {
        "eval_loss": float(loss.item()),
        "eval_psnr": float(_psnr(loss)),
        "eval_hit_fraction": hit_fraction,
    }


def _evaluate_frame_hit_fractions(
    nemo: Nemo,
    frames,
    args: TrainArgs,
    *,
    resolved_t_far: float,
) -> list[float]:
    hit_fractions: list[float] = []
    for frame in frames:
        uv, _ = _sample_pixels(frame.image, args.frame_hit_eval_pixels, device=args.device)
        with torch.no_grad():
            render = render_color_samples(
                nemo.field,
                frame.intrinsics,
                frame.world_T_camera,
                uv,
                t_near=args.t_near,
                t_far=resolved_t_far,
                num_bracket_samples=args.num_bracket_samples,
                num_bisection_steps=args.num_bisection_steps,
                num_newton_steps=args.num_newton_steps,
            )
        hit_fractions.append(float(render.hit_mask.float().mean().item()))
    return hit_fractions


def _render_preview(
    nemo: Nemo,
    frame,
    args: TrainArgs,
    path: Path,
    *,
    resolved_t_far: float,
) -> None:
    h, w = frame.image.shape[:2]
    scale = min(1.0, float(args.render_preview_max_dim) / max(h, w))
    h_small = max(1, int(round(h * scale)))
    w_small = max(1, int(round(w * scale)))
    vv, uu = torch.meshgrid(
        torch.linspace(0, h - 1, h_small, dtype=torch.float32, device=args.device),
        torch.linspace(0, w - 1, w_small, dtype=torch.float32, device=args.device),
        indexing="ij",
    )
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, uv.shape[0], args.pixel_batch_size):
            render = render_color_samples(
                nemo.field,
                frame.intrinsics,
                frame.world_T_camera,
                uv[start : start + args.pixel_batch_size],
                t_near=args.t_near,
                t_far=resolved_t_far,
                num_bracket_samples=args.num_bracket_samples,
                num_bisection_steps=args.num_bisection_steps,
                num_newton_steps=args.num_newton_steps,
            )
            chunks.append(render.rgb.detach().cpu())
    pred = torch.cat(chunks, dim=0).reshape(h_small, w_small, 3).numpy()
    target = (
        torch.nn.functional.interpolate(
            frame.image.permute(2, 0, 1).unsqueeze(0),
            size=(h_small, w_small),
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )
    canvas = np.concatenate([target, pred], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(canvas, 0.0, 1.0))


def _psnr(loss: torch.Tensor) -> float:
    return float(-10.0 * torch.log10(loss.clamp_min(1e-8)).item())


def _auto_t_far(dataset) -> float:
    sparse = dataset.sparse_points
    xyz_min = sparse.min(axis=0)
    xyz_max = sparse.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = float(np.linalg.norm(xyz_max - xyz_min) * 0.5)
    camera_positions = np.stack([frame.world_T_camera[:3, 3] for frame in dataset.frames], axis=0)
    max_camera_dist = float(np.max(np.linalg.norm(camera_positions - center[None, :], axis=-1)))
    return max(2.0 * radius + max_camera_dist, 100.0)


def _supports_image_geometry_gradients(nemo: Nemo) -> bool:
    field = nemo.field
    if isinstance(field, SmoothGridHeightField) and field.residual_type == "grid":
        return False
    return True


def _scheduled_weight(step: int, total_steps: int, *, start: float, end: float) -> float:
    if total_steps <= 1:
        return float(end)
    alpha = float(step - 1) / float(total_steps - 1)
    return (1.0 - alpha) * float(start) + alpha * float(end)


def _sample_xy_in_bounds(nemo: Nemo, batch_size: int, *, device: str) -> torch.Tensor:
    bounds = nemo.field.bounds
    x = torch.rand(batch_size, device=device) * float(bounds[0][1] - bounds[0][0]) + float(bounds[0][0])
    y = torch.rand(batch_size, device=device) * float(bounds[1][1] - bounds[1][0]) + float(bounds[1][0])
    return torch.stack([x, y], dim=-1)


def _geometry_smoothness_loss(nemo: Nemo, xy: torch.Tensor) -> torch.Tensor:
    xy = xy.clone().detach().requires_grad_(True)
    _, grad = nemo.field.h_and_grad(xy, create_graph=True)
    d2x_full = torch.autograd.grad(grad[:, 0].sum(), xy, create_graph=True, allow_unused=True)[0]
    d2y_full = torch.autograd.grad(grad[:, 1].sum(), xy, create_graph=True, allow_unused=True)[0]
    if d2x_full is None:
        d2x = torch.zeros(xy.shape[0], dtype=xy.dtype, device=xy.device)
    else:
        d2x = d2x_full[:, 0]
    if d2y_full is None:
        d2y = torch.zeros(xy.shape[0], dtype=xy.dtype, device=xy.device)
    else:
        d2y = d2y_full[:, 1]
    laplacian = d2x + d2y
    return torch.mean(laplacian * laplacian)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
