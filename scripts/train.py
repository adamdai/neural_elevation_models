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
    pixel_batch_size: int = 4096
    eval_every: int = 25
    eval_pixel_count: int = 8192
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

    if args.fit_geometry:
        _fit_geometry(nemo, dataset, args)

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
        frame = dataset.frames[random.randrange(len(dataset.frames))]
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
        )

        hit_mask = render.hit_mask
        if torch.any(hit_mask):
            pred = render.rgb[hit_mask]
            target = target_rgb[hit_mask]
            color_loss = torch.nn.functional.mse_loss(pred, target)
            psnr = _psnr(color_loss.detach())
            hit_fraction = float(hit_mask.float().mean().item())
        else:
            skipped_no_hit += 1
            if step == 1 or skipped_no_hit <= 5 or skipped_no_hit % 25 == 0:
                print(f"[train] step={step} skipped batch with no ray hits")
            continue

        optimizer.zero_grad(set_to_none=True)
        color_loss.backward()
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
                "train_loss": float(color_loss.item()),
                "train_psnr": float(psnr),
                "train_hit_fraction": hit_fraction,
                "skipped_no_hit": skipped_no_hit,
                **eval_metrics,
            }
            metrics.append(record)
            print(
                f"[train] step={step} "
                f"train_loss={record['train_loss']:.6f} "
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
    xy = torch.from_numpy(points[:, :2].astype(np.float32))
    z = torch.from_numpy(points[:, 2:3].astype(np.float32))
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
