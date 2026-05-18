"""Render NEMo from a specific camera perspective using the model's internal logic."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import mediapy as media
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-16_threshold-vertical-fix_5000/config.yml"))
    parser.add_argument("--camera-idx", type=int, default=20)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_diagnostics/camera_20_best_nemo.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    terrain_nerf_root = Path("/home/addai/NeRF/terrain-nerf").resolve()
    nerfstudio_root = Path("/home/addai/NeRF/nerfstudio").resolve()
    
    sys.path.insert(0, str(terrain_nerf_root))
    sys.path.insert(0, str(nerfstudio_root))
    os.chdir(terrain_nerf_root)

    import terrain_nerf  # noqa: F401
    from nerfstudio.utils.eval_utils import eval_setup

    # 1. Load Pipeline
    config, pipeline, _, _ = eval_setup(args.config.expanduser(), test_mode="val")
    model = pipeline.model.eval()
    datamanager = pipeline.datamanager

    # 2. Get Camera
    camera = datamanager.train_dataset.cameras[args.camera_idx : args.camera_idx + 1].to(model.device)
    if args.downscale != 1.0:
        camera.rescale_output_resolution(1.0 / args.downscale)
    
    print(f"Rendering high quality NEMo view for camera {args.camera_idx}...")
    
    with torch.no_grad():
        # model._render_height_camera returns (H, W, 3) normalized [0, 1]
        # This method performs ray-casting against the NEMo surface
        nemo_render = model._render_height_camera(camera)
        
        # Also get RGB for reference if possible
        outputs = model.get_outputs_for_camera(camera)
        rgb_render = outputs["rgb"]

    # Convert to 8-bit
    nemo_img = (nemo_render.cpu().numpy() * 255).astype(np.uint8)
    rgb_img = (rgb_render.cpu().numpy() * 255).astype(np.uint8)
    
    # Combined frame (Side by side)
    combined = np.concatenate([rgb_img, nemo_img], axis=1)
    
    # Save
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(combined).save(output_path)
    print(f"Render (RGB + NEMo) saved to {output_path}")

if __name__ == "__main__":
    main()
