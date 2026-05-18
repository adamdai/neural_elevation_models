"""Render NEMo height field from a specific camera perspective."""

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
    parser.add_argument("--config", type=Path, default=Path("/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-16_robust-depth-dem100_5000/config.yml"))
    parser.add_argument("--camera-idx", type=int, default=20)
    parser.add_argument("--nerfstudio-root", type=Path, default=Path("/home/addai/NeRF/nerfstudio"))
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_diagnostics/camera_20_nemo.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    terrain_nerf_root = Path("/home/addai/NeRF/terrain-nerf").resolve()
    nerfstudio_root = args.nerfstudio_root.expanduser().resolve()
    
    sys.path.insert(0, str(terrain_nerf_root))
    sys.path.insert(0, str(nerfstudio_root))
    os.chdir(terrain_nerf_root)

    import terrain_nerf  # noqa: F401
    from nerfstudio.utils.eval_utils import eval_setup

    config, pipeline, _, _ = eval_setup(args.config.expanduser(), test_mode="val")
    model = pipeline.model.eval()
    datamanager = pipeline.datamanager

    # Load Camera
    camera = datamanager.train_dataset.cameras[args.camera_idx : args.camera_idx + 1].to(model.device)
    
    print(f"Rendering camera {args.camera_idx} perspective...")
    with torch.no_grad():
        # model._render_height_camera returns (H, W, 3) normalized [0, 1]
        nemo_render = model._render_height_camera(camera)
        
    # Convert to 8-bit image
    img_np = (nemo_render.cpu().numpy() * 255).astype(np.uint8)
    
    # Ensure output dir exists
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(img_np).save(output_path)
    print(f"Render saved to {output_path}")
    
    # Also save the raw height values for inspection if possible
    # We can get them by calling the internal method directly
    ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True).to(model.device)
    if model.collider is not None:
        ray_bundle = model.collider(ray_bundle)
    
    height_vals = []
    num_rays = len(ray_bundle.origins.view(-1, 3))
    for start in range(0, num_rays, config.pipeline.model.eval_num_rays_per_chunk):
        chunk = ray_bundle.get_row_major_sliced_ray_bundle(start, start + config.pipeline.model.eval_num_rays_per_chunk)
        # _height_depth_for_ray_chunk returns (N,)
        # Note: this returns DEPTH along ray, not elevation.
        height_vals.append(model._height_depth_for_ray_chunk(chunk).cpu())
    
    depth_map = torch.cat(height_vals).reshape(camera.image_height.item(), camera.image_width.item()).numpy()
    np.save(output_path.with_suffix(".npy"), depth_map)
    print(f"Raw depth map saved to {output_path.with_suffix('.npy')}")


if __name__ == "__main__":
    main()
