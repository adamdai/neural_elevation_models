"""Bake Terrain-Nerfacto appearance onto the learned NEMo height-field mesh."""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
import torch


TERRAIN_NERF_ROOT = Path("/home/addai/NeRF/terrain-nerf")
NERFSTUDIO_ROOT = Path("/home/addai/NeRF/nerfstudio")
DEFAULT_CONFIG = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000/config.yml"
)
AIRSIM_SPIRAL_CENTER_X = 524.38
AIRSIM_SPIRAL_CENTER_Y = 168.34
AIRSIM_Z_OFFSET = 24.15


def load_pipeline(config_path: Path, *, terrain_nerf_root: Path, nerfstudio_root: Path):
    terrain_nerf_root = terrain_nerf_root.expanduser().resolve()
    if str(terrain_nerf_root) not in sys.path:
        sys.path.insert(0, str(terrain_nerf_root))

    import terrain_nerf  # noqa: F401
    from nerfstudio.utils.eval_utils import eval_setup

    old_cwd = Path.cwd()
    try:
        os.chdir(nerfstudio_root.expanduser().resolve())
        _, pipeline, checkpoint_path, _ = eval_setup(config_path)
    finally:
        os.chdir(old_cwd)
    return pipeline, checkpoint_path


def height_bounds_from_model(model) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if getattr(model, "height_bounds", None) is None:
        raise RuntimeError("Loaded model does not expose height_bounds.")
    x_bounds = (float(model.height_bounds[0][0]), float(model.height_bounds[0][1]))
    y_bounds = (float(model.height_bounds[1][0]), float(model.height_bounds[1][1]))
    dem_points = getattr(model, "dem_points_cpu", None)
    if dem_points is None:
        z_bounds = (-1000.0, 1000.0)
    else:
        z = dem_points[:, 2]
        z_bounds = (float(z.min().item()), float(z.max().item()))
    return x_bounds, y_bounds, z_bounds


@torch.no_grad()
def sample_height_field(model, *, resolution: int, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if getattr(model, "height_field", None) is None:
        raise RuntimeError("Loaded model does not expose a trained height_field.")

    device = model.device
    (x_min, x_max), (y_min, y_max), _ = height_bounds_from_model(model)
    xs = torch.linspace(x_min, x_max, int(resolution), device=device)
    ys = torch.linspace(y_min, y_max, int(resolution), device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    heights = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, xy.shape[0], max(int(batch_size), 1)):
            heights.append(model.height_field.h(xy[start : start + int(batch_size)]).detach().float().cpu())
    finally:
        model.train(was_training)

    z = torch.cat(heights, dim=0).reshape(int(resolution), int(resolution)).numpy()
    return xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), z


@torch.no_grad()
def render_nerf_rgb_for_direction(
    model,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    direction_raw: np.ndarray,
    batch_size: int,
    z_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    from nerfstudio.cameras.rays import RayBundle

    if getattr(model, "raw_to_nerf_points", None) is None or getattr(model, "nerf_to_raw_points", None) is None:
        raise RuntimeError("Model does not expose raw<->nerf coordinate transforms.")

    device = model.device
    _, _, (z_min, z_max) = height_bounds_from_model(model)
    z_top = z_max + float(z_margin)
    z_bottom = z_min - float(z_margin)
    direction = np.asarray(direction_raw, dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-8:
        raise ValueError("Ray direction must be nonzero.")
    direction = direction / direction_norm
    if float(direction[2]) >= -1e-5:
        raise ValueError("Ray direction must point downward in raw-world z.")

    x_flat = torch.as_tensor(x.reshape(-1), dtype=torch.float32, device=device)
    y_flat = torch.as_tensor(y.reshape(-1), dtype=torch.float32, device=device)
    z_flat = torch.as_tensor(z.reshape(-1), dtype=torch.float32, device=device)
    direction_t = torch.as_tensor(direction, dtype=torch.float32, device=device)
    t_before = (float(z_top) - z_flat) / max(abs(float(direction[2])), 1e-6)
    t_after = (z_flat - float(z_bottom)) / max(abs(float(direction[2])), 1e-6)
    surface_raw = torch.stack([x_flat, y_flat, z_flat], dim=-1)
    origins_raw = surface_raw - direction_t[None, :] * t_before[:, None]
    ends_raw = surface_raw + direction_t[None, :] * t_after[:, None]
    origins_nerf = model.raw_to_nerf_points(origins_raw)
    ends_nerf = model.raw_to_nerf_points(ends_raw)
    ray_vec = ends_nerf - origins_nerf
    ray_len = torch.linalg.norm(ray_vec, dim=-1, keepdim=True).clamp_min(1e-8)
    directions_nerf = ray_vec / ray_len

    rgb_chunks = []
    accumulation_chunks = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, origins_nerf.shape[0], max(int(batch_size), 1)):
            end = min(start + max(int(batch_size), 1), origins_nerf.shape[0])
            ray_bundle = RayBundle(
                origins=origins_nerf[start:end],
                directions=directions_nerf[start:end],
                pixel_area=torch.ones((end - start, 1), dtype=torch.float32, device=device),
                nears=torch.zeros((end - start, 1), dtype=torch.float32, device=device),
                fars=ray_len[start:end].to(dtype=torch.float32),
                camera_indices=torch.zeros((end - start, 1), dtype=torch.long, device=device),
                metadata={},
            )
            outputs = model.forward(ray_bundle=ray_bundle)
            if "rgb" not in outputs:
                raise KeyError(f"Model outputs do not contain 'rgb'. Available: {list(outputs)}")
            rgb_chunks.append(outputs["rgb"].detach().float().cpu())
            accumulation = outputs.get("accumulation")
            if accumulation is None:
                accumulation = torch.ones((end - start, 1), dtype=torch.float32, device=device)
            accumulation_chunks.append(accumulation.reshape(-1, 1).detach().float().cpu())
    finally:
        model.train(was_training)

    rgb = torch.cat(rgb_chunks, dim=0).reshape(*x.shape, -1).numpy()[..., :3]
    accumulation = torch.cat(accumulation_chunks, dim=0).reshape(x.shape).numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, accumulation


def make_ray_directions(
    *,
    include_vertical: bool,
    num_oblique_rays: int,
    oblique_angle_deg: float,
    azimuth_offset_deg: float,
) -> np.ndarray:
    directions = []
    if include_vertical:
        directions.append(np.asarray([0.0, 0.0, -1.0], dtype=np.float32))

    count = max(int(num_oblique_rays), 0)
    if count > 0:
        angle = np.deg2rad(float(oblique_angle_deg))
        horizontal = float(np.sin(angle))
        vertical = -float(np.cos(angle))
        azimuth_offset = np.deg2rad(float(azimuth_offset_deg))
        for idx in range(count):
            azimuth = azimuth_offset + 2.0 * np.pi * float(idx) / float(count)
            directions.append(
                np.asarray(
                    [
                        horizontal * np.cos(azimuth),
                        horizontal * np.sin(azimuth),
                        vertical,
                    ],
                    dtype=np.float32,
                )
            )

    if not directions:
        raise ValueError("At least one ray direction must be enabled.")
    return np.stack(directions, axis=0)


@torch.no_grad()
def render_multidirection_nerf_rgb(
    model,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    directions_raw: np.ndarray,
    batch_size: int,
    z_margin: float,
    accumulation_threshold: float,
    color_aggregation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_sum = np.zeros((*x.shape, 3), dtype=np.float64)
    weight_sum = np.zeros(x.shape, dtype=np.float64)
    accumulation_values = []

    fallback_rgb = None
    for direction in directions_raw:
        rgb, accumulation = render_nerf_rgb_for_direction(
            model,
            x,
            y,
            z,
            direction_raw=direction,
            batch_size=batch_size,
            z_margin=z_margin,
        )
        if fallback_rgb is None:
            fallback_rgb = rgb

        valid = accumulation >= float(accumulation_threshold)
        if color_aggregation == "mean":
            weights = valid.astype(np.float64)
        elif color_aggregation == "accumulation":
            weights = np.where(valid, accumulation, 0.0).astype(np.float64)
        else:
            raise ValueError("color_aggregation must be 'accumulation' or 'mean'.")

        rgb_sum += rgb.astype(np.float64) * weights[..., None]
        weight_sum += weights
        accumulation_values.append(accumulation)

    rgb_out = np.zeros((*x.shape, 3), dtype=np.float32)
    valid_any = weight_sum > 0.0
    rgb_out[valid_any] = (rgb_sum[valid_any] / weight_sum[valid_any, None]).astype(np.float32)
    if fallback_rgb is not None and np.any(~valid_any):
        rgb_out[~valid_any] = np.clip(fallback_rgb[~valid_any] * 0.25, 0.0, 1.0)

    accumulation_stack = np.stack(accumulation_values, axis=0)
    accumulation_mean = np.mean(accumulation_stack, axis=0)
    valid_ray_count = np.sum(accumulation_stack >= float(accumulation_threshold), axis=0)
    return np.clip(rgb_out, 0.0, 1.0), accumulation_mean, valid_ray_count


def build_faces(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.meshgrid(np.arange(resolution - 1), np.arange(resolution - 1), indexing="ij")
    v00 = rows * resolution + cols
    v01 = v00 + 1
    v10 = v00 + resolution
    v11 = v10 + 1
    i = np.concatenate([v00.reshape(-1), v00.reshape(-1)])
    j = np.concatenate([v10.reshape(-1), v11.reshape(-1)])
    k = np.concatenate([v11.reshape(-1), v01.reshape(-1)])
    return i.astype(np.int32), j.astype(np.int32), k.astype(np.int32)


def dataset_grid_to_airsim(
    x_dataset: np.ndarray, y_dataset: np.ndarray, z_dataset: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_airsim = y_dataset + AIRSIM_SPIRAL_CENTER_X
    y_airsim = x_dataset + AIRSIM_SPIRAL_CENTER_Y
    z_airsim = z_dataset - AIRSIM_Z_OFFSET
    return x_airsim, y_airsim, z_airsim


def camera_from_controls(*, eye_distance: float, yaw_deg: float, pitch_deg: float) -> dict[str, dict[str, float]]:
    eye = np.asarray([1.3, -1.5, 0.9], dtype=np.float64)
    yaw = np.arctan2(eye[1], eye[0]) + np.deg2rad(float(yaw_deg))
    pitch = np.arctan2(eye[2], np.linalg.norm(eye[:2])) + np.deg2rad(float(pitch_deg))
    pitch = float(np.clip(pitch, np.deg2rad(-89.0), np.deg2rad(89.0)))
    direction = np.asarray(
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
        dtype=np.float64,
    )
    camera_eye = direction * float(eye_distance)
    return {
        "eye": {"x": float(camera_eye[0]), "y": float(camera_eye[1]), "z": float(camera_eye[2])},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


def build_figure(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rgb: np.ndarray,
    width: int,
    height: int,
    eye_distance: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
) -> go.Figure:
    i, j, k = build_faces(x.shape[0])
    colors = np.clip(np.round(rgb.reshape(-1, 3) * 255.0), 0, 255).astype(np.uint8)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x.reshape(-1),
                y=y.reshape(-1),
                z=z.reshape(-1),
                i=i,
                j=j,
                k=k,
                vertexcolor=colors,
                flatshading=False,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
                name="NeRF-baked NEMo surface",
            )
        ]
    )
    clean_axis = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
        title="",
    )
    fig.update_layout(
        height=height,
        width=width,
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        scene=dict(
            aspectmode="data",
            xaxis=clean_axis,
            yaxis=dict(autorange="reversed", **clean_axis),
            zaxis=clean_axis,
            camera=camera_from_controls(
                eye_distance=eye_distance,
                yaw_deg=camera_yaw_deg,
                pitch_deg=camera_pitch_deg,
            ),
        ),
    )
    return fig


def write_binary_ply(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, rgb: np.ndarray) -> None:
    i, j, k = build_faces(x.shape[0])
    vertices = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1).astype(np.float32)
    colors = np.clip(np.round(rgb.reshape(-1, 3) * 255.0), 0, 255).astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {i.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        for vertex, color in zip(vertices, colors, strict=True):
            f.write(struct.pack("<fffBBB", float(vertex[0]), float(vertex[1]), float(vertex[2]), int(color[0]), int(color[1]), int(color[2])))
        for face in zip(i, j, k, strict=True):
            f.write(struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2])))


def start_preview_server(directory: Path, *, host: str) -> tuple[subprocess.Popen[bytes], str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = int(sock.getsockname()[1])
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", host, "--directory", str(directory)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process, f"http://{host}:{port}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nerf_textured_surface"))
    parser.add_argument("--html-name", type=str, default="nerf_textured_height_field.html")
    parser.add_argument("--ply-name", type=str, default="nerf_textured_height_field.ply")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--height-batch-size", type=int, default=65536)
    parser.add_argument("--ray-batch-size", type=int, default=16384)
    parser.add_argument("--accumulation-threshold", type=float, default=0.15)
    parser.add_argument("--z-margin", type=float, default=20.0)
    parser.add_argument("--num-oblique-rays", type=int, default=4)
    parser.add_argument("--oblique-angle-deg", type=float, default=35.0)
    parser.add_argument("--azimuth-offset-deg", type=float, default=45.0)
    parser.add_argument("--no-vertical-ray", action="store_true")
    parser.add_argument("--color-aggregation", choices=("accumulation", "mean"), default="accumulation")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1300)
    parser.add_argument("--eye-distance", type=float, default=0.42)
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--coordinate-frame", choices=("airsim", "dataset"), default="airsim")
    parser.add_argument("--terrain-nerf-root", type=Path, default=TERRAIN_NERF_ROOT)
    parser.add_argument("--nerfstudio-root", type=Path, default=NERFSTUDIO_ROOT)
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / args.html_name
    ply_path = output_dir / args.ply_name

    pipeline, checkpoint_path = load_pipeline(
        config_path,
        terrain_nerf_root=args.terrain_nerf_root,
        nerfstudio_root=args.nerfstudio_root,
    )
    model = pipeline.model
    x, y, z = sample_height_field(
        model,
        resolution=max(int(args.resolution), 2),
        batch_size=max(int(args.height_batch_size), 1),
    )
    directions = make_ray_directions(
        include_vertical=not bool(args.no_vertical_ray),
        num_oblique_rays=int(args.num_oblique_rays),
        oblique_angle_deg=float(args.oblique_angle_deg),
        azimuth_offset_deg=float(args.azimuth_offset_deg),
    )
    rgb, accumulation, valid_ray_count = render_multidirection_nerf_rgb(
        model,
        x,
        y,
        z,
        directions_raw=directions,
        batch_size=max(int(args.ray_batch_size), 1),
        z_margin=float(args.z_margin),
        accumulation_threshold=float(args.accumulation_threshold),
        color_aggregation=str(args.color_aggregation),
    )
    if str(args.coordinate_frame) == "airsim":
        x_plot, y_plot, z_plot = dataset_grid_to_airsim(x, y, z)
    else:
        x_plot, y_plot, z_plot = x, y, z

    fig = build_figure(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        rgb=rgb,
        width=int(args.width),
        height=int(args.height),
        eye_distance=float(args.eye_distance),
        camera_yaw_deg=float(args.camera_yaw_deg),
        camera_pitch_deg=float(args.camera_pitch_deg),
    )
    fig.write_html(html_path, include_plotlyjs=True, config={"responsive": True})
    np.savez_compressed(
        output_dir / "nerf_textured_height_field_grid.npz",
        x=x_plot,
        y=y_plot,
        z=z_plot,
        x_dataset=x,
        y_dataset=y,
        z_dataset=z,
        rgb=rgb,
        accumulation=accumulation,
        valid_ray_count=valid_ray_count,
        ray_directions_raw=directions,
        coordinate_frame=str(args.coordinate_frame),
    )
    if not bool(args.no_ply):
        write_binary_ply(ply_path, x_plot, y_plot, z_plot, rgb)

    metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "html_path": str(html_path),
        "ply_path": None if bool(args.no_ply) else str(ply_path),
        "coordinate_frame": str(args.coordinate_frame),
        "dataset_to_airsim": {
            "as_x": "ds_y + 524.38",
            "as_y": "ds_x + 168.34",
            "as_z": "ds_z - 24.15",
        },
        "resolution": int(args.resolution),
        "height_batch_size": int(args.height_batch_size),
        "ray_batch_size": int(args.ray_batch_size),
        "accumulation_threshold": float(args.accumulation_threshold),
        "z_margin": float(args.z_margin),
        "ray_directions_raw": directions.tolist(),
        "num_ray_directions": int(directions.shape[0]),
        "num_oblique_rays": int(args.num_oblique_rays),
        "oblique_angle_deg": float(args.oblique_angle_deg),
        "azimuth_offset_deg": float(args.azimuth_offset_deg),
        "include_vertical_ray": not bool(args.no_vertical_ray),
        "color_aggregation": str(args.color_aggregation),
        "valid_rgb_fraction": float(np.mean(valid_ray_count > 0)),
        "mean_valid_ray_count": float(np.mean(valid_ray_count)),
        "height_range": [float(np.nanmin(z_plot)), float(np.nanmax(z_plot))],
        "dataset_height_range": [float(np.nanmin(z)), float(np.nanmax(z))],
        "accumulation_range": [float(np.nanmin(accumulation)), float(np.nanmax(accumulation))],
    }
    (output_dir / "nerf_textured_height_field_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))

    if not bool(args.no_serve):
        server, base_url = start_preview_server(output_dir, host="127.0.0.1")
        url = f"{base_url}/{quote(html_path.name)}"
        print(f"Plot preview: {url}")
        print(f"Preview server PID: {server.pid}")
        if not bool(args.no_open):
            print(f"Auto-open requested: {webbrowser.open(url, new=2)}")


if __name__ == "__main__":
    main()
