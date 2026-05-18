"""Render a Plotly height-field view from a trained NEMo checkpoint.

The Plotly camera is an approximation of a Nerfstudio training camera. Plotly
does not expose the exact same pinhole projection controls, so this maps the
selected camera center into raw DEM coordinates and uses its relative direction
as the Plotly orbit camera.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import torch

from nemo.nemo import Nemo
from nemo.particle_sim import sample_height_field_grid


DEFAULT_RUN = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "nemo_model.pt")
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN / "config.yml")
    parser.add_argument("--camera-idx", type=int, default=20)
    parser.add_argument("--image-filename", type=str, default="images/20.png")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--eye-distance", type=float, default=1.2)
    parser.add_argument("--output-html", type=Path, default=Path("outputs/depth_diagnostics/camera_20_plotly_height_field.html"))
    parser.add_argument("--output-png", type=Path, default=Path("outputs/depth_diagnostics/camera_20_plotly_height_field.png"))
    parser.add_argument("--serve-host", type=str, default="127.0.0.1")
    parser.add_argument("--serve-port", type=int, default=8765)
    parser.add_argument(
        "--serve-root",
        type=Path,
        default=Path("outputs/depth_diagnostics"),
        help="Directory expected to be served by python -m http.server for printed URLs.",
    )
    parser.add_argument("--terrain-nerf-root", type=Path, default=Path("/home/addai/NeRF/terrain-nerf"))
    parser.add_argument("--nerfstudio-root", type=Path, default=Path("/home/addai/NeRF/nerfstudio"))
    return parser.parse_args()


def _load_terrain_pipeline(config_path: Path, terrain_nerf_root: Path, nerfstudio_root: Path):
    sys.path.insert(0, str(terrain_nerf_root.resolve()))
    sys.path.insert(0, str(nerfstudio_root.resolve()))
    os.chdir(terrain_nerf_root.resolve())

    import terrain_nerf  # noqa: F401
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(config_path.expanduser().resolve(), test_mode="val")
    return pipeline


def _plotly_camera_from_training_camera(
    pipeline,
    camera_idx: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    eye_distance: float,
) -> dict:
    datamanager = pipeline.datamanager
    camera = datamanager.train_dataset.cameras[camera_idx : camera_idx + 1].to(pipeline.model.device)
    c2w = camera.camera_to_worlds.reshape(-1, 3, 4)[0]

    camera_raw = datamanager.nerf_to_raw_points(c2w[:3, 3].reshape(1, 3))[0]
    center_raw = torch.tensor(
        [
            0.5 * (float(np.nanmin(x)) + float(np.nanmax(x))),
            0.5 * (float(np.nanmin(y)) + float(np.nanmax(y))),
            float(np.nanmean(z)),
        ],
        dtype=camera_raw.dtype,
        device=camera_raw.device,
    )
    direction = camera_raw - center_raw
    scale = torch.linalg.norm(direction).clamp_min(1e-6)
    eye = direction / scale * float(eye_distance)

    return {
        "eye": {
            "x": float(eye[0].detach().cpu().item()),
            "y": float(eye[1].detach().cpu().item()),
            "z": float(eye[2].detach().cpu().item()),
        },
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


def _resolve_camera_idx(pipeline, *, camera_idx: int, image_filename: str | None) -> int:
    if not image_filename:
        return int(camera_idx)
    wanted = Path(image_filename)
    wanted_name = wanted.name
    wanted_suffix = str(wanted)
    train_dataset = pipeline.datamanager.train_dataset
    for idx, path in enumerate(train_dataset.image_filenames):
        path_str = str(path)
        if path.name == wanted_name or path_str.endswith(wanted_suffix):
            return idx
    raise ValueError(f"Could not find source image in training dataset: {image_filename}")


def _camera_label(pipeline, camera_idx: int) -> str:
    train_dataset = pipeline.datamanager.train_dataset
    try:
        return str(train_dataset.image_filenames[camera_idx])
    except Exception:
        return f"train camera index {camera_idx}"


def _make_figure(x: np.ndarray, y: np.ndarray, z: np.ndarray, camera: dict, *, width: int, height: int) -> go.Figure:
    finite = np.isfinite(z)
    values = z[finite] if np.any(finite) else np.array([0.0, 1.0])
    z_min = float(np.nanpercentile(values, 1))
    z_max = float(np.nanpercentile(values, 99))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        z_min = float(np.nanmin(values))
        z_max = float(np.nanmax(values) + 1e-6)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=z,
                cmin=z_min,
                cmax=z_max,
                colorscale="Viridis",
                showscale=False,
                contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                lighting={
                    "ambient": 0.62,
                    "diffuse": 0.72,
                    "fresnel": 0.08,
                    "roughness": 0.78,
                    "specular": 0.08,
                },
                lightposition={"x": -500.0, "y": -750.0, "z": 1800.0},
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        width=int(width),
        height=int(height),
        scene={
            "bgcolor": "black",
            "camera": camera,
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.08},
            "xaxis": {"visible": False, "showgrid": False, "showbackground": False, "showspikes": False},
            "yaxis": {"visible": False, "showgrid": False, "showbackground": False, "showspikes": False},
            "zaxis": {"visible": False, "showgrid": False, "showbackground": False, "showspikes": False},
        },
    )
    return fig


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    output_png = args.output_png.expanduser().resolve()
    serve_root = args.serve_root.expanduser().resolve()

    nemo = Nemo.load_checkpoint(checkpoint, map_location="cpu")
    x, y, z = sample_height_field_grid(
        nemo,
        resolution_x=int(args.resolution),
        resolution_y=int(args.resolution),
    )

    pipeline = _load_terrain_pipeline(
        config,
        args.terrain_nerf_root.expanduser(),
        args.nerfstudio_root.expanduser(),
    )
    camera_idx = _resolve_camera_idx(pipeline, camera_idx=int(args.camera_idx), image_filename=args.image_filename)
    camera = _plotly_camera_from_training_camera(
        pipeline,
        camera_idx,
        x,
        y,
        z,
        eye_distance=float(args.eye_distance),
    )
    source_image = _camera_label(pipeline, camera_idx)

    fig = _make_figure(x, y, z, camera, width=int(args.width), height=int(args.height))
    fig.update_layout(
        meta={
            "checkpoint": str(checkpoint),
            "camera_idx": camera_idx,
            "source_image": source_image,
            "plotly_camera": camera,
        }
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs=True, config={"responsive": True, "displayModeBar": True})
    fig.write_image(output_png, width=int(args.width), height=int(args.height), scale=1)

    print(f"source_image={source_image}")
    print(f"train_camera_idx={camera_idx}")
    print(f"plotly_camera={camera}")
    print(f"html={output_html}")
    print(f"png={output_png}")
    try:
        html_rel = output_html.relative_to(serve_root).as_posix()
        png_rel = output_png.relative_to(serve_root).as_posix()
        base_url = f"http://{args.serve_host}:{int(args.serve_port)}"
        print(f"html_url={base_url}/{html_rel}")
        print(f"png_url={base_url}/{png_rel}")
        print(f"serve_command=python3 -m http.server {int(args.serve_port)} --bind {args.serve_host} -d {serve_root}")
    except ValueError:
        print(f"serve_root={serve_root}")
        print("url_note=output paths are not under serve-root, so no local URLs were generated")


if __name__ == "__main__":
    main()
