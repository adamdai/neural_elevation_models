"""Render a standalone NEMo height-field surface plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from nemo import Nemo
from scripts.compare_physical_objectives import sample_airsim_grid


DEFAULT_CHECKPOINT = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
)


def camera_from_controls(*, eye_distance: float, yaw_deg: float, pitch_deg: float) -> dict:
    eye = np.asarray([1.3, -1.5, 0.9], dtype=np.float64)
    yaw = np.arctan2(eye[1], eye[0]) + np.deg2rad(float(yaw_deg))
    pitch = np.arctan2(eye[2], np.linalg.norm(eye[:2])) + np.deg2rad(float(pitch_deg))
    pitch = float(np.clip(pitch, np.deg2rad(-89.0), np.deg2rad(89.0)))

    direction = np.asarray(
        [
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ],
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
    terrain: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    colorscale: str,
    cmin_percentile: float,
    cmax_percentile: float,
    width: int,
    height: int,
    show_colorbar: bool,
    eye_distance: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
) -> go.Figure:
    as_x, as_y, as_z, _slope = terrain
    zmin = float(np.nanpercentile(as_z, cmin_percentile))
    zmax = float(np.nanpercentile(as_z, cmax_percentile))

    fig = go.Figure(
        data=[
            go.Surface(
                x=as_x,
                y=as_y,
                z=as_z,
                surfacecolor=as_z,
                colorscale=colorscale,
                cmin=zmin,
                cmax=zmax,
                showscale=show_colorbar,
                lighting=dict(
                    ambient=1.0,
                    diffuse=0.0,
                    specular=0.0,
                    roughness=1.0,
                    fresnel=0.0,
                ),
                colorbar=dict(title="Elevation<br>up (m)", thickness=16),
                name="NEMo surface",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nemo_surface"))
    parser.add_argument("--html-name", type=str, default="nemo_height_field_surface.html")
    parser.add_argument("--terrain-res", type=int, default=320)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1300)
    parser.add_argument("--colorscale", type=str, default="Viridis")
    parser.add_argument("--cmin-percentile", type=float, default=1.0)
    parser.add_argument("--cmax-percentile", type=float, default=99.0)
    parser.add_argument("--hide-colorbar", action="store_true")
    parser.add_argument("--eye-distance", type=float, default=0.42)
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    args = parser.parse_args()

    if int(args.terrain_res) < 2:
        raise ValueError("--terrain-res must be at least 2")
    if not 0.0 <= float(args.cmin_percentile) <= 100.0:
        raise ValueError("--cmin-percentile must be in [0, 100]")
    if not 0.0 <= float(args.cmax_percentile) <= 100.0:
        raise ValueError("--cmax-percentile must be in [0, 100]")
    if float(args.cmax_percentile) <= float(args.cmin_percentile):
        raise ValueError("--cmax-percentile must be greater than --cmin-percentile")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / args.html_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(args.checkpoint.expanduser(), map_location=device).to(device)
    terrain = sample_airsim_grid(nemo, res=int(args.terrain_res))
    fig = build_figure(
        terrain=terrain,
        colorscale=str(args.colorscale),
        cmin_percentile=float(args.cmin_percentile),
        cmax_percentile=float(args.cmax_percentile),
        width=int(args.width),
        height=int(args.height),
        show_colorbar=not bool(args.hide_colorbar),
        eye_distance=float(args.eye_distance),
        camera_yaw_deg=float(args.camera_yaw_deg),
        camera_pitch_deg=float(args.camera_pitch_deg),
    )
    fig.write_html(html_path, include_plotlyjs=True, config={"responsive": True})

    metadata = {
        "checkpoint": str(args.checkpoint.expanduser()),
        "html_path": str(html_path),
        "terrain_res": int(args.terrain_res),
        "width": int(args.width),
        "height": int(args.height),
        "colorscale": str(args.colorscale),
        "cmin_percentile": float(args.cmin_percentile),
        "cmax_percentile": float(args.cmax_percentile),
        "show_colorbar": not bool(args.hide_colorbar),
    }
    (output_dir / "nemo_height_field_surface_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(html_path)


if __name__ == "__main__":
    main()
