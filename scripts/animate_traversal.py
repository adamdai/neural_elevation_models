"""Animate a marker traversing a saved optimized terrain path."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from urllib.parse import quote

import mediapy as media
import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from nemo import Nemo
from scripts.compare_physical_objectives import sample_airsim_grid
from scripts.crater_test import start_preview_server
from scripts.terrain_aware_planner import PATH_PLOT_Z_OFFSET_M


DEFAULT_CHECKPOINT = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
)
DEFAULT_PATH_DIR = Path("outputs/path_opt_animation")

ASTAR_CANDIDATES = (
    "airsim_path_opt_astar_path.npy",
    "airsim_terrain_aware_astar_path.npy",
    "airsim_astar_seed.npy",
    "astar_seed.npy",
    "airsim_crater_astar_path.npy",
    "airsim_crater_basic_astar_path.npy",
    "airsim_crater_gt_dem_astar_path.npy",
)
OPTIMIZED_CANDIDATES = (
    "airsim_path_opt_path.npy",
    "airsim_terrain_aware_path.npy",
    "airsim_energy_path.npy",
    "airsim_crater_path.npy",
    "airsim_crater_basic_path.npy",
    "airsim_crater_gt_dem_path.npy",
    "airsim_path.npy",
    "airsim_path_opt_control_path.npy",
    "airsim_terrain_aware_control_path.npy",
)


def camera_from_controls(
    *,
    eye_distance: float,
    yaw_deg: float,
    pitch_deg: float,
    pan_right: float,
    pan_up: float,
    pan_forward: float,
) -> dict[str, dict[str, float]]:
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

    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(direction, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= right_norm
    up = np.cross(right, direction)
    up /= np.linalg.norm(up)

    center = right * float(pan_right) + up * float(pan_up) + direction * float(pan_forward)
    return {
        "eye": {"x": float(camera_eye[0]), "y": float(camera_eye[1]), "z": float(camera_eye[2])},
        "up": {"x": float(up[0]), "y": float(up[1]), "z": float(up[2])},
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
    }


def camera_from_json(value: str) -> dict[str, dict[str, float]]:
    path = Path(value).expanduser()
    text = path.read_text(encoding="utf-8") if path.exists() else value
    payload = json.loads(text)
    if "scene.camera" in payload:
        payload = payload["scene.camera"]
    if "camera" in payload and {"eye", "up", "center"}.issubset(payload["camera"]):
        payload = payload["camera"]

    camera: dict[str, dict[str, float]] = {}
    for key in ("eye", "up", "center"):
        if key not in payload:
            raise ValueError(f"Camera JSON is missing '{key}'")
        vector = payload[key]
        camera[key] = {axis: float(vector[axis]) for axis in ("x", "y", "z")}
    return camera


def first_existing(path_dir: Path, names: tuple[str, ...], *, label: str) -> Path:
    for name in names:
        candidate = path_dir / name
        if candidate.exists():
            return candidate
    joined = ", ".join(names)
    raise FileNotFoundError(f"Could not find {label} path in {path_dir}. Tried: {joined}")


def load_path(path: Path, *, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    points = np.asarray(np.load(path), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{name} path must have shape (N, >=3), got {points.shape}")
    if points.shape[0] < 2:
        raise ValueError(f"{name} path must contain at least two points, got {points.shape[0]}")
    return points[:, :3]


def interpolate_path_by_distance(path: np.ndarray, num_samples: int) -> np.ndarray:
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    if total <= 1e-8:
        return np.repeat(path[:1], repeats=num_samples, axis=0)
    targets = np.linspace(0.0, total, num_samples)
    out = np.empty((num_samples, 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(targets, cumulative, path[:, axis])
    return out


def build_figure(
    *,
    terrain: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    astar_airsim: np.ndarray,
    optimized_airsim: np.ndarray,
    marker_airsim: np.ndarray,
    marker_color: str,
    marker_size: float,
    show_markers: bool,
    eye_distance: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    camera_pan_right: float,
    camera_pan_up: float,
    camera_pan_forward: float,
    camera_json: str | None,
    height: int,
    width: int,
    astar_line_width: float,
    optimized_line_width: float,
) -> go.Figure:
    as_x, as_y, as_z, _slope = terrain
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=as_x,
            y=as_y,
            z=as_z,
            surfacecolor=as_z,
            colorscale="Viridis",
            cmin=float(np.nanpercentile(as_z, 1.0)),
            cmax=float(np.nanpercentile(as_z, 99.0)),
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            colorbar=dict(title="Elevation<br>up (m)", thickness=16),
            name="NEMo surface",
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=astar_airsim[:, 0],
            y=astar_airsim[:, 1],
            z=astar_airsim[:, 2] + PATH_PLOT_Z_OFFSET_M,
            mode="lines+markers",
            line=dict(color="red", width=astar_line_width, dash="dash"),
            marker=dict(color="red", size=3),
            name="A* seed",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=optimized_airsim[:, 0],
            y=optimized_airsim[:, 1],
            z=optimized_airsim[:, 2] + PATH_PLOT_Z_OFFSET_M,
            mode="lines",
            line=dict(color="orange", width=optimized_line_width),
            name="optimized path",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[marker_airsim[0]],
            y=[marker_airsim[1]],
            z=[marker_airsim[2] + PATH_PLOT_Z_OFFSET_M],
            mode="markers",
            marker=dict(color=marker_color, size=marker_size, symbol="circle"),
            name="traversal marker",
        )
    )

    if show_markers:
        fig.add_trace(
            go.Scatter3d(
                x=[optimized_airsim[0, 0]],
                y=[optimized_airsim[0, 1]],
                z=[optimized_airsim[0, 2] + PATH_PLOT_Z_OFFSET_M],
                mode="markers",
                marker=dict(color="#00ffff", size=10, symbol="circle"),
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[optimized_airsim[-1, 0]],
                y=[optimized_airsim[-1, 1]],
                z=[optimized_airsim[-1, 2] + PATH_PLOT_Z_OFFSET_M],
                mode="markers",
                marker=dict(color="#ff00ff", size=12, symbol="diamond"),
                name="Goal",
            )
        )

    clean_axis = dict(
        showbackground=False,
        backgroundcolor="black",
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
        title="",
    )
    if camera_json:
        camera = camera_from_json(camera_json)
    else:
        camera = camera_from_controls(
            eye_distance=eye_distance,
            yaw_deg=camera_yaw_deg,
            pitch_deg=camera_pitch_deg,
            pan_right=camera_pan_right,
            pan_up=camera_pan_up,
            pan_forward=camera_pan_forward,
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
            camera=camera,
        ),
    )
    return fig


def save_frame(fig: go.Figure, path: Path, *, width: int, height: int, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(path, width=width, height=height, scale=scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--path-dir", type=Path, default=DEFAULT_PATH_DIR)
    parser.add_argument("--astar-path", type=Path, default=None)
    parser.add_argument("--optimized-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/animate_traversal"))
    parser.add_argument("--frames-dir-name", type=str, default="frames")
    parser.add_argument("--video-filename", type=str, default="traversal_animation.mp4")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=1500)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--terrain-res", type=int, default=240)
    parser.add_argument("--astar-line-width", type=float, default=7.0)
    parser.add_argument("--optimized-line-width", type=float, default=11.0)
    parser.add_argument("--video-crf", type=float, default=14.0)
    parser.add_argument("--video-encoded-format", type=str, default="yuv420p")
    parser.add_argument("--marker-color", type=str, default="#003f88")
    parser.add_argument("--marker-size", type=float, default=9.0)
    parser.add_argument("--eye-distance", type=float, default=0.4)
    parser.add_argument(
        "--camera-yaw-deg",
        type=float,
        default=0.0,
        help="Yaw camera left/right in degrees relative to the default view.",
    )
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=0.0,
        help="Pitch camera up/down in degrees relative to the default view.",
    )
    parser.add_argument(
        "--camera-pan-right",
        type=float,
        default=0.0,
        help="Translate the camera target right in view-relative Plotly scene units.",
    )
    parser.add_argument(
        "--camera-pan-up",
        type=float,
        default=0.0,
        help="Translate the camera target up in view-relative Plotly scene units.",
    )
    parser.add_argument(
        "--camera-pan-forward",
        type=float,
        default=0.0,
        help="Translate the camera target forward along the view direction in Plotly scene units.",
    )
    parser.add_argument(
        "--camera-json",
        type=str,
        default=None,
        help="Raw Plotly camera JSON copied from scripts/render_path_opt.py's HTML panel.",
    )
    parser.add_argument(
        "--camera-json-file",
        type=Path,
        default=None,
        help="Path to raw Plotly camera JSON copied from scripts/render_path_opt.py's HTML panel.",
    )
    parser.add_argument("--show-markers", action="store_true", help="Show start and goal markers.")
    parser.add_argument(
        "--no-serve", action="store_true", help="Do not start an HTTP preview server."
    )
    args = parser.parse_args()

    if float(args.duration_seconds) <= 0.0:
        raise ValueError("--duration-seconds must be positive")
    if int(args.fps) <= 0:
        raise ValueError("--fps must be positive")

    path_dir = args.path_dir.expanduser().resolve()
    astar_path = args.astar_path or first_existing(path_dir, ASTAR_CANDIDATES, label="A*")
    optimized_path = args.optimized_path or first_existing(
        path_dir, OPTIMIZED_CANDIDATES, label="optimized"
    )
    astar_path = astar_path.expanduser().resolve()
    optimized_path = optimized_path.expanduser().resolve()

    output_dir = args.output_dir.expanduser().resolve()
    frames_dir = output_dir / args.frames_dir_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame-*.png"):
        old_frame.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(args.checkpoint.expanduser(), map_location=device).to(device)

    print(f"Sampling terrain grid at resolution {args.terrain_res}...")
    terrain = sample_airsim_grid(nemo, res=int(args.terrain_res))
    astar_airsim = load_path(astar_path, name="A*")
    optimized_airsim = load_path(optimized_path, name="optimized")
    frame_count = max(2, int(round(float(args.duration_seconds) * int(args.fps))))
    traversal_points = interpolate_path_by_distance(optimized_airsim, frame_count)
    camera_json = (
        str(args.camera_json_file.expanduser().resolve())
        if args.camera_json_file is not None
        else args.camera_json
    )

    print(f"Rendering {frame_count} traversal frames to {frames_dir}...")
    for frame_index, marker_airsim in enumerate(tqdm(traversal_points, desc="Rendering frames")):
        fig = build_figure(
            terrain=terrain,
            astar_airsim=astar_airsim,
            optimized_airsim=optimized_airsim,
            marker_airsim=marker_airsim,
            marker_color=str(args.marker_color),
            marker_size=float(args.marker_size),
            show_markers=bool(args.show_markers),
            eye_distance=float(args.eye_distance),
            camera_yaw_deg=float(args.camera_yaw_deg),
            camera_pitch_deg=float(args.camera_pitch_deg),
            camera_pan_right=float(args.camera_pan_right),
            camera_pan_up=float(args.camera_pan_up),
            camera_pan_forward=float(args.camera_pan_forward),
            camera_json=camera_json,
            height=int(args.height),
            width=int(args.width),
            astar_line_width=float(args.astar_line_width),
            optimized_line_width=float(args.optimized_line_width),
        )
        save_frame(
            fig,
            frames_dir / f"frame-{frame_index:05d}.png",
            width=int(args.width),
            height=int(args.height),
            scale=float(args.scale),
        )

    frame_paths = sorted(frames_dir.glob("frame-*.png"))
    if not frame_paths:
        raise RuntimeError("No frames were generated.")
    video_path = output_dir / args.video_filename
    first_frame = np.asarray(media.read_image(frame_paths[0]))[..., :3]
    with media.VideoWriter(
        path=video_path,
        shape=first_frame.shape[:2],
        fps=int(args.fps),
        codec="h264",
        crf=float(args.video_crf),
        encoded_format=str(args.video_encoded_format),
    ) as writer:
        writer.add_image(first_frame)
        for frame_path in frame_paths[1:]:
            writer.add_image(np.asarray(media.read_image(frame_path))[..., :3])

    metadata = {
        "checkpoint": str(args.checkpoint.expanduser()),
        "path_dir": str(path_dir),
        "astar_path": str(astar_path),
        "optimized_path": str(optimized_path),
        "video_path": str(video_path),
        "duration_seconds": float(args.duration_seconds),
        "fps": int(args.fps),
        "num_frames": len(frame_paths),
        "marker_color": str(args.marker_color),
        "marker_size": float(args.marker_size),
        "terrain_res": int(args.terrain_res),
        "width": int(args.width),
        "height": int(args.height),
        "scale": float(args.scale),
        "astar_line_width": float(args.astar_line_width),
        "optimized_line_width": float(args.optimized_line_width),
        "video_crf": float(args.video_crf),
        "video_encoded_format": str(args.video_encoded_format),
        "camera": {
            "eye_distance": float(args.eye_distance),
            "yaw_deg": float(args.camera_yaw_deg),
            "pitch_deg": float(args.camera_pitch_deg),
            "pan_right": float(args.camera_pan_right),
            "pan_up": float(args.camera_pan_up),
            "pan_forward": float(args.camera_pan_forward),
            "json": camera_json,
        },
    }
    (output_dir / "traversal_animation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved {len(frame_paths)} frames to {frames_dir}")
    print(f"Saved video to {video_path}")

    if not args.no_serve:
        server_process, base_url = start_preview_server(output_dir)
        video_url = f"{base_url}/{quote(video_path.name)}"
        print(f"Video preview: {video_url}")
        print(f"Preview server PID: {server_process.pid}")
        webbrowser.open(video_url, new=2)


if __name__ == "__main__":
    main()
