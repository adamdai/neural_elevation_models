"""Render one Plotly frame of a saved path optimization result."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
import torch

from nemo import Nemo
from scripts.compare_physical_objectives import sample_airsim_grid
from scripts.crater_test import start_preview_server
from scripts.terrain_aware_planner import PATH_PLOT_Z_OFFSET_M


DEFAULT_CHECKPOINT = Path(
    "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/"
    "2026-05-16_threshold-vertical-fix_5000/nemo_model.pt"
)
DEFAULT_PATH_DIR = Path("outputs/crater_traverse")


CAMERA_PANEL_JS = r"""
(function() {
  const gd = document.getElementById('{plot_id}');
  const panel = document.createElement('div');
  panel.style.position = 'fixed';
  panel.style.right = '12px';
  panel.style.bottom = '12px';
  panel.style.zIndex = '9999';
  panel.style.width = '430px';
  panel.style.maxWidth = 'calc(100vw - 24px)';
  panel.style.maxHeight = '42vh';
  panel.style.padding = '10px';
  panel.style.background = 'rgba(0, 0, 0, 0.82)';
  panel.style.color = '#f5f5f5';
  panel.style.border = '1px solid rgba(255, 255, 255, 0.25)';
  panel.style.borderRadius = '6px';
  panel.style.font = '12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  panel.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.35)';

  const title = document.createElement('div');
  title.textContent = 'Plotly camera JSON (click box to copy)';
  title.style.marginBottom = '6px';
  title.style.fontWeight = '700';
  panel.appendChild(title);

  const pre = document.createElement('pre');
  pre.style.whiteSpace = 'pre-wrap';
  pre.style.overflow = 'auto';
  pre.style.maxHeight = '32vh';
  pre.style.margin = '0';
  panel.appendChild(pre);
  document.body.appendChild(panel);

  function roundValue(value) {
    return Number(value.toFixed(8));
  }

  function cleanVector(vec) {
    return {
      x: roundValue(Number(vec.x || 0)),
      y: roundValue(Number(vec.y || 0)),
      z: roundValue(Number(vec.z || 0)),
    };
  }

  function currentCamera() {
    const scene = gd._fullLayout && gd._fullLayout.scene;
    const camera = scene && scene.camera;
    if (!camera) {
      return {};
    }
    return {
      eye: cleanVector(camera.eye || {}),
      up: cleanVector(camera.up || {}),
      center: cleanVector(camera.center || {}),
    };
  }

  function updateCameraPanel() {
    pre.textContent = JSON.stringify(currentCamera(), null, 2);
  }

  panel.addEventListener('click', async function() {
    try {
      await navigator.clipboard.writeText(pre.textContent);
      title.textContent = 'Copied Plotly camera JSON';
      setTimeout(function() {
        title.textContent = 'Plotly camera JSON (click box to copy)';
      }, 1200);
    } catch (error) {
      title.textContent = 'Copy failed; select the JSON manually';
    }
  });

  gd.on('plotly_relayout', updateCameraPanel);
  gd.on('plotly_relayouting', updateCameraPanel);
  setTimeout(updateCameraPanel, 250);
})();
"""


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


def load_path(path: Path, *, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    points = np.asarray(np.load(path), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{name} path must have shape (N, >=3), got {points.shape}")
    return points[:, :3]


def build_figure(
    *,
    terrain: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    astar_airsim: np.ndarray | None,
    optimized_airsim: np.ndarray,
    show_markers: bool,
    eye_distance: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    camera_pan_right: float,
    camera_pan_up: float,
    camera_pan_forward: float,
    camera_json: str | None,
    dragmode: str,
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
        )
    )

    if astar_airsim is not None:
        fig.add_trace(
            go.Scatter3d(
                x=astar_airsim[:, 0],
                y=astar_airsim[:, 1],
                z=astar_airsim[:, 2] + PATH_PLOT_Z_OFFSET_M,
                mode="lines+markers",
                line=dict(color="red", width=5, dash="dash"),
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
            line=dict(color="orange", width=8),
            name="optimized path",
        )
    )

    if show_markers:
        endpoints = [optimized_airsim[0], optimized_airsim[-1]]
        fig.add_trace(
            go.Scatter3d(
                x=[endpoints[0][0]],
                y=[endpoints[0][1]],
                z=[endpoints[0][2] + PATH_PLOT_Z_OFFSET_M],
                mode="markers",
                marker=dict(color="#00ffff", size=10, symbol="circle"),
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[endpoints[1][0]],
                y=[endpoints[1][1]],
                z=[endpoints[1][2] + PATH_PLOT_Z_OFFSET_M],
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
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.35)",
        ),
        scene=dict(
            aspectmode="data",
            xaxis=clean_axis,
            yaxis=dict(autorange="reversed", **clean_axis),
            zaxis=clean_axis,
            camera=camera,
            dragmode=dragmode,
        ),
    )
    return fig


def write_camera_html(fig: go.Figure, path: Path) -> None:
    fig.write_html(
        path,
        include_plotlyjs=True,
        config={"responsive": True, "displayModeBar": True},
        post_script=CAMERA_PANEL_JS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--path-dir", type=Path, default=DEFAULT_PATH_DIR)
    parser.add_argument("--astar-path", type=Path, default=None)
    parser.add_argument("--optimized-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/render_path_opt"))
    parser.add_argument("--html-name", type=str, default="path_opt_render.html")
    parser.add_argument("--png-name", type=str, default="path_opt_render.png")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=1300)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--terrain-res", type=int, default=240)
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
        help="Raw Plotly camera JSON copied from the interactive HTML panel.",
    )
    parser.add_argument(
        "--camera-json-file",
        type=Path,
        default=None,
        help="Path to raw Plotly camera JSON copied from the interactive HTML panel.",
    )
    parser.add_argument("--show-markers", action="store_true", help="Show start and goal markers.")
    parser.add_argument(
        "--dragmode",
        choices=("pan", "orbit", "turntable", "zoom"),
        default="pan",
        help="Default 3D click-drag interaction mode for the HTML view.",
    )
    parser.add_argument("--no-png", action="store_true", help="Skip static PNG export.")
    parser.add_argument(
        "--no-serve", action="store_true", help="Do not start an HTTP preview server."
    )
    args = parser.parse_args()

    path_dir = args.path_dir.expanduser().resolve()
    astar_path = args.astar_path or path_dir / "airsim_terrain_aware_astar_path.npy"
    optimized_path = args.optimized_path or path_dir / "airsim_terrain_aware_control_path.npy"
    astar_path = astar_path.expanduser().resolve()
    optimized_path = optimized_path.expanduser().resolve()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nemo = Nemo.load_checkpoint(args.checkpoint.expanduser(), map_location=device).to(device)

    print(f"Sampling terrain grid at resolution {args.terrain_res}...")
    terrain = sample_airsim_grid(nemo, res=int(args.terrain_res))
    astar_airsim = load_path(astar_path, name="A*")
    optimized_airsim = load_path(optimized_path, name="optimized")
    fig = build_figure(
        terrain=terrain,
        astar_airsim=astar_airsim,
        optimized_airsim=optimized_airsim,
        show_markers=bool(args.show_markers),
        eye_distance=float(args.eye_distance),
        camera_yaw_deg=float(args.camera_yaw_deg),
        camera_pitch_deg=float(args.camera_pitch_deg),
        camera_pan_right=float(args.camera_pan_right),
        camera_pan_up=float(args.camera_pan_up),
        camera_pan_forward=float(args.camera_pan_forward),
        camera_json=(
            str(args.camera_json_file.expanduser().resolve())
            if args.camera_json_file is not None
            else args.camera_json
        ),
        dragmode=str(args.dragmode),
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / args.html_name
    png_path = output_dir / args.png_name
    write_camera_html(fig, html_path)
    if not args.no_png:
        fig.write_image(
            png_path, width=int(args.width), height=int(args.height), scale=float(args.scale)
        )

    metadata = {
        "checkpoint": str(args.checkpoint.expanduser()),
        "astar_path": str(astar_path),
        "optimized_path": str(optimized_path),
        "terrain_res": int(args.terrain_res),
        "width": int(args.width),
        "height": int(args.height),
        "scale": float(args.scale),
        "camera": {
            "eye_distance": float(args.eye_distance),
            "yaw_deg": float(args.camera_yaw_deg),
            "pitch_deg": float(args.camera_pitch_deg),
            "pan_right": float(args.camera_pan_right),
            "pan_up": float(args.camera_pan_up),
            "pan_forward": float(args.camera_pan_forward),
            "json": (
                str(args.camera_json_file.expanduser().resolve())
                if args.camera_json_file is not None
                else args.camera_json
            ),
        },
        "dragmode": str(args.dragmode),
    }
    (output_dir / "path_opt_render_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved HTML to {html_path}")
    if not args.no_png:
        print(f"Saved PNG to {png_path}")

    if not args.no_serve:
        server_process, base_url = start_preview_server(output_dir)
        html_url = f"{base_url}/{quote(html_path.name)}"
        print(f"HTML preview: {html_url}")
        if not args.no_png:
            print(f"PNG preview: {base_url}/{quote(png_path.name)}")
        print(f"Preview server PID: {server_process.pid}")
        webbrowser.open(html_url, new=2)


if __name__ == "__main__":
    main()
