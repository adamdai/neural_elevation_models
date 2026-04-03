from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np
import torch
import tyro

from nemo import Nemo
from nemo.viewer import (
    CameraViewState,
    default_camera,
    estimate_z_bounds,
    preview_point_cloud,
    render_rgb,
)


@dataclass
class InteractiveRenderArgs:
    checkpoint_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    host: str = "127.0.0.1"
    port: int = 8080
    render_scale: float = 0.5
    max_fps: float = 8.0
    jpeg_quality: int = 85
    preview_resolution_x: int = 180
    preview_resolution_y: int = 180
    point_size: float = 0.015
    t_near: float = 1.0
    t_far: float | None = None
    num_bracket_samples: int = 32
    num_bisection_steps: int = 6
    num_newton_steps: int = 1
    ray_batch_size: int = 32768


@dataclass
class _RenderState:
    latest_view: CameraViewState | None = None
    pending: bool = False
    running: bool = False
    last_render_end: float = 0.0
    frames_rendered: int = 0


def _camera_view_from_client(client: object) -> CameraViewState:
    camera = client.camera
    image_width = int(getattr(camera, "image_width", 1280) or 1280)
    image_height = int(getattr(camera, "image_height", 720) or 720)
    return CameraViewState(
        position=np.asarray(camera.position, dtype=np.float32),
        look_at=np.asarray(camera.look_at, dtype=np.float32),
        up_direction=np.asarray(camera.up_direction, dtype=np.float32),
        fov=float(camera.fov),
        image_width=image_width,
        image_height=image_height,
    )


def main(args: InteractiveRenderArgs) -> None:
    try:
        import viser
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "viser is required for the interactive renderer. Install it with "
            "`conda run -n nemo pip install viser`."
        ) from exc

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    nemo = Nemo.load_checkpoint(checkpoint_path, map_location=args.device).to(args.device)
    z_bounds = estimate_z_bounds(nemo)
    points, colors = preview_point_cloud(
        nemo,
        resolution_x=args.preview_resolution_x,
        resolution_y=args.preview_resolution_y,
    )
    eye, target = default_camera(nemo.field.bounds, z_bounds)

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.world_axes.visible = True
    cloud_handle = server.scene.add_point_cloud(
        "/terrain_preview",
        points=points,
        colors=colors,
        point_size=float(args.point_size),
        point_shape="circle",
    )

    with server.gui.add_folder("Render"):
        gui_scale = server.gui.add_slider(
            "Resolution scale",
            min=0.1,
            max=1.0,
            step=0.05,
            initial_value=float(np.clip(args.render_scale, 0.1, 1.0)),
        )
        gui_fps = server.gui.add_slider(
            "Max FPS",
            min=1.0,
            max=20.0,
            step=0.5,
            initial_value=float(np.clip(args.max_fps, 1.0, 20.0)),
        )
        gui_quality = server.gui.add_slider(
            "JPEG quality",
            min=40,
            max=100,
            step=1,
            initial_value=int(np.clip(args.jpeg_quality, 40, 100)),
        )
        gui_auto = server.gui.add_checkbox("Auto render", initial_value=True)
        gui_render = server.gui.add_button("Render now")

    with server.gui.add_folder("Raymarch"):
        gui_tnear = server.gui.add_number("t_near", initial_value=float(args.t_near), step=0.5)
        gui_tfar = server.gui.add_number(
            "t_far",
            initial_value=float(args.t_far if args.t_far is not None else 0.0),
            step=10.0,
        )
        gui_bracket = server.gui.add_slider(
            "Bracket samples",
            min=16,
            max=192,
            step=8,
            initial_value=int(args.num_bracket_samples),
        )
        gui_bisection = server.gui.add_slider(
            "Bisection steps",
            min=4,
            max=24,
            step=1,
            initial_value=int(args.num_bisection_steps),
        )
        gui_newton = server.gui.add_slider(
            "Newton steps",
            min=0,
            max=6,
            step=1,
            initial_value=int(args.num_newton_steps),
        )

    with server.gui.add_folder("Scene"):
        gui_preview = server.gui.add_checkbox("Show preview cloud", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.05,
            step=0.001,
            initial_value=float(args.point_size),
        )

    status = server.gui.add_text("Status", initial_value="Waiting for a client connection.", disabled=True)

    @gui_preview.on_update
    def _(_) -> None:
        cloud_handle.visible = bool(gui_preview.value)

    @gui_point_size.on_update
    def _(_) -> None:
        cloud_handle.point_size = float(gui_point_size.value)

    render_states: dict[int, _RenderState] = {}
    state_lock = threading.Lock()

    def schedule_render(client: object) -> None:
        client_id = int(client.client_id)
        with state_lock:
            state = render_states.setdefault(client_id, _RenderState())
            state.latest_view = _camera_view_from_client(client)
            state.pending = True
            if state.running:
                return
            state.running = True
        thread = threading.Thread(target=render_loop, args=(client,), daemon=True)
        thread.start()

    def render_loop(client: object) -> None:
        client_id = int(client.client_id)
        while True:
            with state_lock:
                state = render_states[client_id]
                view = state.latest_view
                state.pending = False
                last_render_end = state.last_render_end
            if view is None:
                with state_lock:
                    render_states[client_id].running = False
                return

            min_interval = 1.0 / max(float(gui_fps.value), 1e-3)
            wait_seconds = last_render_end + min_interval - time.perf_counter()
            if wait_seconds > 0.0:
                time.sleep(wait_seconds)

            status.value = f"Rendering {view.image_width}x{view.image_height} view for client {client_id}."
            start = time.perf_counter()
            rgb, render = render_rgb(
                nemo,
                view,
                resolution_scale=float(gui_scale.value),
                t_near=float(gui_tnear.value),
                t_far=None if float(gui_tfar.value) <= 0.0 else float(gui_tfar.value),
                num_bracket_samples=int(gui_bracket.value),
                num_bisection_steps=int(gui_bisection.value),
                num_newton_steps=int(gui_newton.value),
                ray_batch_size=int(args.ray_batch_size),
            )
            elapsed = time.perf_counter() - start
            with client.atomic():
                client.scene.set_background_image(
                    rgb,
                    format="jpeg",
                    jpeg_quality=int(gui_quality.value),
                )
            finite = int(np.isfinite(render.depth).sum())
            with state_lock:
                state = render_states[client_id]
                state.last_render_end = time.perf_counter()
                state.frames_rendered += 1
                should_continue = state.pending and bool(gui_auto.value)
                if not should_continue:
                    state.running = False
            status.value = (
                f"Last render: {rgb.shape[1]}x{rgb.shape[0]}, "
                f"{elapsed:.2f}s, finite pixels={finite}, frames={render_states[client_id].frames_rendered}."
            )
            if not should_continue:
                return

    @gui_render.on_click
    def _(event: object) -> None:
        schedule_render(event.client)

    @server.on_client_connect
    def _(client: object) -> None:
        with client.atomic():
            client.camera.position = eye
            client.camera.look_at = target
            client.camera.up_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        render_states[int(client.client_id)] = _RenderState()
        status.value = f"Client {client.client_id} connected. Move the camera to render."

        @client.camera.on_update
        def _(_: object) -> None:
            if bool(gui_auto.value):
                schedule_render(client)

    @server.on_client_disconnect
    def _(client: object) -> None:
        with state_lock:
            render_states.pop(int(client.client_id), None)
        status.value = f"Client {client.client_id} disconnected."

    print(f"Interactive viewer running at http://{args.host}:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(InteractiveRenderArgs))
