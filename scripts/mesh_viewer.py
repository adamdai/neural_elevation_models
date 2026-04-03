from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from nemo import Nemo
from nemo.viewer import default_camera, estimate_z_bounds, sample_height_field_mesh


@dataclass
class MeshViewerArgs:
    checkpoint_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    host: str = "127.0.0.1"
    port: int = 8081
    resolution_x: int = 280
    resolution_y: int = 280
    material: str = "standard"
    wireframe: bool = False
    flat_shading: bool = False
    show_world_axes: bool = True
    normalize_scene: bool = True


def main(args: MeshViewerArgs) -> None:
    try:
        import viser
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "viser is required for the mesh viewer. Install it with "
            "`conda run -n nemo pip install viser`."
        ) from exc

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    nemo = Nemo.load_checkpoint(checkpoint_path, map_location=args.device).to(args.device)
    mesh = sample_height_field_mesh(
        nemo,
        resolution_x=int(args.resolution_x),
        resolution_y=int(args.resolution_y),
    )
    mesh, eye, target = _prepare_viewer_mesh(mesh, normalize_scene=bool(args.normalize_scene))

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = bool(args.show_world_axes)

    with server.gui.add_folder("Mesh"):
        gui_wireframe = server.gui.add_checkbox("Wireframe", initial_value=bool(args.wireframe))
        gui_flat = server.gui.add_checkbox("Flat shading", initial_value=bool(args.flat_shading))
        gui_axes = server.gui.add_checkbox("Show world axes", initial_value=bool(args.show_world_axes))
    status = server.gui.add_text(
        "Status",
        initial_value=(
            f"Loaded {checkpoint_path.name} as a mesh with "
            f"{mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces."
        ),
        disabled=True,
    )

    def _refresh_mesh() -> None:
        server.scene.reset()
        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = bool(gui_axes.value)
        server.scene.add_mesh_trimesh("/terrain_mesh", mesh=mesh)
        if bool(gui_wireframe.value):
            server.scene.add_mesh_simple(
                "/terrain_mesh_overlay",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(255, 255, 255),
                wireframe=True,
                flat_shading=bool(gui_flat.value),
                opacity=1.0,
                material=str(args.material),
                side="double",
            )
        status.value = (
            f"Mesh viewer ready. vertices={mesh.vertices.shape[0]} faces={mesh.faces.shape[0]} "
            f"wireframe={bool(gui_wireframe.value)} flat={bool(gui_flat.value)} "
            f"normalized={bool(args.normalize_scene)}"
        )

    @gui_wireframe.on_update
    def _(_) -> None:
        _refresh_mesh()

    @gui_flat.on_update
    def _(_) -> None:
        _refresh_mesh()

    @gui_axes.on_update
    def _(_) -> None:
        _refresh_mesh()

    @server.on_client_connect
    def _(client: object) -> None:
        with client.atomic():
            client.camera.position = eye
            client.camera.look_at = target
            client.camera.up_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    _refresh_mesh()
    print(f"Mesh viewer running at http://{args.host}:{args.port}")
    while True:
        __import__("time").sleep(1.0)


def _prepare_viewer_mesh(mesh, *, normalize_scene: bool):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    span = np.maximum(bounds_max - bounds_min, 1e-6)

    if normalize_scene:
        scale = 2.0 / float(np.max(span))
        vertices = (vertices - center[None, :]) * scale
        mesh = mesh.copy()
        mesh.vertices = vertices
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        span = np.maximum(bounds_max - bounds_min, 1e-6)

    eye = center + np.array(
        [-0.9 * span[0], -0.9 * span[1], 0.9 * max(span[0], span[1]) + 1.5 * span[2]],
        dtype=np.float32,
    )
    target = center.astype(np.float32)
    return mesh, eye, target


if __name__ == "__main__":
    main(tyro.cli(MeshViewerArgs))
