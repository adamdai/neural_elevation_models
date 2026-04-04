from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from nemo.dem import CameraIntrinsics


@dataclass(frozen=True)
class NerfstudioFrame:
    image_path: Path
    world_T_camera: np.ndarray
    intrinsics: CameraIntrinsics
    image: torch.Tensor


@dataclass(frozen=True)
class NerfstudioDataset:
    root: Path
    frames: list[NerfstudioFrame]
    sparse_points: np.ndarray
    bounds_xy: tuple[tuple[float, float], tuple[float, float]]
    z_range: tuple[float, float]
    aligned_T_world: np.ndarray


def load_nerfstudio_dataset(
    root: str | Path,
    *,
    image_folder: str | None = "images_4",
    max_frames: int | None = None,
    align_to_height_field: bool = True,
) -> NerfstudioDataset:
    root_path = Path(root).expanduser().resolve()
    transforms = json.loads((root_path / "transforms.json").read_text())
    sparse_points_world = _load_sparse_points(root_path / transforms["ply_file_path"])
    aligned_T_world = (
        _estimate_height_field_alignment(sparse_points_world, transforms)
        if align_to_height_field
        else np.eye(4, dtype=np.float32)
    )

    frames: list[NerfstudioFrame] = []
    for frame_spec in transforms["frames"][: max_frames if max_frames is not None else None]:
        image_path = _resolve_image_path(root_path, frame_spec["file_path"], image_folder=image_folder)
        image = _load_image_tensor(image_path)
        scale_x = image.shape[1] / float(transforms["w"])
        scale_y = image.shape[0] / float(transforms["h"])
        intrinsics = CameraIntrinsics(
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            fx=float(transforms["fl_x"]) * scale_x,
            fy=float(transforms["fl_y"]) * scale_y,
            cx=float(transforms["cx"]) * scale_x,
            cy=float(transforms["cy"]) * scale_y,
        )
        frames.append(
            NerfstudioFrame(
                image_path=image_path,
                world_T_camera=aligned_T_world @ _nerfstudio_pose_to_nemo(frame_spec["transform_matrix"]),
                intrinsics=intrinsics,
                image=image,
            )
        )

    sparse_points = _transform_points(sparse_points_world, aligned_T_world)
    x = sparse_points[:, 0]
    y = sparse_points[:, 1]
    z = sparse_points[:, 2]
    bounds_xy = (
        (float(np.quantile(x, 0.01)), float(np.quantile(x, 0.99))),
        (float(np.quantile(y, 0.01)), float(np.quantile(y, 0.99))),
    )
    z_range = (float(np.quantile(z, 0.01)), float(np.quantile(z, 0.99)))
    return NerfstudioDataset(
        root=root_path,
        frames=frames,
        sparse_points=sparse_points,
        bounds_xy=bounds_xy,
        z_range=z_range,
        aligned_T_world=aligned_T_world,
    )


def _resolve_image_path(root: Path, relative_path: str, *, image_folder: str | None) -> Path:
    original = root / relative_path
    if image_folder is None:
        if not original.exists():
            raise FileNotFoundError(f"Image not found: {original}")
        return original
    candidate = root / image_folder / Path(relative_path).name
    if candidate.exists():
        return candidate
    if original.exists():
        return original
    raise FileNotFoundError(f"Image not found: {candidate} or {original}")


def _load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _nerfstudio_pose_to_nemo(transform_matrix: list[list[float]]) -> np.ndarray:
    c2w = np.asarray(transform_matrix, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = c2w[:3, 0]
    pose[:3, 1] = -c2w[:3, 1]
    pose[:3, 2] = -c2w[:3, 2]
    pose[:3, 3] = c2w[:3, 3]
    return pose


def _load_sparse_points(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    vertices = ply["vertex"]
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=-1)
    return (transform @ pts_h.T).T[:, :3].astype(np.float32)


def _estimate_height_field_alignment(points_world: np.ndarray, transforms: dict) -> np.ndarray:
    center = points_world.mean(axis=0)
    centered = points_world - center[None, :]
    cov = centered.T @ centered / max(points_world.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    x_axis = eigvecs[:, 0]
    z_axis = eigvecs[:, 2]
    camera_positions = np.stack(
        [np.asarray(frame["transform_matrix"], dtype=np.float32)[:3, 3] for frame in transforms["frames"]],
        axis=0,
    )
    mean_camera_offset = camera_positions.mean(axis=0) - center
    if float(np.dot(mean_camera_offset, z_axis)) < 0.0:
        z_axis = -z_axis
    x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-8)

    rotation = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(rotation @ center.astype(np.float32))
    return transform
