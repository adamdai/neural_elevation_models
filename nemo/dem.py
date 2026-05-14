from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from nemo.io.dem_loaders import load_dem

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - optional dependency
    pv = None

ArrayLike = float | Sequence[float] | np.ndarray


@dataclass(frozen=True)
class DEMBounds:
    x: tuple[float, float]
    y: tuple[float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _as_grid(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected DEM data with shape (H, W, 3).")
    return array


class DEM:
    """Standardized DEM grid with interpolation, gradients, and cropping."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        data: np.ndarray | None = None,
        xlims: tuple[float, float] | None = None,
        ylims: tuple[float, float] | None = None,
        max_side: int | None = None,
        zero_origin: bool = False,
    ) -> None:
        if (path is None) == (data is None):
            raise ValueError("Provide exactly one of `path` or `data`.")

        if data is not None and (xlims is not None or ylims is not None):
            raise ValueError("Crop bounds are only supported when loading from `path`.")

        grid = load_dem(path, xlims=xlims, ylims=ylims, max_side=max_side) if path is not None else _as_grid(data)
        x = np.asarray(grid[..., 0], dtype=np.float32)
        y = np.asarray(grid[..., 1], dtype=np.float32)
        z = np.asarray(grid[..., 2], dtype=np.float32)

        if x.ndim != 2 or y.ndim != 2 or z.ndim != 2:
            raise ValueError("DEM coordinate grids must be 2D.")

        if zero_origin:
            x = x - x[0, 0]
            y = y - y[0, 0]

        self.x = x
        self.y = y
        self.z = z
        self.data = np.stack([self.x, self.y, self.z], axis=-1)

        self.x_axis = self.x[0, :].copy()
        self.y_axis = self.y[:, 0].copy()
        self.bounds = DEMBounds(
            x=(float(np.nanmin(self.x_axis)), float(np.nanmax(self.x_axis))),
            y=(float(np.nanmin(self.y_axis)), float(np.nanmax(self.y_axis))),
        )

        self.dx = self._estimate_spacing(self.x_axis)
        self.dy = self._estimate_spacing(self.y_axis)
        self.gy, self.gx = np.gradient(self.z, self.y_axis, self.x_axis, edge_order=1)
        self._render_plotter = None
        self._render_grid = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        xlims: tuple[float, float] | None = None,
        ylims: tuple[float, float] | None = None,
        max_side: int | None = None,
        zero_origin: bool = False,
    ) -> "DEM":
        return cls(path=path, xlims=xlims, ylims=ylims, max_side=max_side, zero_origin=zero_origin)

    @classmethod
    def from_grid(cls, data: np.ndarray, *, zero_origin: bool = False) -> "DEM":
        return cls(data=data, zero_origin=zero_origin)

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def resolution(self) -> tuple[float, float]:
        return self.dx, self.dy

    def xy_in_bounds(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        return (
            (x_arr >= self.bounds.x[0])
            & (x_arr <= self.bounds.x[1])
            & (y_arr >= self.bounds.y[0])
            & (y_arr <= self.bounds.y[1])
        )

    def rc_in_bounds(self, r: int, c: int) -> bool:
        h, w = self.shape
        return 0 <= r < h and 0 <= c < w

    def rc_to_xy(self, r: int, c: int) -> tuple[float, float]:
        if not self.rc_in_bounds(r, c):
            raise ValueError(f"Indices {(r, c)} are out of bounds for shape {self.shape}.")
        return float(self.x[r, c]), float(self.y[r, c])

    def xy_to_rc(self, x: float, y: float) -> tuple[int, int]:
        if not bool(self.xy_in_bounds(x, y)):
            raise ValueError(f"Coordinates {(x, y)} are out of bounds.")
        c = int(np.argmin(np.abs(self.x_axis - x)))
        r = int(np.argmin(np.abs(self.y_axis - y)))
        return r, c

    def query(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return self._interp(self.z, x, y)

    def grad(self, x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        gx = self._interp(self.gx, x, y)
        gy = self._interp(self.gy, x, y)
        return gx, gy

    def crop(self, xlims: tuple[float, float], ylims: tuple[float, float]) -> "DEM":
        x_mask = (self.x_axis >= xlims[0]) & (self.x_axis <= xlims[1])
        y_mask = (self.y_axis >= ylims[0]) & (self.y_axis <= ylims[1])
        if not np.any(x_mask) or not np.any(y_mask):
            raise ValueError("Crop bounds do not overlap the DEM.")
        cropped = self.data[np.ix_(y_mask, x_mask)]
        return DEM.from_grid(cropped)

    def to_xyz(self) -> np.ndarray:
        return self.data.reshape(-1, 3)

    def setup_render(self, width: int, height: int) -> None:
        if pv is None:  # pragma: no cover - optional dependency
            raise ImportError("pyvista is required for DEM rendering.")
        self._render_grid = pv.StructuredGrid(self.x, self.y, self.z)
        self._render_plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        self._render_plotter.set_background("white")
        self._render_plotter.add_mesh(self._render_grid, cmap="terrain", smooth_shading=False)
        self._render_plotter.show(auto_close=False, interactive=False, screenshot=False)

    def render_view(
        self,
        world_T_camera: np.ndarray,
        intrinsics: CameraIntrinsics,
        *,
        return_rgb: bool = False,
        return_horizon: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if pv is None:  # pragma: no cover - optional dependency
            raise ImportError("pyvista is required for DEM rendering.")
        if self._render_plotter is None:
            self.setup_render(intrinsics.width, intrinsics.height)

        Rwc = np.asarray(world_T_camera[:3, :3], dtype=np.float64)
        twc = np.asarray(world_T_camera[:3, 3], dtype=np.float64)
        forward_w = Rwc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up_w = Rwc @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        fovy_deg = np.degrees(2.0 * np.arctan2(intrinsics.height / 2.0, intrinsics.fy))

        camera = pv.Camera()
        camera.position = tuple(twc)
        camera.focal_point = tuple(twc + forward_w)
        camera.up = tuple(up_w / max(np.linalg.norm(up_w), 1e-8))
        camera.view_angle = float(fovy_deg)
        camera.SetUseHorizontalViewAngle(False)
        camera.OrthogonalizeViewUp()

        self._render_plotter.camera = camera
        self._render_plotter.renderer.ResetCameraClippingRange()
        self._render_plotter.render()
        depth = self._render_plotter.get_image_depth()

        if return_horizon:
            mask = np.isfinite(depth) & (depth > -2000.0)
            horizon = np.argmax(mask, axis=0)
            horizon[~mask.any(axis=0)] = -1
            return depth, horizon

        if return_rgb:
            rgb = self._render_plotter.screenshot(None, return_img=True)
            return depth, rgb

        return depth

    @staticmethod
    def _estimate_spacing(axis: np.ndarray) -> float:
        if axis.size < 2:
            return 0.0
        diffs = np.diff(axis)
        return float(np.mean(np.abs(diffs)))

    def _interp(self, grid: np.ndarray, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same shape.")

        out = np.full(x_arr.shape, np.nan, dtype=np.float32)
        in_bounds = self.xy_in_bounds(x_arr, y_arr)
        if not np.any(in_bounds):
            return float(out) if out.ndim == 0 else out

        xq = x_arr[in_bounds]
        yq = y_arr[in_bounds]

        x_hi = np.searchsorted(self.x_axis, xq, side="right")
        y_hi = np.searchsorted(self.y_axis, yq, side="right")
        x_hi = np.clip(x_hi, 1, len(self.x_axis) - 1)
        y_hi = np.clip(y_hi, 1, len(self.y_axis) - 1)
        x_lo = x_hi - 1
        y_lo = y_hi - 1

        x0 = self.x_axis[x_lo]
        x1 = self.x_axis[x_hi]
        y0 = self.y_axis[y_lo]
        y1 = self.y_axis[y_hi]

        tx = np.where(np.abs(x1 - x0) > 0, (xq - x0) / (x1 - x0), 0.0)
        ty = np.where(np.abs(y1 - y0) > 0, (yq - y0) / (y1 - y0), 0.0)

        z00 = grid[y_lo, x_lo]
        z10 = grid[y_lo, x_hi]
        z01 = grid[y_hi, x_lo]
        z11 = grid[y_hi, x_hi]

        interp = (
            (1.0 - tx) * (1.0 - ty) * z00
            + tx * (1.0 - ty) * z10
            + (1.0 - tx) * ty * z01
            + tx * ty * z11
        )
        out[in_bounds] = interp
        return float(out) if out.ndim == 0 else out
