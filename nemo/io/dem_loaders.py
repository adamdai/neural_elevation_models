from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _as_grid(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError("Expected DEM grid data with shape (H, W, C) where C >= 3.")

    x_candidate = array[..., 0]
    y_candidate = array[..., 1]
    z = array[..., 2]

    # Some stored DEM grids use (y, x, z, ...) channel order. Detect and normalize
    # to the library's expected (x, y, z).
    x_row_variation = float(np.nanmax(np.abs(np.diff(x_candidate, axis=1)))) if x_candidate.shape[1] > 1 else 0.0
    x_col_variation = float(np.nanmax(np.abs(np.diff(x_candidate, axis=0)))) if x_candidate.shape[0] > 1 else 0.0
    y_row_variation = float(np.nanmax(np.abs(np.diff(y_candidate, axis=1)))) if y_candidate.shape[1] > 1 else 0.0
    y_col_variation = float(np.nanmax(np.abs(np.diff(y_candidate, axis=0)))) if y_candidate.shape[0] > 1 else 0.0

    if x_row_variation < x_col_variation and y_col_variation < y_row_variation:
        x, y = y_candidate, x_candidate
    else:
        x, y = x_candidate, y_candidate

    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _crop_grid(grid: np.ndarray, xlims: tuple[float, float] | None, ylims: tuple[float, float] | None) -> np.ndarray:
    cropped = _as_grid(grid)
    if xlims is None and ylims is None:
        return cropped

    x_axis = cropped[0, :, 0]
    y_axis = cropped[:, 0, 1]

    x_mask = np.ones_like(x_axis, dtype=bool)
    y_mask = np.ones_like(y_axis, dtype=bool)

    if xlims is not None:
        x_min, x_max = sorted((float(xlims[0]), float(xlims[1])))
        x_mask = (x_axis >= x_min) & (x_axis <= x_max)
    if ylims is not None:
        y_min, y_max = sorted((float(ylims[0]), float(ylims[1])))
        y_mask = (y_axis >= y_min) & (y_axis <= y_max)

    if not np.any(x_mask) or not np.any(y_mask):
        raise ValueError("Crop bounds do not overlap the DEM.")
    return cropped[np.ix_(y_mask, x_mask)]


def _load_npy(
    path: Path,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
) -> np.ndarray:
    return _crop_grid(np.load(path), xlims, ylims)


def _grid_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x.astype(np.float32), y.astype(np.float32), indexing="xy")
    return np.stack([xx, yy, z.astype(np.float32)], axis=-1)


def _load_ascii_grid(
    path: Path,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        header = {}
        for _ in range(6):
            key, value = handle.readline().split()
            header[key.lower()] = float(value)
        z = np.loadtxt(handle, dtype=np.float32)

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    if z.shape != (nrows, ncols):
        raise ValueError(f"Unexpected ASCII grid shape {z.shape}, expected {(nrows, ncols)}.")

    xll = header.get("xllcorner", header.get("xllcenter"))
    yll = header.get("yllcorner", header.get("yllcenter"))
    cellsize = header["cellsize"]
    nodata = header.get("nodata_value")

    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)

    x = xll + np.arange(ncols, dtype=np.float32) * cellsize
    y = yll + np.arange(nrows, dtype=np.float32) * cellsize
    return _crop_grid(_grid_from_axes(x, y, z), xlims, ylims)


def _load_dat_grid(
    path: Path,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
) -> np.ndarray:
    try:
        return _crop_grid(np.load(path, allow_pickle=True), xlims, ylims)
    except Exception:
        pass

    try:
        with path.open("rb") as handle:
            return _crop_grid(pickle.load(handle), xlims, ylims)
    except Exception:
        pass

    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    if data.ndim != 2 or data.size == 0:
        raise ValueError(f"Unsupported .dat contents in {path}.")

    if data.shape[1] == 3:
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        x_axis = np.unique(x)
        y_axis = np.unique(y)
        if x_axis.size * y_axis.size != len(data):
            raise ValueError(
                f".dat xyz points in {path} do not form a complete rectilinear grid."
            )

        x_index = {float(value): idx for idx, value in enumerate(x_axis)}
        y_index = {float(value): idx for idx, value in enumerate(y_axis)}
        z_grid = np.full((y_axis.size, x_axis.size), np.nan, dtype=np.float32)

        for x_value, y_value, z_value in data:
            row = y_index[float(y_value)]
            col = x_index[float(x_value)]
            if not np.isnan(z_grid[row, col]):
                raise ValueError(f".dat xyz points in {path} contain duplicate coordinates.")
            z_grid[row, col] = z_value

        return _crop_grid(_grid_from_axes(x_axis, y_axis, z_grid), xlims, ylims)

    if data.shape[1] >= 1:
        x_axis = np.arange(data.shape[1], dtype=np.float32)
        y_axis = np.arange(data.shape[0], dtype=np.float32)
        return _crop_grid(_grid_from_axes(x_axis, y_axis, data), xlims, ylims)

    raise ValueError(f"Unsupported .dat contents in {path}.")


def _load_rasterio_grid(
    path: Path,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    max_side: int | None = None,
) -> np.ndarray:
    try:
        import rasterio
        from rasterio import Affine
        from rasterio.windows import Window
        from rasterio.windows import transform as window_transform
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading GeoTIFF DEMs requires `rasterio`. Install it to use `.tif` or `.tiff` files."
        ) from exc

    with rasterio.open(path) as dataset:
        window = None
        if xlims is not None or ylims is not None:
            bounds = dataset.bounds
            x_min, x_max = sorted(xlims if xlims is not None else (bounds.left, bounds.right))
            y_min, y_max = sorted(ylims if ylims is not None else (bounds.bottom, bounds.top))
            left = max(float(x_min), float(bounds.left))
            right = min(float(x_max), float(bounds.right))
            bottom = max(float(y_min), float(bounds.bottom))
            top = min(float(y_max), float(bounds.top))
            if left >= right or bottom >= top:
                raise ValueError("Crop bounds do not overlap the DEM.")

            raw_window = dataset.window(left, bottom, right, top)
            col_off = max(0, int(np.floor(raw_window.col_off)))
            row_off = max(0, int(np.floor(raw_window.row_off)))
            col_end = min(dataset.width, int(np.ceil(raw_window.col_off + raw_window.width)))
            row_end = min(dataset.height, int(np.ceil(raw_window.row_off + raw_window.height)))
            width = col_end - col_off
            height = row_end - row_off
            if width <= 0 or height <= 0:
                raise ValueError("Crop bounds do not overlap the DEM.")
            window = Window(col_off, row_off, width, height)

        out_shape = None
        if max_side is not None:
            max_side = max(int(max_side), 1)
            if window is None:
                read_width = int(dataset.width)
                read_height = int(dataset.height)
            else:
                read_width = int(window.width)
                read_height = int(window.height)
            scale = max(read_width / max_side, read_height / max_side, 1.0)
            out_width = max(1, int(np.ceil(read_width / scale)))
            out_height = max(1, int(np.ceil(read_height / scale)))
            out_shape = (out_height, out_width)

        z = dataset.read(1, window=window, out_shape=out_shape).astype(np.float32)
        nodata = dataset.nodata
        if nodata is not None:
            z = np.where(z == nodata, np.nan, z)

        transform = window_transform(window, dataset.transform) if window is not None else dataset.transform
        if out_shape is not None:
            scale_x = (window.width if window is not None else dataset.width) / float(out_shape[1])
            scale_y = (window.height if window is not None else dataset.height) / float(out_shape[0])
            transform = transform * Affine.scale(scale_x, scale_y)
        rows, cols = np.indices(z.shape, dtype=np.float32)
        xs, ys = rasterio.transform.xy(
            transform,
            rows.reshape(-1),
            cols.reshape(-1),
            offset="center",
        )
        x = np.asarray(xs, dtype=np.float32).reshape(z.shape)
        y = np.asarray(ys, dtype=np.float32).reshape(z.shape)
    return np.stack([x, y, z], axis=-1)


def load_dem(
    path: str | Path,
    *,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    max_side: int | None = None,
) -> np.ndarray:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"DEM file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return _load_npy(file_path, xlims=xlims, ylims=ylims)
    if suffix == ".asc":
        return _load_ascii_grid(file_path, xlims=xlims, ylims=ylims)
    if suffix == ".dat":
        return _load_dat_grid(file_path, xlims=xlims, ylims=ylims)
    if suffix in {".tif", ".tiff"}:
        return _load_rasterio_grid(file_path, xlims=xlims, ylims=ylims, max_side=max_side)
    raise ValueError(f"Unsupported DEM format: {file_path.suffix}")
