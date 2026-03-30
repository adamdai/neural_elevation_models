from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from nemo import DEM


def make_dem_grid(nx: int = 5, ny: int = 4) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    z = 2.0 * xx - 0.5 * yy
    return np.stack([xx, yy, z], axis=-1)


def test_dem_query_matches_plane() -> None:
    dem = DEM.from_grid(make_dem_grid())
    z = dem.query(0.25, -0.5)
    assert np.isclose(z, 2.0 * 0.25 - 0.5 * (-0.5), atol=1e-5)


def test_dem_gradient_matches_plane() -> None:
    dem = DEM.from_grid(make_dem_grid())
    gx, gy = dem.grad(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
    assert np.allclose(gx, np.array([2.0], dtype=np.float32), atol=1e-5)
    assert np.allclose(gy, np.array([-0.5], dtype=np.float32), atol=1e-5)


def test_dem_crop_and_index_conversion() -> None:
    dem = DEM.from_grid(make_dem_grid())
    cropped = dem.crop((-0.5, 1.0), (-2.0, 0.0))
    assert cropped.shape == (2, 4)
    r, c = dem.xy_to_rc(0.0, -2.0)
    x, y = dem.rc_to_xy(r, c)
    assert np.isclose(x, 0.0)
    assert np.isclose(y, -2.0)


def test_dem_from_dat_xyz_triplets(tmp_path: Path) -> None:
    path = tmp_path / "surface.dat"
    path.write_text(
        "\n".join(
            [
                "0.0 10.0 1.0",
                "1.0 10.0 2.0",
                "0.0 20.0 3.0",
                "1.0 20.0 4.0",
            ]
        ),
        encoding="utf-8",
    )

    dem = DEM.from_path(path)

    assert dem.shape == (2, 2)
    assert np.allclose(dem.x_axis, np.array([0.0, 1.0], dtype=np.float32))
    assert np.allclose(dem.y_axis, np.array([10.0, 20.0], dtype=np.float32))
    assert np.allclose(dem.z, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_dem_from_dat_z_matrix(tmp_path: Path) -> None:
    path = tmp_path / "surface.dat"
    path.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n", encoding="utf-8")

    dem = DEM.from_path(path)

    assert dem.shape == (2, 3)
    assert np.allclose(dem.x_axis, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(dem.y_axis, np.array([0.0, 1.0], dtype=np.float32))
    assert np.allclose(dem.z, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))


def test_dem_from_pickled_dat_grid_with_extra_channel(tmp_path: Path) -> None:
    x = np.array([[10.0, 11.0], [10.0, 11.0]], dtype=np.float64)
    y = np.array([[20.0, 20.0], [21.0, 21.0]], dtype=np.float64)
    z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    extra = np.zeros_like(z)
    stored = np.stack([y, x, z, extra], axis=-1)

    path = tmp_path / "surface.dat"
    with path.open("wb") as handle:
        pickle.dump(stored, handle)

    dem = DEM.from_path(path)

    assert dem.shape == (2, 2)
    assert np.allclose(dem.x_axis, np.array([10.0, 11.0], dtype=np.float32))
    assert np.allclose(dem.y_axis, np.array([20.0, 21.0], dtype=np.float32))
    assert np.allclose(dem.z, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
