import numpy as np
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neighbors import KDTree


class DEMBicubic:
    """
    Fit z=f(x,y) from a dense regular DEM grid using a bicubic spline.
    Optionally denoise with a light Gaussian filter first.
    """

    def __init__(self, x, y, Z, sigma=None, smoothing=None):
        """
        x: 1D array of x-coordinates (monotonic)
        y: 1D array of y-coordinates (monotonic)
        Z: 2D array, shape (len(y), len(x)) with Z[j,i] = z(y[j], x[i])
        sigma: optional Gaussian blur (in pixels) for denoising (e.g., 0.5–1.5)
        smoothing: RectBivariateSpline 's' smoothing factor (None => exact)
        """
        if sigma is not None and sigma > 0:
            Z = gaussian_filter(Z, sigma=sigma, mode="nearest")
        # kx=ky=3 => bicubic; s controls smoothing (regularization)
        self.spline = RectBivariateSpline(y, x, Z, kx=3, ky=3, s=(smoothing or 0.0))

    def __call__(self, xq, yq):
        """
        Evaluate z at query coordinates.
        xq, yq: scalars or arrays. Returns array broadcast to shape of np.broadcast(xq,yq)
        """
        xq = np.asarray(xq)
        yq = np.asarray(yq)
        # RectBivariateSpline expects 1D vectors; it returns a matrix for outer product.
        x1 = xq.ravel()
        y1 = yq.ravel()
        Zq = self.spline(y1, x1, grid=False)  # vectorized eval
        return Zq.reshape(np.broadcast(xq, yq).shape)


# --- Example ---
# x, y are 1D axes (meters); Z is 2D with shape (len(y), len(x))
# fitter = DEMBicubic(x, y, Z, sigma=1.0, smoothing=None)
# zq = fitter(xq, yq)  # arrays/scalars of query points


def fit_gpr_from_dem(
    x, y, Z, max_train=20000, nu=1.5, length_scale=None, noise_level=None, random_state=0
):
    """
    Fit a GP (kriging) model on a reduced training set from a regular DEM grid.

    x: (Nx,) x-coordinates (monotonic)
    y: (Ny,) y-coordinates (monotonic)
    Z: (Ny, Nx) elevations
    max_train: cap on number of training samples for tractable O(n^3) training
    nu: Matérn smoothness (1.5 or 2.5 are common in geostatistics)
    length_scale: initial kernel length scale (float or None for auto)
    noise_level: initial white noise level (float or None)
    """
    Ny, Nx = Z.shape
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])
    z = Z.ravel()

    # Subsample training points (uniform random or strided) to keep training tractable
    n_all = XY.shape[0]
    rng = np.random.RandomState(random_state)
    if n_all > max_train:
        idx = rng.choice(n_all, size=max_train, replace=False)
    else:
        idx = np.arange(n_all)

    X_train = XY[idx]
    y_train = z[idx]

    # Kernel: C * Matern + White
    if length_scale is None:
        # heuristic: ~ 5% of domain diagonal
        dx = x.max() - x.min()
        dy = y.max() - y.min()
        length_scale = 0.05 * np.hypot(dx, dy)
    if noise_level is None:
        # small nugget; increase if DEM is noisy
        noise_level = 1e-2 * np.std(y_train) if np.std(y_train) > 0 else 1e-3

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=length_scale, nu=nu
    ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-8, 1e1))

    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=random_state
    )
    gpr.fit(X_train, y_train)
    return gpr


def predict_grid_chunked(gpr, x, y, chunk_rows=256, return_std=False):
    """
    Predict GP over the whole grid in row-chunks to bound memory.
    Returns Z_pred (and optionally Z_std) with shape (Ny, Nx).
    """
    Ny, Nx = len(y), len(x)
    Z_pred = np.empty((Ny, Nx), dtype=float)
    Z_std = np.empty((Ny, Nx), dtype=float) if return_std else None

    for y0 in range(0, Ny, chunk_rows):
        y1 = min(y0 + chunk_rows, Ny)
        Xg, Yg = np.meshgrid(x, y[y0:y1], indexing="xy")
        XY = np.column_stack([Xg.ravel(), Yg.ravel()])
        if return_std:
            z_mean, z_std = gpr.predict(XY, return_std=True)
            Z_pred[y0:y1, :] = z_mean.reshape(y1 - y0, Nx)
            Z_std[y0:y1, :] = z_std.reshape(y1 - y0, Nx)
        else:
            z_mean = gpr.predict(XY, return_std=False)
            Z_pred[y0:y1, :] = z_mean.reshape(y1 - y0, Nx)
    return (Z_pred, Z_std) if return_std else Z_pred


# ---------- Usage ----------
# x, y: 1D arrays; Z: 2D DEM (Ny,Nx)
# gpr = fit_gpr_from_dem(x, y, Z, max_train=20000, nu=1.5)
# Z_pred, Z_std = predict_grid_chunked(gpr, x, y, chunk_rows=256, return_std=True)


def predict_local_kriging(
    x, y, Z, xq, yq, tile=128, k_neighbors=500, nu=1.5, length_scale=None, noise_level=None
):
    """
    Local kriging: split the query grid into tiles and, for each tile,
    fit a small GP on the k nearest training points around that tile.
    This avoids one huge O(n^3) fit.

    x,y: 1D axes; Z: (Ny,Nx)
    xq,yq: 1D query axes for output grid
    tile: tile size in rows for query grid
    k_neighbors: number of nearest training points per tile (balance quality/speed)
    """
    Ny, Nx = Z.shape
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    XY_train = np.column_stack([Xg.ravel(), Yg.ravel()])
    z_train = Z.ravel()

    tree = KDTree(XY_train)

    Nyq, Nxq = len(yq), len(xq)
    Z_out = np.empty((Nyq, Nxq), dtype=float)

    for y0 in tqdm(range(0, Nyq, tile)):
        y1 = min(y0 + tile, Nyq)
        Xtile, Ytile = np.meshgrid(xq, yq[y0:y1], indexing="xy")
        XY_tile = np.column_stack([Xtile.ravel(), Ytile.ravel()])

        # Find neighbors around the tile centroid (faster) OR per query (slower but better)
        centroid = XY_tile.mean(axis=0).reshape(1, -2)
        dists, idx = tree.query(centroid, k=min(k_neighbors, XY_train.shape[0]))
        idx = idx.ravel()

        Xn = XY_train[idx]
        zn = z_train[idx]

        # Fit a small GP on local neighbors
        dx = x.max() - x.min()
        dy = y.max() - y.min()
        if length_scale is None:
            length_scale_loc = 0.02 * np.hypot(dx, dy)
        else:
            length_scale_loc = length_scale
        if noise_level is None:
            noise_level_loc = 1e-3 * np.std(zn) if np.std(zn) > 0 else 1e-4
        else:
            noise_level_loc = noise_level

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scale_loc, nu=nu
        ) + WhiteKernel(noise_level=noise_level_loc, noise_level_bounds=(1e-8, 1e1))
        gpr_local = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=1
        )
        gpr_local.fit(Xn, zn)

        z_pred_tile = gpr_local.predict(XY_tile, return_std=False)
        Z_out[y0:y1, :] = z_pred_tile.reshape(y1 - y0, Nxq)

    return Z_out


# ---------- Usage ----------
# xq, yq could be same as x, y (reconstruction) or denser (upsampling)
# Z_local = predict_local_kriging(x, y, Z, xq=x, yq=y, tile=128, k_neighbors=1000)
