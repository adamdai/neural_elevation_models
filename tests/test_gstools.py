#!/usr/bin/env python3
"""
GSTools kriging demo to mirror your GPyTorch test:

- load DEM (or generate synthetic)
- subsample training points
- estimate empirical variogram + fit model (Exponential by default)
- build Ordinary Kriging with the fitted model
- predict on full grid (tiled for memory)
- print RMSE/MAE/max error vs ground truth Z
- save model + predictions + variance + coords

Author: you + ChatGPT
"""

import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import gstools as gs


# ----------------------------
# 0) Helpers & synthetic DEM
# ----------------------------
def make_synthetic_dem(nx=512, ny=512, seed=0):
    """
    Make a smooth terrain-like function on a regular grid.
    Returns x(1D), y(1D), Z(2D) as float32.
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 10_000.0, nx, dtype=np.float32)  # meters
    y = np.linspace(0.0, 10_000.0, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")

    Z = (
        80 * np.exp(-((X - 2500) ** 2 + (Y - 3000) ** 2) / (2 * (1200**2)))
        + 40 * np.exp(-((X - 7000) ** 2 + (Y - 7000) ** 2) / (2 * (900**2)))
        + 0.002 * X
        + 0.003 * Y
    ).astype(np.float32)

    Z += rng.normal(0, 0.2, size=Z.shape).astype(np.float32)  # tiny noise
    return x, y, Z


def prepare_data():
    """
    Match your GPyTorch script:
    data/Site01_final_adj_5mpp_surf.tif → nemo.dem.DEM
    """
    from nemo.dem import DEM  # your module

    tif_path = Path("data/Site01_final_adj_5mpp_surf.tif")
    dem = DEM.from_file(tif_path)
    XYZ = dem.data  # HxWx3: (x,y,z)
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2]
    x = X[0, :].astype(np.float32)
    y = Y[:, 0].astype(np.float32)
    Z = Z.astype(np.float32)
    return x, y, Z


def subsample_training(x, y, Z, max_train=20000, seed=42):
    """
    Uniform random subsample from a regular grid to keep kriging tractable.
    Returns pos_sub (N,2), val_sub (N,)
    """
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    pos = np.column_stack([Xg.ravel(), Yg.ravel()]).astype(np.float32)
    val = Z.ravel().astype(np.float32)

    n = len(val)
    if n > max_train:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_train, replace=False)
        return pos[idx], val[idx]
    return pos, val


# ----------------------------
# 1) Variogram fit (GSTools)
# ----------------------------
def fit_variogram_model(pos_sub, val_sub, model_ctor=gs.Exponential, n_lags=40, max_dist=None):
    """
    Estimate empirical variogram from subset, then fit model parameters.
    Returns a configured GSTools covariance model.
    """
    # Empirical variogram (unstructured)
    # GSTools expects unstructured positions as a tuple (x, y) or an array with shape (dim, N)
    # Convert from (N, 2) to a tuple of arrays to avoid shape errors
    x_pos = pos_sub[:, 0].astype(np.float32)
    y_pos = pos_sub[:, 1].astype(np.float32)
    # Try without specifying problematic parameters, let GSTools use defaults
    try:
        bins, gamma = gs.vario_estimate_unstructured(
            (x_pos, y_pos), val_sub.astype(np.float32), max_dist=max_dist
        )
    except TypeError:
        # Fallback: try with minimal parameters
        bins, gamma = gs.vario_estimate_unstructured((x_pos, y_pos), val_sub.astype(np.float32))

    # Choose model and fit (Exponential, Gaussian, Matern, etc.)
    model = model_ctor(dim=2)
    # Provide gentle initial guesses for stability (optional)
    # variance ~ data variance; len_scale ~ 5% of domain diagonal
    var0 = float(np.var(val_sub))
    # derive domain extents from the subset
    x_min, y_min = np.min(pos_sub, axis=0)
    x_max, y_max = np.max(pos_sub, axis=0)
    len0 = 0.05 * float(np.hypot(x_max - x_min, y_max - y_min))
    model.var = max(var0, 1e-6)
    model.len_scale = max(len0, 1e-6)
    model.nugget = 0.01 * model.var  # small nugget

    # Fit modifies model in-place - use the correct GSTools API
    try:
        # Try to fit the variogram model with more robust parameters
        model.fit_variogram(bins, gamma, max_eval=1000)
    except (AttributeError, RuntimeError) as e:
        # Fallback: manually set parameters based on empirical variogram
        print(f"Warning: Could not fit variogram automatically ({e}), using empirical estimates")
        # Estimate parameters from the empirical variogram
        # Find the range where gamma stabilizes (approximate sill)
        stable_idx = np.where(gamma > 0.8 * np.max(gamma))[0]
        if len(stable_idx) > 0:
            # Use the distance where variogram stabilizes as length scale
            length_scale = bins[stable_idx[0]] if len(stable_idx) > 0 else bins[-1] * 0.1
        else:
            length_scale = bins[-1] * 0.1

        # Set model parameters manually
        model.var = float(np.var(val_sub))
        model.len_scale = float(length_scale)
        if hasattr(model, "nugget"):
            model.nugget = 0.01 * model.var

    return model, (bins, gamma)


# ----------------------------
# 2) Build OK and predict
# ----------------------------
def build_ok_and_predict(model, pos_sub, val_sub, xq, yq, tile_rows=256):
    """
    Build an Ordinary Kriging object with the fitted model and subsampled
    conditioning points, then predict on the (xq, yq) grid in row tiles.
    Returns predicted mean Zk and kriging variance Vk.
    """
    # Ordinary Kriging object
    OK = gs.krige.Ordinary(model, cond_pos=pos_sub, cond_val=val_sub)

    Ny, Nx = len(yq), len(xq)
    Zk = np.empty((Ny, Nx), dtype=np.float32)
    Vk = np.empty_like(Zk)

    for y0 in range(0, Ny, tile_rows):
        y1 = min(y0 + tile_rows, Ny)
        OK.grid(xx=xq, yy=yq[y0:y1])  # computes on the subgrid
        Zk[y0:y1, :] = OK.field.reshape(y1 - y0, Nx).astype(np.float32)
        Vk[y0:y1, :] = OK.var.reshape(y1 - y0, Nx).astype(np.float32)

    return Zk, Vk, OK


# ----------------------------
# 3) End-to-end demo
# ----------------------------
if __name__ == "__main__":
    # --- Load data ---
    # Uncomment ONE of the following:
    # x, y, Z = make_synthetic_dem(nx=512, ny=512, seed=0)
    print("Loading DEM data...")
    x, y, Z = prepare_data()
    print(
        f"DEM loaded: shape={Z.shape}, x_range=[{x.min():.1f}, {x.max():.1f}], y_range=[{y.min():.1f}, {y.max():.1f}]"
    )

    # --- Subsample training points (analogous to your GP subsample) ---
    print("Subsampling training points...")
    pos_sub, val_sub = subsample_training(x, y, Z, max_train=20000, seed=42)
    print(f"Training points: {len(val_sub):,}")

    # --- Variogram fitting (Exponential; try gs.Matern for other smoothness) ---
    # You can tune bin_num or max_dist if needed (max_dist ~ 0.5 * domain diagonal is typical)
    print("Fitting variogram model...")
    model, (bins, gamma) = fit_variogram_model(
        pos_sub,
        val_sub,
        model_ctor=gs.Exponential,  # or gs.Gaussian, gs.Matern, etc.
        max_dist=None,
    )
    print(
        f"Variogram fitted: {model.name} model with length_scale={model.len_scale:.1f}, variance={model.var:.1f}"
    )

    # --- Predict over full grid in tiles (keeps RAM in check) ---
    print("Starting kriging prediction...")
    Z_pred, Z_var, OK = build_ok_and_predict(model, pos_sub, val_sub, xq=x, yq=y, tile_rows=256)
    print(f"Prediction completed: shape={Z_pred.shape}")

    # --- Metrics vs. ground truth ---
    rmse = float(np.sqrt(np.mean((Z_pred - Z) ** 2)))
    mae = float(np.mean(np.abs(Z_pred - Z)))
    max_err = float(np.max(np.abs(Z_pred - Z)))
    print(f"RMSE vs ground truth: {rmse:.3f} m")
    print(f"MAE vs ground truth:  {mae:.3f} m")
    print(f"Max error:            {max_err:.3f} m")

    # --- Save artifacts ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/gstools_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(results_dir / "prediction.npy", Z_pred)
    np.save(results_dir / "variance.npy", Z_var)
    np.save(results_dir / "ground_truth.npy", Z)
    np.save(results_dir / "x_coords.npy", x)
    np.save(results_dir / "y_coords.npy", y)

    # Save model parameters as JSON (human-readable)
    model_dict = {
        "name": model.name,
        "dim": model.dim,
        "var": float(model.var),
        "len_scale": float(model.len_scale),
        "nugget": float(getattr(model, "nugget", 0.0)),
        "anis": {
            "anis": getattr(model, "anis", None),
            "angles": getattr(model, "angles", None),
        },
    }
    with open(results_dir / "model.json", "w") as f:
        json.dump(model_dict, f, indent=2)

    # Save full GSTools objects via pickle (for exact reload)
    with open(results_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(results_dir / "ok_object.pkl", "wb") as f:
        pickle.dump(OK, f)

    # (Optional) Save variogram points used for fitting
    np.save(results_dir / "variogram_bins.npy", bins)
    np.save(results_dir / "variogram_gamma.npy", gamma)

    print(f"\nArtifacts saved to: {results_dir.resolve()}")
    print(f"Model params (JSON): {results_dir / 'model.json'}")
    print(f"Pickled model:       {results_dir / 'model.pkl'}")
    print(f"Pickled OK object:   {results_dir / 'ok_object.pkl'}")
