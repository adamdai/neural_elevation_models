import numpy as np
from pathlib import Path

from nemo.dem import DEM
from nemo.fit import DEMBicubic, fit_gpr_from_dem, predict_grid_chunked, predict_local_kriging


def prepare_data():
    tif_path = Path("data/Site01_final_adj_5mpp_surf.tif")
    dem = DEM.from_file(tif_path)
    XYZ = dem.data
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2]
    return X, Y, Z


def test_spline_fit():
    X, Y, Z = prepare_data()
    x = X[0, :]
    y = Y[:, 0]
    fitter = DEMBicubic(x, y, Z)
    print("Fit cubic spline to DEM")

    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_pred_flat = fitter(X_flat, Y_flat)
    height_error = np.abs(Z_pred_flat - Z.flatten())
    print(f"Height MAE: {height_error.mean():.5f} m")


def test_gpr_fit():
    X, Y, Z = prepare_data()
    x = X[0, :]
    y = Y[:, 0]
    print("Fitting GPR to DEM...")
    gpr = fit_gpr_from_dem(x, y, Z, max_train=20000, nu=1.5)
    print("Fit GPR to DEM")
    Z_pred, Z_std = predict_grid_chunked(gpr, x, y, chunk_rows=256, return_std=True)


def test_local_gpr_fit():
    X, Y, Z = prepare_data()
    x = X[0, :]
    y = Y[:, 0]
    print("Fitting local GPR to DEM...")
    Z_local = predict_local_kriging(x, y, Z, xq=x, yq=y, tile=128, k_neighbors=1000)
    print(f"Local GPR MAE: {np.abs(Z_local - Z).mean():.5f} m")


if __name__ == "__main__":
    # test_spline_fit()
    test_local_gpr_fit()
