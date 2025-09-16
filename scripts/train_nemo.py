import torch
import numpy as np
from pathlib import Path
from rich import print
import datetime

from nemo.util.plotting import plot_surface
from nemo.dem import DEM
from nemo.nemov2 import NEMoV2
from nemo.logger import Logger
from nemo.util.paths import config_dir, data_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

DEM_DATA = "LDEM"

if __name__ == "__main__":
    logger = Logger(
        project_name="neural-elevation-models",
        run_name="nemo-v2",
        config={
            "lr": 1e-3,
            "max_epochs": 5000,
            "batch_size": 20000,
        },
    )

    if DEM_DATA == "LAC":
        dem = DEM.from_file(data_dir() / "Moon_Map_01_0_rep0.dat")
    else:
        tif_path = Path(data_dir() / "Site01_final_adj_5mpp_surf.tif")
        dem = DEM.from_file(tif_path)

    xyz = dem.get_xyz_combined()
    xy = torch.from_numpy(xyz[:, :2]).float().to(device)
    z = torch.from_numpy(xyz[:, 2]).float().to(device)

    print(f"Real LDEM data shape: xy={xy.shape}, z={z.shape}")
    print(
        f"Coordinate ranges: X({xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}), Y({xy[:, 1].min():.3f}, {xy[:, 1].max():.3f})"
    )
    print(f"Elevation range: Z({z.min():.3f}, {z.max():.3f})")

    nemo = NEMoV2(device=device)
    nemo.from_config(config_dir() / "nemo.yaml")
    nemo.compute_scaling_parameters(xy, z)
    nemo.fit(
        xy,
        z,
        lr=1e-3,
        max_epochs=5000,
        batch_size=20000,
        verbose=False,
        early_stopping=False,
        logger=logger,
    )

    # TODO: evaluate height error in meters
    heights = nemo.predict_height(xy)
    height_error = heights - z
    print(f"Height MAE: {height_error.abs().mean():.3f} m")
    print(f"Height error std: {height_error.std():.3f} m")
    print(f"Height error max: {height_error.abs().max():.3f} m")
    print(f"Height RMSE: {height_error.pow(2).mean().sqrt():.3f} m")

    # Create timestamped folder for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"results/{DEM_DATA}_nemo_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model and config
    nemo.save_model(save_dir / "model.pth")
