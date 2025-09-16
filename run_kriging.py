#!/usr/bin/env python3
"""
Standalone kriging interpolation script.
This script demonstrates how to perform kriging interpolation on DEM data
with memory-efficient tiling to avoid memory issues.
"""

import numpy as np
from pykrige.ok import OrdinaryKriging
from pathlib import Path
import gc
import psutil
import os
from tqdm import tqdm

from nemo.dem import DEM

# Configuration parameters - adjust these based on your system's memory
GRID_RESOLUTION = 400  # This is now just a reference, actual grid uses original DEM size
TILE_SIZE = 128  # Larger tiles for better memory efficiency with larger grid
MAX_TRAINING_POINTS = 10000  # Fewer training points for faster processing


def print_memory_usage():
    """Print current memory usage"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.1f} MB")
    except ImportError:
        print("psutil not available - cannot monitor memory usage")


def subsample(x, y, z, max_n=10000, seed=0):
    """Subsample data to keep training manageable"""
    rng = np.random.RandomState(seed)
    if len(z) > max_n:
        idx = rng.choice(len(z), max_n, replace=False)
        return x[idx], y[idx], z[idx]
    return x, y, z


def krige_grid_memory_efficient(OK, xq, yq, tile_rows=64, tile_cols=64):
    """
    Memory-efficient kriging using points style.
    This approach processes individual points rather than creating full grids.
    """
    Z_pred = np.empty((len(yq), len(xq)), float)
    Var = np.empty_like(Z_pred)

    total_tiles = ((len(yq) + tile_rows - 1) // tile_rows) * (
        (len(xq) + tile_cols - 1) // tile_cols
    )
    tile_count = 0

    print(f"Processing {total_tiles} tiles...")

    # Create progress bar for tiles
    pbar = tqdm(total=total_tiles, desc="Processing tiles", unit="tile")

    for y0 in range(0, len(yq), tile_rows):
        y1 = min(y0 + tile_rows, len(yq))
        for x0 in range(0, len(xq), tile_cols):
            x1 = min(x0 + tile_cols, len(xq))

            # Create coordinate pairs for this tile
            X_tile, Y_tile = np.meshgrid(xq[x0:x1], yq[y0:y1], indexing="xy")
            coords_tile = np.column_stack([X_tile.ravel(), Y_tile.ravel()])

            # Execute kriging on these points
            zhat, var = OK.execute(
                style="points", xpoints=coords_tile[:, 0], ypoints=coords_tile[:, 1]
            )

            # Reshape and store results
            Z_pred[y0:y1, x0:x1] = zhat.reshape(y1 - y0, x1 - x0)
            Var[y0:y1, x0:x1] = var.reshape(y1 - y0, x1 - x0)

            tile_count += 1
            pbar.set_postfix({"y": f"{y0}:{y1}", "x": f"{x0}:{x1}"})
            pbar.update(1)

            # Clean up tile variables and force garbage collection every few tiles
            if tile_count % 5 == 0:
                del X_tile, Y_tile, coords_tile, zhat, var
                gc.collect()
                print_memory_usage()

    pbar.close()
    return Z_pred, Var


def main():
    print("=== Kriging Interpolation Script ===")
    print(f"Configuration: Grid={GRID_RESOLUTION}x{GRID_RESOLUTION}, Tiles={TILE_SIZE}x{TILE_SIZE}")
    print(f"Max training points: {MAX_TRAINING_POINTS:,}")
    print_memory_usage()

    # Load DEM data
    tif_path = Path("data/Site01_final_adj_5mpp_surf.tif")
    if not tif_path.exists():
        print(f"Error: DEM file not found at {tif_path}")
        print("Please ensure the DEM file exists in the data/ directory")
        return

    print(f"Loading DEM from {tif_path}")
    dem = DEM.from_file(tif_path)
    XYZ = dem.data
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2]
    x, y, z = X.ravel(), Y.ravel(), Z.ravel()

    print(f"Original DEM shape: {X.shape}")
    print(f"Total points: {len(z):,}")

    # Subsample for training
    print("Subsampling data for training...")
    xs, ys, zs = subsample(x, y, z, max_n=MAX_TRAINING_POINTS)
    print(f"Training points: {len(zs):,}")

    # Fit Ordinary Kriging
    print("Fitting Ordinary Kriging model...")
    OK = OrdinaryKriging(
        xs,
        ys,
        zs,
        variogram_model="spherical",
        enable_plotting=False,
        coordinates_type="euclidean",
    )

    # Create prediction grid at original DEM resolution
    print("Creating prediction grid...")
    xq = np.linspace(x.min(), x.max(), Z.shape[1])  # Use original X dimension
    yq = np.linspace(y.min(), y.max(), Z.shape[0])  # Use original Y dimension
    print(f"Grid size: {Z.shape[0]}x{Z.shape[1]} = {Z.shape[0] * Z.shape[1]:,} points")

    # Perform kriging
    print("Starting kriging interpolation...")
    print_memory_usage()

    Zk, Vk = krige_grid_memory_efficient(OK, xq, yq, tile_rows=TILE_SIZE, tile_cols=TILE_SIZE)

    print("Kriging completed successfully!")
    print(f"Prediction shape: {Zk.shape}")
    print(f"Variance shape: {Vk.shape}")
    print_memory_usage()

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "kriging_prediction.npy", Zk)
    np.save(output_dir / "kriging_variance.npy", Vk)
    np.save(output_dir / "kriging_x_coords.npy", xq)
    np.save(output_dir / "kriging_y_coords.npy", yq)

    print(f"Results saved to {output_dir}/")

    # Basic statistics
    print("\nResults Summary:")
    print(f"Prediction range: {Zk.min():.2f} to {Zk.max():.2f}")
    print(f"Variance range: {Vk.min():.2f} to {Vk.max():.2f}")
    print(f"Mean variance: {Vk.mean():.2f}")

    # Calculate error metrics at original resolution
    print(f"Max error: {np.max(np.abs(Zk - Z)):.2f}")
    print(f"MAE: {np.mean(np.abs(Zk - Z)):.2f}")
    print(f"RMSE: {np.sqrt(np.mean((Zk - Z) ** 2)):.2f}")


if __name__ == "__main__":
    main()
