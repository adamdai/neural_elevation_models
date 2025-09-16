import numpy as np
from pykrige.ok import OrdinaryKriging
from pathlib import Path
import gc
import psutil
import os
from tqdm import tqdm

from nemo.dem import DEM
from nemo.util.paths import data_dir

# Configuration parameters
GRID_RESOLUTION = 800  # Reduce this if you still get memory errors
TILE_SIZE = 128  # Reduce this if you still get memory errors
MAX_TRAINING_POINTS = 20000  # Maximum number of training points to use


def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")


print(f"Configuration: Grid={GRID_RESOLUTION}x{GRID_RESOLUTION}, Tiles={TILE_SIZE}x{TILE_SIZE}")
print(f"Max training points: {MAX_TRAINING_POINTS:,}")
print_memory_usage()

# 1) Your samples (scattered or gridded flattened)
# x, y: 1D coords of N samples; z: elevations (N,)
# If you start from a regular DEM Z[y,x], do:
# Xg, Yg = np.meshgrid(x_axis, y_axis, indexing="xy")
# x, y, z = Xg.ravel(), Yg.ravel(), Z.ravel()
tif_path = Path("data/Site01_final_adj_5mpp_surf.tif")
dem = DEM.from_file(tif_path)
# dem = DEM.from_file(data_dir() / "Moon_Map_01_0_rep0.dat")
XYZ = dem.data
X = XYZ[:, :, 0]
Y = XYZ[:, :, 1]
Z = XYZ[:, :, 2]
x, y, z = X.ravel(), Y.ravel(), Z.ravel()


# 2) (Optional) subsample to keep training light
def subsample(x, y, z, max_n=20000, seed=0):
    rng = np.random.RandomState(seed)
    if len(z) > max_n:
        idx = rng.choice(len(z), max_n, replace=False)
        return x[idx], y[idx], z[idx]
    return x, y, z


xs, ys, zs = subsample(x, y, z, max_n=MAX_TRAINING_POINTS)

# 3) Fit Ordinary Kriging with a common variogram model
OK = OrdinaryKriging(
    xs,
    ys,
    zs,
    variogram_model="spherical",  # "exponential", "gaussian", etc.
    # If you already know parameters, pass dict:
    # variogram_parameters={"sill": 500, "range": 1500, "nugget": 5},
    enable_plotting=False,
    coordinates_type="euclidean",
)

# 4) Predict on a grid (tile if large)
# Reduce resolution to prevent memory issues
xq = np.linspace(x.min(), x.max(), GRID_RESOLUTION)
yq = np.linspace(y.min(), y.max(), GRID_RESOLUTION)

print(f"Grid size: {GRID_RESOLUTION}x{GRID_RESOLUTION} = {GRID_RESOLUTION**2:,} points")
print(f"Training data: {len(zs):,} points")


# Tiled execution (prevents RAM blowups on large grids)
def krige_grid_tiled(OK, xq, yq, tile_rows=128, tile_cols=128):
    Z_pred = np.empty((len(yq), len(xq)), float)
    Var = np.empty_like(Z_pred)

    total_tiles = ((len(yq) + tile_rows - 1) // tile_rows) * (
        (len(xq) + tile_cols - 1) // tile_cols
    )
    tile_count = 0

    # Create progress bar for tiles
    pbar = tqdm(total=total_tiles, desc="Processing tiles", unit="tile")

    for y0 in range(0, len(yq), tile_rows):
        y1 = min(y0 + tile_rows, len(yq))
        for x0 in range(0, len(xq), tile_cols):
            x1 = min(x0 + tile_cols, len(xq))

            # Extract tile coordinates
            xq_tile = xq[x0:x1]
            yq_tile = yq[y0:y1]

            # Execute kriging on this tile
            zhat, var = OK.execute(style="grid", xpoints=xq_tile, ypoints=yq_tile)

            # Store results
            Z_pred[y0:y1, x0:x1] = zhat
            Var[y0:y1, x0:x1] = var

            tile_count += 1
            pbar.set_postfix({"y": f"{y0}:{y1}", "x": f"{x0}:{x1}"})
            pbar.update(1)

            # Clean up tile variables and force garbage collection every few tiles
            if tile_count % 10 == 0:
                del zhat, var
                gc.collect()
                print_memory_usage()

    pbar.close()
    return Z_pred, Var


# Alternative memory-efficient approach using points style
def krige_grid_memory_efficient(OK, xq, yq, tile_rows=128, tile_cols=128):
    """
    Memory-efficient kriging using points style instead of grid style.
    This approach processes individual points rather than creating full grids.
    """
    Z_pred = np.empty((len(yq), len(xq)), float)
    Var = np.empty_like(Z_pred)

    total_tiles = ((len(yq) + tile_rows - 1) // tile_rows) * (
        (len(xq) + tile_cols - 1) // tile_cols
    )
    tile_count = 0

    # Create progress bar for tiles
    pbar = tqdm(total=total_tiles, desc="Processing tiles (memory-efficient)", unit="tile")

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
            if tile_count % 10 == 0:
                del X_tile, Y_tile, coords_tile, zhat, var
                gc.collect()
                print_memory_usage()

    pbar.close()
    return Z_pred, Var


# Choose which method to use based on grid size
print_memory_usage()
if GRID_RESOLUTION <= 400:
    print("Using grid-style kriging (faster for small grids)")
    Zk, Vk = krige_grid_tiled(OK, xq, yq, tile_rows=TILE_SIZE, tile_cols=TILE_SIZE)
else:
    print("Using memory-efficient points-style kriging")
    Zk, Vk = krige_grid_memory_efficient(OK, xq, yq, tile_rows=TILE_SIZE, tile_cols=TILE_SIZE)

print_memory_usage()
gc.collect()  # Force garbage collection
print_memory_usage()

# Evaluate the error (only if original DEM data is available and comparable)
try:
    # Resample original DEM to match prediction grid for comparison
    from scipy.interpolate import griddata

    # Create a grid from the original DEM data for comparison
    X_orig, Y_orig = np.meshgrid(xq, yq, indexing="xy")

    # Interpolate original DEM to the prediction grid
    Z_orig_interp = griddata((x, y), z, (X_orig, Y_orig), method="linear", fill_value=np.nan)

    # Calculate error only where we have valid data
    valid_mask = ~np.isnan(Z_orig_interp)
    if valid_mask.any():
        error = np.abs(Zk[valid_mask] - Z_orig_interp[valid_mask])
        print(f"Kriging MAE: {error.mean():.5f} m")
        print(f"Kriging RMSE: {np.sqrt((error**2).mean()):.5f} m")
        print(f"Valid comparison points: {valid_mask.sum():,}")
    else:
        print("No valid comparison points available")

except Exception as e:
    print(f"Could not evaluate error: {e}")
    print("Continuing without error evaluation...")

# Clean up large arrays to free memory
del X_orig, Y_orig, Z_orig_interp
if "valid_mask" in locals():
    del valid_mask
if "error" in locals():
    del error

print("Kriging completed successfully!")
print(f"Prediction shape: {Zk.shape}")
print(f"Variance shape: {Vk.shape}")

# Save results
print("Saving results...")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Save kriging results
np.save(output_dir / "kriging_prediction.npy", Zk)
np.save(output_dir / "kriging_variance.npy", Vk)
np.save(output_dir / "kriging_x_coords.npy", xq)
np.save(output_dir / "kriging_y_coords.npy", yq)

# Save training data used for the model
np.save(output_dir / "training_x.npy", xs)
np.save(output_dir / "training_y.npy", ys)
np.save(output_dir / "training_z.npy", zs)

# Save kriging model parameters
model_params = {
    "variogram_model": OK.variogram_model,
    "variogram_parameters": OK.variogram_model_parameters,
    "coordinates_type": OK.coordinates_type,
    "grid_resolution": GRID_RESOLUTION,
    "tile_size": TILE_SIZE,
    "max_training_points": MAX_TRAINING_POINTS,
    "x_range": [x.min(), x.max()],
    "y_range": [y.min(), y.max()],
    "z_range": [z.min(), z.max()],
}
np.save(output_dir / "kriging_model_params.npy", model_params)

print("Results saved to {output_dir}/")
print("Files saved:")
print(f"  - kriging_prediction.npy: {Zk.shape} prediction grid")
print(f"  - kriging_variance.npy: {Vk.shape} variance grid")
print(f"  - kriging_x_coords.npy: {len(xq)} x-coordinates")
print(f"  - kriging_y_coords.npy: {len(yq)} y-coordinates")
print(f"  - training_x.npy: {len(xs)} training x-coordinates")
print(f"  - training_y.npy: {len(ys)} training y-coordinates")
print(f"  - training_z.npy: {len(zs)} training elevations")
print("  - kriging_model_params.npy: Model parameters and metadata")


# Example of how to load and use the saved results later
def load_kriging_results(output_dir="output"):
    """Load saved kriging results for later use"""
    output_path = Path(output_dir)

    # Load predictions and variance
    Zk = np.load(output_path / "kriging_prediction.npy")
    Vk = np.load(output_path / "kriging_variance.npy")

    # Load coordinates
    xq = np.load(output_path / "kriging_x_coords.npy")
    yq = np.load(output_path / "kriging_y_coords.npy")

    # Load training data
    xs = np.load(output_path / "training_x.npy")
    ys = np.load(output_path / "training_y.npy")
    zs = np.load(output_path / "training_z.npy")

    # Load model parameters
    model_params = np.load(output_path / "kriging_model_params.npy", allow_pickle=True).item()

    print("Loaded kriging results:")
    print(f"  Prediction grid: {Zk.shape}")
    print(f"  Variance grid: {Vk.shape}")
    print(f"  Training points: {len(xs)}")
    print(f"  Model: {model_params['variogram_model']}")

    return Zk, Vk, xq, yq, xs, ys, zs, model_params


print("\nTo load these results later, use:")
print("Zk, Vk, xq, yq, xs, ys, zs, params = load_kriging_results()")
