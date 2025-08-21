import torch

from nemo.util.plotting import plot_surface
from nemo.dem import DEM
from nemo.nemov2 import NEMoV2

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the lunar DEM data
dem = DEM.from_file("../../data/Moon_Map_01_0_rep0.dat")

# Downsample to manageable size for testing
dem_ds = dem.downsample(1)  # Reduce from 180x180 to 45x45

# Get XYZ data
xyz = dem_ds.get_xyz_combined()

# Convert to torch tensors and move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
xy = torch.from_numpy(xyz[:, :2]).float().to(device)
z = torch.from_numpy(xyz[:, 2]).float().to(device)

print(f"Real LDEM data shape: xy={xy.shape}, z={z.shape}")
print(
    f"Coordinate ranges: X({xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}), Y({xy[:, 1].min():.3f}, {xy[:, 1].max():.3f})"
)
print(f"Elevation range: Z({z.min():.3f}, {z.max():.3f})")

# Create new NEMoV2 instance
nemov2 = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

# Test fitting with conservative settings
print("\nFitting NEMoV2 to real LDEM data...")
losses = nemov2.fit(
    xy,
    z,
    lr=1e-3,
    max_epochs=5000,
    batch_size=20000,
    verbose=False,
    early_stopping=False,
    enable_spatial=False,
)

# Plot the fitted surface
pred_z = nemov2(xy)

pred_grid = dem_ds.data.copy()
pred_grid[:, :, 2] = (
    pred_z.detach().cpu().numpy().reshape(dem_ds.data.shape[0], dem_ds.data.shape[1])
)

plot_surface(pred_grid)
