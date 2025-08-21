#!/usr/bin/env python3
"""
Test script for NEMoV2 class - specifically designed for fitting to DEM data.
"""

import torch
from rich import print
import argparse
from pathlib import Path

from nemo.dem import DEM
from nemo.nemov2 import NEMoV2
from nemo.logger import Logger

device = "cuda" if torch.cuda.is_available() else "cpu"

tif_path = Path("data/Site01_final_adj_5mpp_surf.tif")
dem = DEM.from_file(tif_path)
dem = dem.downsample(10)


xyz = dem.get_xyz_combined()
xy = torch.from_numpy(xyz[:, :2]).float().to(device)
z = torch.from_numpy(xyz[:, 2]).float().to(device)

print(f"Real LDEM data shape: xy={xy.shape}, z={z.shape}")
print(
    f"Coordinate ranges: X({xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}), Y({xy[:, 1].min():.3f}, {xy[:, 1].max():.3f})"
)
print(f"Elevation range: Z({z.min():.3f}, {z.max():.3f})")

nemo = NEMoV2(device=device)
nemo.compute_scaling_parameters(xy, z)
nemo.fit(
    xy,
    z,
    lr=1e-3,
    max_epochs=1000,
    batch_size=20000,
    verbose=False,
    early_stopping=False,
)
