"""Visualize COLMAP overlay using the robustly estimated similarity transform."""

import argparse
import json
import os
import struct
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator

# AirSim Constants
AIRSIM_SPIRAL_CENTER = np.array([524.38, 168.34, 0.0])
Z_CORRECTION = 24.15 # AirSim_Z = DS_Z - 24.15

# Robust Transform (COLMAP -> Dataset)
S_ROBUST = 145.177055
R_ROBUST = np.array([
    [ 0.84097628, -0.32481522,  0.43272851],
    [-0.5310307 , -0.64884621,  0.54498165],
    [ 0.10375592, -0.68810876, -0.71815113]
])
T_ROBUST = np.array([-3.68426704, 3.52782202, 384.97561668])

def read_points3d_bin(path):
    points = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, 1)
            points.append(xyz)
    return np.array(points)

def to_airsim(pts):
    """X_as = Y_ds + 524.38, Y_as = X_ds + 168.34, Z_as = Z_ds - 24.15"""
    as_x = pts[:, 1] + AIRSIM_SPIRAL_CENTER[0]
    as_y = pts[:, 0] + AIRSIM_SPIRAL_CENTER[1]
    as_z = pts[:, 2] - Z_CORRECTION
    return np.stack([as_x, as_y, as_z], axis=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap-bin", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/colmap_alignment/colmap/sparse/0/points3D.bin"))
    parser.add_argument("--gt-dem", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/dem_pc.npy"))
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_diagnostics/colmap_robust_aligned.html"))
    args = parser.parse_args()

    # 1. Load Data
    print("Loading COLMAP points...")
    colmap_pc = read_points3d_bin(args.colmap_bin)
    
    # 2. Apply Robust Similarity (COLMAP -> Dataset)
    print("Applying robust similarity transform...")
    aligned_pc_ds = (S_ROBUST * (R_ROBUST @ colmap_pc.T).T) + T_ROBUST
    
    # 3. Transform to AirSim Space
    as_colmap = to_airsim(aligned_pc_ds)
    
    # 4. Load GT for Surface Plot
    print("Loading GT DEM...")
    gt_full = np.load(args.gt_dem)
    
    # Create Surface
    order = np.lexsort((gt_full[:, 0], gt_full[:, 1]))
    sorted_dem = gt_full[order]
    xs_unq = np.unique(sorted_dem[:, 0])
    ys_unq = np.unique(sorted_dem[:, 1])
    z_grid = sorted_dem[:, 2].reshape(len(ys_unq), len(xs_unq))
    dem_interp = RegularGridInterpolator((ys_unq, xs_unq), z_grid, bounds_error=False, fill_value=np.nan)

    res = 256
    as_x_axis = np.linspace(AIRSIM_SPIRAL_CENTER[0] - 1000, AIRSIM_SPIRAL_CENTER[0] + 1000, res)
    as_y_axis = np.linspace(AIRSIM_SPIRAL_CENTER[1] - 1000, AIRSIM_SPIRAL_CENTER[1] + 1000, res)
    as_xx, as_yy = np.meshgrid(as_x_axis, as_y_axis, indexing="xy")
    
    ds_xx_grid = as_yy - 168.34
    ds_yy_grid = as_xx - 524.38
    as_z_grid = dem_interp(np.stack([ds_xx_grid.flatten(), ds_yy_grid.flatten()], axis=-1)).reshape(res, res)
    as_z_final = as_z_grid - 24.15

    # 5. Visualization
    fig = go.Figure()

    # GT Surface (Elevation view, plot -Z for orientation but we established Z is Down, so Z_elev = -Z_as)
    # Wait, my previous correct check used as_z directly and it worked? 
    # Let\''s stick to what was finally correct.
    fig.add_trace(go.Surface(
        x=as_x_axis, y=as_y_axis, z=as_z_final,
        colorscale="Greens", opacity=0.6, name="GT DEM Surface",
        showscale=False
    ))

    # COLMAP Points
    # Filter by radius for cleaner view
    dist_sq = (as_colmap[:, 0] - AIRSIM_SPIRAL_CENTER[0])**2 + (as_colmap[:, 1] - AIRSIM_SPIRAL_CENTER[1])**2
    mask = dist_sq < 1500**2
    colmap_plot = as_colmap[mask]
    if len(colmap_plot) > 100000:
        colmap_plot = colmap_plot[np.random.choice(len(colmap_plot), 100000, replace=False)]

    fig.add_trace(go.Scatter3d(
        x=colmap_plot[:, 0], y=colmap_plot[:, 1], z=colmap_plot[:, 2],
        mode="markers", marker=dict(size=1.5, color="red", opacity=0.8), name="Robust Aligned SfM"
    ))

    fig.update_layout(
        title="COLMAP Robust Alignment (RANSAC Transform)",
        scene=dict(
            aspectmode="data",
            xaxis_title="AirSim X",
            yaxis_title="AirSim Y",
            zaxis_title="Elev",
            yaxis=dict(autorange="reversed")
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1200, height=800
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output)
    print(f"Robust alignment plot saved to {args.output}")

if __name__ == "__main__":
    main()
