"""Debug and refine COLMAP point cloud alignment using ICP against GT DEM."""

import argparse
import json
import os
import struct
import re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator

# AirSim Constants
AIRSIM_SPIRAL_CENTER = np.array([524.38, 168.34, 0.0])
Z_CORRECTION = 24.15 # AirSim_Z = DS_Z - 24.15

# --- HELPERS ---

def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num_reg_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32)) # qw, qx, qy, qz
            t = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                char = f.read(1).decode("utf-8")
                if char == "\0": break
                name += char
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points2d * 24, 1)
            # center = -R^T * t
            rot = R.from_quat([q[1], q[2], q[3], q[0]])
            center = -rot.inv().apply(t)
            images[name] = center
    return images

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

def umeyama(X, Y):
    """Y = s*R*X + t"""
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    sigma_x = np.mean(np.linalg.norm(X - mu_x, axis=1)**2)
    Sigma = ((Y - mu_y).T @ (X - mu_x)) / len(X)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    Rot = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / sigma_x
    trans = mu_y - scale * Rot @ mu_x
    return scale, Rot, trans

def apply_similarity(pts, s, Rot, t):
    return (s * (Rot @ pts.T).T) + t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap-dir", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/colmap_alignment/colmap/sparse/0"))
    parser.add_argument("--transforms", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/transforms.json"))
    parser.add_argument("--gt-dem", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/dem_pc.npy"))
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_diagnostics/colmap_debug_alignment.html"))
    args = parser.parse_args()

    # 1. Load Data
    print("Loading COLMAP data...")
    colmap_images = read_images_bin(args.colmap_dir / "images.bin")
    colmap_pc = read_points3d_bin(args.colmap_dir / "points3D.bin")
    
    print("Loading GT poses...")
    with open(args.transforms, "r") as f:
        tf = json.load(f)
    gt_centers = {Path(fr["file_path"]).name: np.array(fr["transform_matrix"])[:3, 3] for fr in tf["frames"]}

    # 2. Match Cameras for Initial Umeyama
    paired_colmap = []
    paired_gt = []
    for c_name, c_center in colmap_images.items():
        match = re.search(r"(\d+)", c_name)
        if match:
            gt_name = f"{int(match.group(1))-1}.png"
            if gt_name in gt_centers:
                paired_colmap.append(c_center)
                paired_gt.append(gt_centers[gt_name])
    
    X = np.array(paired_colmap)
    Y = np.array(paired_gt)
    s0, R0, t0 = umeyama(X, Y)
    
    # 3. Setup DEM Interpolator for Refinement
    print("Loading GT DEM for ICP...")
    gt_full = np.load(args.gt_dem)
    order = np.lexsort((gt_full[:, 0], gt_full[:, 1]))
    sorted_dem = gt_full[order]
    xs_unq = np.unique(sorted_dem[:, 0])
    ys_unq = np.unique(sorted_dem[:, 1])
    z_grid = sorted_dem[:, 2].reshape(len(ys_unq), len(xs_unq))
    dem_interp = RegularGridInterpolator((ys_unq, xs_unq), z_grid, bounds_error=False, fill_value=np.nan)

    # 4. Refinement Loop (Simple ICP)
    curr_s, curr_R, curr_t = s0, R0, t0
    
    # Crop COLMAP to reasonable radius early to save compute
    # Radius of 2000m from spiral center in AirSim => ...
    # but ICP works in Dataset space. Let\''s just use all points for now if count < 100k.
    pts = colmap_pc
    if len(pts) > 50000:
        pts = pts[np.random.choice(len(pts), 50000, replace=False)]

    print("Refining alignment...")
    for i in range(10):
        # 1. Transform to Dataset space
        pts_ds = apply_similarity(pts, curr_s, curr_R, curr_t)
        
        # 2. Find "closest points" on DEM (Vertical projection)
        # Using [y, x] for the interpolator
        query_yx = np.stack([pts_ds[:, 1], pts_ds[:, 0]], axis=-1)
        z_gt = dem_interp(query_yx)
        
        valid = np.isfinite(z_gt)
        source = pts[valid]
        target = pts_ds[valid].copy()
        target[:, 2] = z_gt[valid]
        
        # 3. Update similarity transform to map source to vertical-corrected targets
        curr_s, curr_R, curr_t = umeyama(source, target)
        
        rmse = np.sqrt(np.mean((pts_ds[valid, 2] - z_gt[valid])**2))
        print(f"  Iteration {i}: Vertical RMSE = {rmse:.4f}m")

    # Final Alignment
    aligned_pc_ds = apply_similarity(colmap_pc, curr_s, curr_R, curr_t)
    
    # 5. Transform to AirSim Space (Fixed project convention)
    def to_airsim(p):
        return np.stack([p[:, 1] + 524.38, p[:, 0] + 168.34, p[:, 2] - 24.15], axis=-1)

    as_colmap = to_airsim(aligned_pc_ds)
    
    # 5a. Create GT Surface Grid in AirSim Space
    print("Generating GT DEM surface...")
    res = 256
    as_x_axis = np.linspace(AIRSIM_SPIRAL_CENTER[0] - 1000, AIRSIM_SPIRAL_CENTER[0] + 1000, res)
    as_y_axis = np.linspace(AIRSIM_SPIRAL_CENTER[1] - 1000, AIRSIM_SPIRAL_CENTER[1] + 1000, res)
    as_xx, as_yy = np.meshgrid(as_x_axis, as_y_axis, indexing="xy")
    
    # Map AS grid to DS for interpolation
    ds_xx_grid = as_yy - 168.34
    ds_yy_grid = as_xx - 524.38
    as_z_grid = dem_interp(np.stack([ds_xx_grid.flatten(), ds_yy_grid.flatten()], axis=-1)).reshape(res, res)
    # Z Correction for Elevation
    as_z_final = as_z_grid - 24.15
    
    # Radius Crop in AirSim Space (2000m from spiral center)
    dist_sq = (as_colmap[:, 0] - 524.38)**2 + (as_colmap[:, 1] - 168.34)**2
    as_colmap = as_colmap[dist_sq < 2000**2]

    # 6. Visualization
    fig = go.Figure()

    # GT Surface (Elevation view, so -Z)
    fig.add_trace(go.Surface(
        x=as_x_axis, y=as_y_axis, z=as_z_final,
        colorscale="Greens", opacity=0.8, name="GT DEM Surface",
        showscale=False
    ))

    # COLMAP Points
    fig.add_trace(go.Scatter3d(
        x=as_colmap[:, 0], y=as_colmap[:, 1], z=as_colmap[:, 2],
        mode="markers", marker=dict(size=2, color="red", opacity=1.0), name="Refined COLMAP Points"
    ))
    
    fig.update_layout(
        title=f"Refined COLMAP overlaid on GT DEM (Vertical RMSE: {rmse:.2f}m)",
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
    print(f"Alignment refined and saved to {args.output}")

if __name__ == "__main__":
    main()
