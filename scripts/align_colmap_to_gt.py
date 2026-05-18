from scipy.interpolate import RegularGridInterpolator
"""Align COLMAP sparse point cloud to Ground Truth poses and visualize."""

import argparse
import json
import os
import struct
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
AIRSIM_SPIRAL_CENTER = [524.38, 168.34, 0.0]
Z_CORRECTION = 24.15
from scipy.spatial.transform import Rotation as R

# --- COLMAP BIN READER HELPERS ---

def read_next_bytes(f, num_bytes, format_char):
    return struct.unpack(format_char, f.read(num_bytes))

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

# --- ALIGNMENT ---


def apply_similarity(pts, s, Rot, t):
    return (s * (Rot @ pts.T).T) + t

def umeyama(X, Y):
    """X, Y: (N, 3). Returns s, R, t such that Y = s*R*X + t"""
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

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap-dir", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/colmap_alignment/colmap/sparse/0"))
    parser.add_argument("--transforms", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/transforms.json"))
    parser.add_argument("--gt-dem", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/dem_pc.npy"))
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_diagnostics/colmap_aligned.html"))
    args = parser.parse_args()

    # 1. Load COLMAP centers
    print("Loading COLMAP images...")
    colmap_images = read_images_bin(args.colmap_dir / "images.bin")
    
    # 2. Load GT centers from transforms.json
    print("Loading GT poses...")
    with open(args.transforms, "r") as f:
        tf = json.load(f)
    
    gt_centers = {}
    for frame in tf["frames"]:
        name = Path(frame["file_path"]).name
        mat = np.array(frame["transform_matrix"])
        gt_centers[name] = mat[:3, 3] # Translation part
        
    # 3. Match
    # Mapping: frame_00001.png -> images/0.png
    import re
    
    paired_colmap = []
    paired_gt = []
    common_names = []
    
    for c_name, c_center in colmap_images.items():
        # Extract number from frame_XXXXX.png
        match = re.search(r"(\d+)", c_name)
        if match:
            idx = int(match.group(1)) - 1
            gt_name = f"{idx}.png"
            if gt_name in gt_centers:
                paired_colmap.append(c_center)
                common_names.append((c_name, gt_name))
                paired_gt.append(gt_centers[gt_name])
                
    X = np.array(paired_colmap) # Source
    Y = np.array(paired_gt)     # Target
    
    print(f"Matched {len(X)} cameras for alignment.")
    
    if len(X) < 3:
        print("Error: Not enough common cameras to compute alignment.")
        print("COLMAP samples:", list(colmap_images.keys())[:5])
        print("GT samples:", list(gt_centers.keys())[:5])
        return
    
    # 4. Align
    s, R_align, t = umeyama(X, Y)
    print(f"Alignment Results:")
    print(f"  Scale: {s:.6f}")
    print(f"  Rotation:\n{R_align}")
    print(f"  Translation: {t}")
    
    # 5. Transform Point Cloud
    print("Transforming COLMAP points...")
    colmap_pc = read_points3d_bin(args.colmap_dir / "points3D.bin")
    aligned_pc_ds = (s * (R_align @ colmap_pc.T).T) + t
    
    # 6. Transform everything to AirSim Space (90-deg rotation + flip fix)
    # Mapping from final correct check: X_as = Y_ds + 524.38, Y_as = X_ds + 168.34, Z_as = Z_ds - 24.15
    def to_airsim(pts):
        as_x = pts[:, 1] + 524.38
        as_y = pts[:, 0] + 168.34
        as_z = pts[:, 2] - 24.15
        return np.stack([as_x, as_y, as_z], axis=-1)

    as_colmap = to_airsim(aligned_pc_ds)
    
    print("Loading GT DEM...")
    gt_full = np.load(args.gt_dem)
    gt_sample = gt_full[np.random.choice(len(gt_full), 50000, replace=False)]
    as_gt = to_airsim(gt_sample)

    
    # 6a. Create GT Surface Grid in AirSim Space
    print("Generating GT DEM surface...")
    order = np.lexsort((gt_full[:, 0], gt_full[:, 1]))
    sorted_dem = gt_full[order]
    xs_unq = np.unique(sorted_dem[:, 0])
    ys_unq = np.unique(sorted_dem[:, 1])
    z_grid = sorted_dem[:, 2].reshape(len(ys_unq), len(xs_unq))
    dem_interp = RegularGridInterpolator((ys_unq, xs_unq), z_grid, bounds_error=False, fill_value=np.nan)

    res = 128
    as_x_axis = np.linspace(AIRSIM_SPIRAL_CENTER[0] - 1000, AIRSIM_SPIRAL_CENTER[0] + 1000, res)
    as_y_axis = np.linspace(AIRSIM_SPIRAL_CENTER[1] - 1000, AIRSIM_SPIRAL_CENTER[1] + 1000, res)
    as_xx, as_yy = np.meshgrid(as_x_axis, as_y_axis, indexing="xy")
    
    ds_xx_grid = as_yy - 168.34
    ds_yy_grid = as_xx - 524.38
    as_z_grid = dem_interp(np.stack([ds_xx_grid.flatten(), ds_yy_grid.flatten()], axis=-1)).reshape(res, res)
    as_z_final = as_z_grid - 24.15

# 7. Visualization
    fig = go.Figure()
    
    # 7a. GT Surface (Sub-sampled for visibility)
    fig.add_trace(go.Surface(
        x=as_xx, y=as_yy, z=as_z_final,
        colorscale="Greens", opacity=0.4, name="GT DEM Surface",
        showscale=False
    ))
    
    # 7b. Aligned COLMAP Points (Red)
    fig.add_trace(go.Scatter3d(
        x=as_colmap[:, 0], y=as_colmap[:, 1], z=as_colmap[:, 2],
        mode="markers", marker=dict(size=1.5, color="red", opacity=0.6), name="Aligned SfM Points"
    ))
    
    # 7c. Trajectories
    # GT Path (Blue)
    gt_path = np.array([gt_centers[gn] for cn, gn in common_names])
    as_gt_path = to_airsim(gt_path)
    fig.add_trace(go.Scatter3d(
        x=as_gt_path[:, 0], y=as_gt_path[:, 1], z=as_gt_path[:, 2],
        mode="lines+markers", line=dict(color="blue", width=4),
        marker=dict(size=3), name="GT Camera Path"
    ))
    
    # Aligned COLMAP Path (Yellow)
    colmap_path = np.array([colmap_images[cn] for cn, gn in common_names])
    aligned_colmap_ds = apply_similarity(colmap_path, s, R_align, t) # We use original t from Umeyama
    as_colmap_path = to_airsim(aligned_colmap_ds)
    fig.add_trace(go.Scatter3d(
        x=as_colmap_path[:, 0], y=as_colmap_path[:, 1], z=as_colmap_path[:, 2],
        mode="lines+markers", line=dict(color="yellow", width=4),
        marker=dict(size=3), name="Aligned SfM Path"
    ))
    
    # Calculate Residual Trajectory Error
    traj_rmse = np.sqrt(np.mean(np.linalg.norm(as_gt_path - as_colmap_path, axis=1)**2))
    print(f"Mean Trajectory Alignment Error: {traj_rmse:.2f} meters")

    fig.update_layout(
        title=f"COLMAP Alignment Diagnostic (Traj RMSE: {traj_rmse:.2f}m)",
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
    print(f"Alignment verified. Plot saved to {args.output}")

if __name__ == "__main__":
    main()
