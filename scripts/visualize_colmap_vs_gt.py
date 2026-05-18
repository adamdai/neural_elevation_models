"""Visualize COLMAP sparse point cloud overlaid on the Ground Truth DEM."""

import argparse
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

# Constants
AIRSIM_SPIRAL_CENTER = np.array([524.38, 168.34, 0.0])
Z_CORRECTION = 24.15 # AirSim_Z = DS_Z - 24.15

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/colmap_pc.npy"))
    parser.add_argument("--gt", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/dem_pc.npy"))
    parser.add_argument("--transform", type=Path, default=Path("/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-16_threshold-vertical-fix_5000/dataparser_transforms.json"))
    parser.add_argument("--num-gt-points", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_diagnostics/colmap_vs_gt.html"))
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading COLMAP points: {args.colmap}")
    colmap_pts = np.load(args.colmap)
    
    print(f"Loading GT points: {args.gt}")
    gt_full = np.load(args.gt)
    
    print(f"Loading transforms: {args.transform}")
    with open(args.transform, "r") as f:
        meta = json.load(f)
    
    T_dp = np.array(meta["transform"]) # (3, 4)
    scale = meta["scale"]
    
    # 2. Align COLMAP to Dataset Space
    # In nerfstudio, Training_Pt = (T @ Raw_Pt_h) * scale
    # So: Raw_Pt_h = T_inv @ (Training_Pt / scale)
    # The provided transform T is [R | t]
    R = T_dp[:3, :3]
    t = T_dp[:3, 3]
    
    # Dataset_pts = R_inv @ (colmap_pts / scale - t)
    # Since R is identity in our case, it simplifies:
    ds_colmap = (colmap_pts / scale) - t
    
    # 3. Transform everything to AirSim Space
    def to_airsim(pts):
        # X_as = Y_ds + Center_X
        # Y_as = X_ds + Center_Y
        # Z_as = Z_ds - Z_Correction
        as_x = pts[:, 1] + AIRSIM_SPIRAL_CENTER[0]
        as_y = pts[:, 0] + AIRSIM_SPIRAL_CENTER[1]
        as_z = pts[:, 2] - Z_CORRECTION
        return np.stack([as_x, as_y, as_z], axis=-1)

    # Transform GT (sub-sampled)
    gt_subset = gt_full[np.random.choice(len(gt_full), args.num_gt_points, replace=False)]
    as_gt = to_airsim(gt_subset)
    
    # Transform COLMAP
    as_colmap = to_airsim(ds_colmap)

    # 4. Visualization
    fig = go.Figure()

    # GT Trace (Green)
    fig.add_trace(go.Scatter3d(
        x=as_gt[:, 0], y=as_gt[:, 1], z=as_gt[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="green", opacity=0.3),
        name="Ground Truth DEM"
    ))

    # COLMAP Trace (Red)
    fig.add_trace(go.Scatter3d(
        x=as_colmap[:, 0], y=as_colmap[:, 1], z=as_colmap[:, 2],
        mode="markers",
        marker=dict(size=2, color="red", opacity=0.8),
        name="COLMAP SfM Points"
    ))

    fig.update_layout(
        title="COLMAP Sparse Points vs Ground Truth DEM (AirSim Frame)",
        scene=dict(
            aspectmode="data",
            xaxis_title="AirSim X (Right)",
            yaxis_title="AirSim Y (Down)",
            zaxis_title="Z (Elevation)",
            yaxis=dict(autorange="reversed")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output)
    print(f"Saved comparison plot to {args.output}")
    print(f"View at: http://127.0.0.1:8000/{args.output}")

if __name__ == "__main__":
    main()
