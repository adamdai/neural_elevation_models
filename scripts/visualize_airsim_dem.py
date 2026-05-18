"""Visualize the GT DEM with the 90-degree rotation to AirSim space."""

import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

# AirSim Constants
AIRSIM_SPIRAL_CENTER = np.array([524.38, 168.34, 0.0])
AIRSIM_START = np.array([0.0, 0.0, 0.0])
AIRSIM_GOAL = np.array([1050.50, 323.84, -19.73])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/dem_pc.npy"))
    parser.add_argument("--num-points", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_diagnostics/airsim_gt_rotated.html"))
    args = parser.parse_args()

    # 1. Load DEM
    print(f"Loading DEM: {args.dem}")
    dem_full = np.load(args.dem)
    
    # 2. Setup Interpolator to find Start Elevation
    order = np.lexsort((dem_full[:, 0], dem_full[:, 1]))
    sorted_dem = dem_full[order]
    xs_unique = np.unique(sorted_dem[:, 0])
    ys_unique = np.unique(sorted_dem[:, 1])
    z_grid = sorted_dem[:, 2].reshape(len(ys_unique), len(xs_unique))
    interp = RegularGridInterpolator((ys_unique, xs_unique), z_grid, bounds_error=False, fill_value=np.nan)

    # Dataset XY corresponding to AirSim Start (0,0)
    # X_as = Y_ds + Cx => 0 = Y_ds + 524.38 => Y_ds = -524.38
    # Y_as = X_ds + Cy => 0 = X_ds + 168.34 => X_ds = -168.34
    ds_start_x = -AIRSIM_SPIRAL_CENTER[1]
    ds_start_y = -AIRSIM_SPIRAL_CENTER[0]
    
    start_z_gt = interp([[ds_start_y, ds_start_x]])[0]
    print(f"GT Elevation at Start ({ds_start_x:.2f}, {ds_start_y:.2f}): {start_z_gt:.2f}m")

    # 3. Transform to AirSim Space
    indices = np.random.choice(len(dem_full), args.num_points, replace=False)
    dem = dem_full[indices]
    
    # Mapping:
    # X_as = Y_ds + Center_X
    # Y_as = X_ds + Center_Y
    as_x = dem[:, 1] + AIRSIM_SPIRAL_CENTER[0]
    as_y = dem[:, 0] + AIRSIM_SPIRAL_CENTER[1]
    # Vertical: Adjust so start is 0
    as_z = dem[:, 2] - start_z_gt

    # Find the "Top" (highest point)
    top_idx = np.argmax(as_z)
    top_pos = np.array([as_x[top_idx], as_y[top_idx], as_z[top_idx]])

    # 4. Visualization
    fig = go.Figure()

    # Plot GT Points
    fig.add_trace(go.Scatter3d(
        x=as_x, y=as_y, z=as_z,
        mode="markers",
        marker=dict(size=1.5, color=as_z, colorscale="Greens", opacity=0.8),
        name="GT DEM Points"
    ))

    # Markers
    fig.add_trace(go.Scatter3d(
        x=[AIRSIM_START[0]], y=[AIRSIM_START[1]], z=[AIRSIM_START[2]],
        mode="markers", marker=dict(size=10, color="blue"), name="Start (0,0)"
    ))
    fig.add_trace(go.Scatter3d(
        x=[AIRSIM_GOAL[0]], y=[AIRSIM_GOAL[1]], z=[AIRSIM_GOAL[2]],
        mode="markers", marker=dict(size=10, color="red"), name="Goal"
    ))
    fig.add_trace(go.Scatter3d(
        x=[top_pos[0]], y=[top_pos[1]], z=[top_pos[2]],
        mode="markers", marker=dict(size=10, color="gold", symbol="diamond"), name="Peak"
    ))

    fig.update_layout(
        title="AirSim GT (Rotated: +X_ds -> +Y_as, +Y_ds -> -X_as)",
        scene=dict(
            aspectmode="data",
            xaxis_title="AirSim X (Right)",
            yaxis_title="AirSim Y (Down)",
            zaxis_title="Z (Relative Elevation)",
            yaxis=dict(autorange="reversed")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output)
    print(f"Saved rotated verification plot to {args.output}")
    print(f"View at: http://127.0.0.1:8000/{args.output}")

if __name__ == "__main__":
    main()
