"""
Given AirSim starting location, retrieve DEM patch from heightmap in AirSim coordinates

"""

import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from nemo.plotting import plot_3d_points, plot_surface

# %% User-specified parameters

heightmap_path = '../data/unreal_moon_heightmap.png'
heightmap_resolution = 1.0  # meters per pixel
PATCH_WIDTH_M = 2000.0  # meters
HEIGHT_SCALE = 257.0  # divide heightmap values by this to convert to meters (AirSim)

# Patch center in AirSim coordinates
patch_center = np.array([576.22075927, -1656.53742188])   # moon_spiral_2 spiral center
# patch_center = np.array([52, -1825])   # moon_spiral_2 rover start

save_path = '../data/dem_pc.ply'

PLOT = False

# %% Get patch extent

image = Image.open(heightmap_path)
heightmap = np.array(image)

HEIGHTMAP_CENTER = np.array(heightmap.shape)/2
center_idx = patch_center + HEIGHTMAP_CENTER

patch_min = np.array([center_idx[0]-PATCH_WIDTH_M//2, center_idx[1]-PATCH_WIDTH_M//2], dtype=int)
patch_max = np.array([center_idx[0]+PATCH_WIDTH_M//2, center_idx[1]+PATCH_WIDTH_M//2], dtype=int)
print(f"patch_min: {patch_min}")
print(f"patch_max: {patch_max}")
patch = heightmap[patch_min[1]:patch_max[1], patch_min[0]:patch_max[0]]

# Plot the full heightmap with a box around the patch
if PLOT:
    plt.imshow(heightmap, cmap='gray')
    plt.plot([patch_min[0], patch_max[0], patch_max[0], patch_min[0], patch_min[0]],
            [patch_min[1], patch_min[1], patch_max[1], patch_max[1], patch_min[1]], color='red')
    plt.show()

# %% Scale heights

x = np.linspace(-PATCH_WIDTH_M/2, PATCH_WIDTH_M/2, patch.shape[1])
y = np.linspace(-PATCH_WIDTH_M/2, PATCH_WIDTH_M/2, patch.shape[0])
X, Y = np.meshgrid(x, y)

patch_adj = patch / HEIGHT_SCALE

# Set height=0 at center of patch
starting_z = patch_adj[patch.shape[1]//2, patch.shape[0]//2]
patch_adj = patch_adj - starting_z 

# Rotate patch so that x-axis is aligned rover forward direction
patch_adj = np.rot90(patch_adj)


# %% Validate by plotting positions from transforms.json poses

transforms_path = '../../nerfstudio/data/moon_spiral_2/transforms.json'
with open(transforms_path, 'r') as f:
    transforms = json.load(f)

positions = []
for frame in transforms['frames']:
    T = np.array(frame['transform_matrix'])
    R, t = T[:3, :3], T[:3, 3]
    positions.append(t)

positions = np.array(positions)

if PLOT:
    fig = go.Figure()
    plot_surface(x=X, y=-Y, z=patch_adj, fig=fig)
    plot_3d_points(x=positions[:,0], y=-positions[:,1], z=positions[:,2], fig=fig, color='red')
    fig.show()


# %% Save patch as point cloud

from nemo.util import generate_ply_from_dem

dem_points = np.stack([X.flatten(), -Y.flatten(), patch_adj.flatten()], axis=1)
colors = 128 * np.ones_like(dem_points)

generate_ply_from_dem(dem_points, colors, save_path)
np.save(save_path.replace('.ply', '.npy'), dem_points)

print(f"Saved DEM patch point cloud to {save_path}")