import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

from nemo.global_planner import AStarGradPlanner
from nemo.nemo import Nemo
from nemo.util import path_metrics, grid_2d
from nemo.planning import path_optimization
from nemo.plotting import plot_surface, plot_path_3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%========================= -- Parameters -- =========================%%#

SCENE_NAME = 'AirSimMountains'  # 'KT22', 'RedRocks', 'UnrealMoon', 'AirSimMountains'

N_GRID = 64  # Grid resolution for A*
N_PLOT = 256  # Grid resolution for plotting
HEIGHT_SCALE = 1e3  # Height scaling factor for A*

AIRSIM = True if SCENE_NAME == 'AirSimMountains' or 'UnrealMoon' else False

if AIRSIM:
    # Get nerf dataparser transform
    dataparser_transforms = json.load(open(f'../models/{SCENE_NAME}/dataparser_transforms.json'))
    transform = np.array(dataparser_transforms['transform'])
    scale = dataparser_transforms['scale']

    # Specify start and end in AirSim coordinates
    airsim_start = np.array([177., -247., -33.])  
    airsim_end = airsim_start + np.array([-192., -328., -68.])  
    center = np.array([99., -449., -57.])

    # Convert to scene coordinates (TODO)
    scene_start = (0.0, 0.0)
    scene_end = (-0.2, -0.2)
else:
    # Specify start and end in scene coordinates
    scene_start = (0.7, 0.7)
    scene_end = (-0.7, -0.7)


print(f"Running Nemo planning for {SCENE_NAME}:\n")


if __name__ == "__main__":
    #%%========================= -- Load Nemo model -- =========================%%#
    
    print("Loading Nemo model...\n")

    # Load the Nemo model (automatically sends to device)
    nemo = Nemo(f'../models/{SCENE_NAME}/encs.pth', f'../models/{SCENE_NAME}/mlp.pth')
    
    # Manual cropping
    if SCENE_NAME == 'KT22':
        bounds = (-0.75, 0.75, -0.75, 0.75) 
    elif SCENE_NAME == 'RedRocks':
        bounds = (-0.4, 0.8, -0.6, 0.6)
    elif SCENE_NAME == 'AirSimMountains':
        bounds = (-0.75, 0.45, -0.6, 0.6)
    elif SCENE_NAME == 'UnrealMoon':
        bounds = (-1., 1., -1., 1.)

    #%%========================= -- A* Initialization -- =========================%%#

    print("Running A* initialization...\n")

    # Form a grid of positions
    positions, XY_grid = grid_2d(N_GRID, bounds)
    # Query heights
    heights = nemo.get_heights(positions)
    z_grid = heights.reshape(N_GRID, N_GRID).detach().cpu().numpy()

    # Initialize the planner with scaled heightmap (add 1.0 to heights to make them all positive)
    scaled_heights = HEIGHT_SCALE * (z_grid + 1.0).reshape(N_GRID, N_GRID)
    astar = AStarGradPlanner(scaled_heights, bounds)

    # Compute path
    path_xy = astar.spatial_plan(scene_start, scene_end)
    path_xy_torch = torch.tensor(path_xy, device=device)
    # Get heights along path
    path_zs = nemo.get_heights(path_xy_torch)  

    # Save path as torch tensor
    astar_path = torch.cat((path_xy_torch, path_zs), dim=1)

    #%%========================= -- Path Optimization -- =========================%%#

    print("Running path optimization...\n")

    path_3d = path_optimization(nemo, path_xy_torch, iterations=500, lr=1e-3)

    #%%========================= -- Plotting and Results -- =========================%%#

    print("\nPlotting results...")

    # Resample heights at higher resolution for plotting
    positions, XY_grid = grid_2d(N_PLOT, bounds)
    heights = nemo.get_heights(positions)
    z_grid = heights.reshape(N_PLOT, N_PLOT).detach().cpu().numpy()
    x_grid = XY_grid[:,:,0].detach().cpu().numpy()
    y_grid = XY_grid[:,:,1].detach().cpu().numpy()

    # Plot height field and paths
    fig = plot_surface(x_grid, y_grid, z_grid, no_axes=False, showscale=True)
    fig = plot_path_3d(x=path_xy[:,0], y=path_xy[:,1], z=path_zs.detach().cpu().numpy().flatten(), 
                       fig=fig, color='red', linewidth=5, markers=True, name='A* Path')
    fig = plot_path_3d(x=path_3d[:,0].detach().cpu().numpy(), 
                       y=path_3d[:,1].detach().cpu().numpy(), 
                       z=path_3d[:,2].detach().cpu().numpy(), 
                       fig=fig, color='orange', linewidth=5, markers=False, name='Optimized Path')
    fig.update_layout(width=1600, height=900)
    fig.show()

    print("\nPath metrics: ")
    print("A*:")
    path_metrics(astar_path)
    print("Optimized:")
    path_metrics(path_3d)
