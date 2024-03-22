import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from nemo.global_planner import GlobalPlanner2
from nemo.nemo import Nemo
from nemo.util import wrap_angle_torch, path_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SCENE_NAME = 'KT22'  # 'KT22' or 'RedRocks'


if __name__ == "__main__":

    # Load the Nemo model (automatically sends to device)
    if SCENE_NAME == 'KT22':
        nemo = Nemo('../models/kt22_encs.pth', '../models/kt22_heightnet.pth')
    elif SCENE_NAME == 'RedRocks':
        nemo = Nemo('../models/redrocks_encs_relu.pth', '../models/redrocks_heightnet_relu.pth')
    
    # Form grid of points
    N = 64
    if SCENE_NAME == 'KT22':
        bounds = (-0.75, 0.75, -0.75, 0.75) # kt22
    elif SCENE_NAME == 'RedRocks':
        bounds = (-0.4, 0.8, -0.6, 0.6) # red rocks
    xs = torch.linspace(bounds[0], bounds[1], N, device=device)
    ys = torch.linspace(bounds[2], bounds[3], N, device=device)
    XY_grid = torch.meshgrid(xs, ys, indexing='xy')
    XY_grid = torch.stack(XY_grid, dim=-1)
    positions = XY_grid.reshape(-1, 2)

    # Query heights
    heights = nemo.get_heights(positions)
    
    # Numpy grid values for plotting
    z_grid = heights.reshape(N, N).detach().cpu().numpy()
    x_grid = XY_grid[:,:,0].detach().cpu().numpy()
    y_grid = XY_grid[:,:,1].detach().cpu().numpy()