#%%
import torch
import numpy as np
import plotly.graph_objects as go

from nemo.nemo import Nemo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Initialization

nemo = Nemo()
nemo.load_weights('../models/AirSimMountains/AirSimMountains_encs.pth', '../models/AirSimMountains/AirSimMountains_mlp.pth')

start = (0.404, 0.156)
end = (-0.252, -0.228)

N_path = 100
init_xy = torch.stack((torch.linspace(start[0], end[0], N_path, device=device), 
                       torch.linspace(start[1], end[1], N_path, device=device))).T

path_start = init_xy[0]
path_end = init_xy[-1]
path_opt = init_xy[1:-1].clone().detach().requires_grad_(True)  # portion of the path to optimize
path = torch.cat((path_start[None], path_opt, path_end[None]), dim=0)  # full path

xy = path
dt = 1.0

x = xy[:, 0]
y = xy[:, 1]

#%% Derivatives

xdot = torch.diff(x) / dt
ydot = torch.diff(y) / dt
xddot = torch.diff(xdot) / dt
yddot = torch.diff(ydot) / dt
v = torch.sqrt(xdot**2 + ydot**2)
theta = torch.arctan2(ydot, xdot)

#%% Compute pitch angles

l = 0.008  # vehicle length 
dl = l/2 * torch.stack((torch.cos(theta), torch.sin(theta))).T 
z_front = nemo.get_heights(xy[1:] + dl).float()
z_back = nemo.get_heights(xy[1:] - dl).float()
phi = torch.arctan2(z_front - z_back, torch.tensor(l))

#%% Effective gravity

g_eff = 9.81 * torch.sin(phi)   # when 9.81 is multipied, gradients become nan
#g_eff = torch.sin(phi)         # without multiplying by 9.81, gradients are fine

# Formulate cost and compute gradients
c = torch.sum(g_eff)
c.backward()

# Debugging: Check gradients
print("path_opt.grad:", path_opt.grad)