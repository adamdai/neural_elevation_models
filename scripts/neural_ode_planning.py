import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import MLP

from nemo.siren import Siren
from nemo.global_planner import GlobalPlanner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    # Load the siren elevation model
    siren = Siren(in_features=2, out_features=1, hidden_features=256,
                    hidden_layers=3, outermost_linear=True).to(device)

    siren.load_state_dict(torch.load('../models/mt_bruno_siren.pt'))
    siren.eval()

    # Planning parameters
    x_0 = torch.tensor([-1.0, -1.0], device=device)
    x_f = torch.tensor([1.0, 1.0], device=device)
    T = torch.linspace(0, 1, 100, device=device)[:,None]

    mse = nn.MSELoss()

    # Neural dynamics
    dyn = MLP(in_channels=1, hidden_channels=[256, 256, 256, 2]).to(device)
    optimizer = torch.optim.Adam(dyn.parameters(), lr=1e-4)

    
    for i in range(1000):
        dx = dyn(T)
        path = torch.cumsum(dx, dim=0) + x_0
        goal_loss = 1e2 * mse(path[-1], x_f)
        dist_loss = torch.norm(dx, dim=1).nanmean()
        
        # heights = siren(path)
        # cost_loss = heights.mean()
        heights, grad = siren.forward_with_grad(path)
        cost_loss = 100 * torch.abs(torch.sum(dx * grad, axis=1)).mean()

        loss = goal_loss + dist_loss + cost_loss 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Loss: {loss.item()}')