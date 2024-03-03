import numpy as np
import plotly.graph_objects as go

import torch

from siren import Siren

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xs = torch.linspace(-1, 1, steps=1000, device=device)
ys = torch.linspace(-1, 1, steps=1000, device=device)
x, y = torch.meshgrid(xs, ys, indexing='xy')
xy = torch.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

siren = Siren(in_features=2, 
              out_features=1, 
              hidden_features=256,
              hidden_layers=3, 
              outermost_linear=True,
              first_omega_0=30.0,
              hidden_omega_0=100.0).to(device)

siren.load_state_dict(torch.load('models/lunar_dem_siren.pth'))
siren.eval()

with torch.no_grad():
    pred, coords = siren(xy)

# Plot the predictions
fig = go.Figure(data=[go.Surface(z=pred.cpu().numpy(), x=x.cpu().numpy(), y=y.cpu().numpy())])
fig.update_layout(width=1200, height=700, scene_aspectmode='data')
fig.show()