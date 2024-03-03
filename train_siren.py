"""Train SIREN from elevation data.

"""

import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

import torch
import torch.nn as nn

from siren import Siren

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Load data
    Z = np.load('data/lunar_dem.npy').astype(np.float32)

    # Subsample
    Z = Z[:1000, :1000]
    
    # Normalize data
    Z_normalized = 1 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z)) 
    z = torch.tensor(Z_normalized, dtype=torch.float32, device=device).view(-1, 1)
    print(torch.min(z), torch.max(z))
    
    # XY points
    EXTENT = 100
    xs = torch.linspace(-EXTENT, EXTENT, steps=Z.shape[0], device=device)
    ys = torch.linspace(-EXTENT, EXTENT, steps=Z.shape[1], device=device)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    xy = torch.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    # Create model
    model = Siren(in_features=2, 
                  out_features=1, 
                  hidden_features=256, 
                  hidden_layers=3, 
                  outermost_linear=True,
                  first_omega_0=30.0,
                  hidden_omega_0=30.0).to(device)

    print(xy.size(), z.size())

    # Dataloader
    dataset = torch.utils.data.TensorDataset(xy, z)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    # for epoch in range(10):
    #     for xy, z in tqdm(dataloader):
    #         xy, z = xy.to(device), z.to(device)

    #         # Forward pass
    #         pred, coords = model(xy)

    #         # Compute loss
    #         loss = criterion(pred, z)

    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     print(f"Epoch {epoch}, Loss {loss.item()}")

    for epoch in range(500):

        # Forward pass
        pred, coords = model(xy)

        # Compute loss
        loss = criterion(pred, z)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss {loss.item()}")

    # Save weights
    torch.save(model.state_dict(), 'models/lunar_dem_siren.pth')

    # Plot
    with torch.no_grad():
        pred, coords = model(xy)

    # Plot the predictions
    fig = go.Figure(data=[go.Surface(z=pred.cpu().numpy().reshape(Z.shape), x=x.cpu().numpy(), y=y.cpu().numpy())])
    fig.update_layout(width=1200, height=700, scene_aspectmode='data')
    fig.show()