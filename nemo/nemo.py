import torch
import torch.nn as nn
import tinycudann as tcnn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nemo.spatial_distortions import SceneContraction
from nemo.util.plotting import plot_surface
from nemo.util import grid_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: should inherit from nn.Module?
class Nemo:
    def __init__(self, encs_pth=None, mlp_pth=None):
        self.encoding = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 8,
                "n_features_per_level": 8,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.2599210739135742,
                "interpolation": "Smoothstep",
            },
        )
        tot_out_dims_2d = self.encoding.n_output_dims

        self.height_net = tcnn.Network(
            n_input_dims=tot_out_dims_2d,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 3,
            },
        )

        self.spatial_distortion = SceneContraction()

        if encs_pth is not None and mlp_pth is not None:
            self.load_weights(encs_pth, mlp_pth)

        # TODO: add xyz scaling
        self.x_scale = None
        self.y_scale = None
        self.z_scale = None

    def load_weights(self, encs_pth, mlp_pth):
        """Load weights for hashgrid encoding and MLP"""
        self.encoding.load_state_dict(torch.load(encs_pth))
        self.encoding.to(device)
        self.height_net.load_state_dict(torch.load(mlp_pth))
        self.height_net.to(device)

    def get_heights(self, positions):
        """Query heights"""
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        encs = self.encoding(positions[:, :2])
        heights = self.height_net(encs)
        return heights

    def get_heights_with_grad(self, positions):
        """Query heights and gradients"""
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        encs = self.encoding(positions[:, :2])
        heights = self.height_net(encs)
        grad = torch.autograd.grad(heights.sum(), positions, create_graph=True)[0]
        return heights, grad[:, :2]

    def fit(self, xy, z, lr=1e-5, iters=5000):
        """Fit to (x,y) and z data

        xy : (N, 2)
        z : (N, 1)

        """
        # Loss function
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.Adam(
            [{"params": self.encoding.parameters()}, {"params": self.height_net.parameters()}],
            lr=lr,
        )

        # Convert the data half precision to match network
        xy = xy.half()
        z = z.half()

        # Train the network
        for step in range(iters):
            # Forward pass
            pred = self.get_heights(xy)

            # Compute loss
            loss = criterion(pred, z)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 500 steps
            if step % 500 == 0:
                print(f"Step {step}, Loss {loss.item()}")

    def plot(self, N=64, bounds=(-1.0, 1.0, -1.0, 1)):
        """Surface plot"""
        xs = torch.linspace(bounds[0], bounds[1], N, device=device)
        ys = torch.linspace(bounds[2], bounds[3], N, device=device)
        XY_grid = torch.meshgrid(xs, ys, indexing="xy")
        XY_grid = torch.stack(XY_grid, dim=-1)
        positions = XY_grid.reshape(-1, 2)
        heights = self.get_heights(positions)

        z_grid = heights.reshape(N, N).detach().cpu().numpy()
        x_grid = XY_grid[:, :, 0].detach().cpu().numpy()
        y_grid = XY_grid[:, :, 1].detach().cpu().numpy()

        fig = plot_surface(x_grid, y_grid, z_grid, showscale=False)
        return fig

    def plot_grads(self, N=64, bounds=(-1.0, 1.0, -1.0, 1), clip=None):
        """Surface plot"""
        # xs = torch.linspace(bounds[0], bounds[1], N, device=device)
        # ys = torch.linspace(bounds[2], bounds[3], N, device=device)
        # XY_grid = torch.meshgrid(xs, ys, indexing='xy')
        # XY_grid = torch.stack(XY_grid, dim=-1)
        # positions = XY_grid.reshape(-1, 2)
        positions, XY_grid = grid_2d(N, bounds)
        positions.requires_grad = True
        _, grad = self.get_heights_with_grad(positions)

        x_grad = grad[:, 0].reshape(N, N).detach().cpu().numpy()
        y_grad = grad[:, 1].reshape(N, N).detach().cpu().numpy()

        if clip:
            x_grad = x_grad.clip(-clip, clip)
            y_grad = y_grad.clip(-clip, clip)

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("X Gradient", "Y Gradient"), horizontal_spacing=0.15
        )
        fig.add_trace(go.Heatmap(z=x_grad, colorbar=dict(len=1.05, x=0.44, y=0.5)), row=1, col=1)
        fig.add_trace(go.Heatmap(z=y_grad, colorbar=dict(len=1.05, x=1.01, y=0.5)), row=1, col=2)
        fig.update_layout(width=1300, height=600, scene_aspectmode="data")
        fig.show()
