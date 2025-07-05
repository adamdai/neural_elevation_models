import torch
import torch.nn as nn
import numpy as np

from nerfstudio.field_components.encodings import RFFEncoding, NeRFEncoding


class NeuralHeightField(nn.Module):
    def __init__(
        self,
        in_dim=2,
        hidden_dim=64,
        n_layers=3,
        out_range=(-2.0, 2.0),
        activation="relu",
        encoding_type="rff",  # 'rff', 'nerf', or 'sinusoidal'
        num_frequencies=10,
        min_freq_exp=0,
        max_freq_exp=8,
        include_input=False,
        scale=10.0,
        device="cpu",
    ):
        super().__init__()
        self.out_range = out_range
        self.device = device

        # Choose encoding 
        if encoding_type == "rff":
            self.encoder = RFFEncoding(in_dim=in_dim, num_frequencies=num_frequencies, scale=scale)
            input_dim = 2 * num_frequencies
        elif encoding_type == "nerf":
            self.encoder = NeRFEncoding(
                in_dim=in_dim,
                num_frequencies=num_frequencies,
                min_freq_exp=min_freq_exp,
                max_freq_exp=max_freq_exp,
                include_input=include_input,
            )
            input_dim = self.encoder.get_out_dim()
        elif encoding_type == "sinusoidal": #should this be removed since it is same as nerf?
            raise NotImplementedError("Use 'rff' or 'nerf' to use Nerfstudio encodings.")
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        self.encoder.to(device)

        # Choose activation
        act_layer = nn.ReLU() if activation == "relu" else nn.Softplus(beta=1.0)

        # MLP layers
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_layer)
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xy):
        xy = 2 * (xy - 0.5)  # Normalize to [-1, 1]
        x_encoded = self.encoder(xy)
        raw_z = self.mlp(x_encoded).squeeze(-1)
        z = torch.tanh(raw_z)
        z = z * (self.out_range[1] - self.out_range[0]) / 2 + sum(self.out_range) / 2
        return z

    def forward_with_grad(self, xy):
        """Forward pass that also returns gradients of height with respect to x,y coordinates.

        Args:
            xy: Input coordinates of shape (..., 2)

        Returns:
            Tuple of:
                z: Height predictions of shape (...)
                grad_xy: Gradients of height with respect to input coordinates, shape (..., 2)
        """
        xy = xy.clone().requires_grad_(True)  # Enable gradient computation
        z = self.forward(xy)
        grad_xy = torch.autograd.grad(z.sum(), xy, create_graph=True)[0]
        return z, grad_xy

    def fit(
        self, xyz: torch.Tensor, tol: float = 1e-6, max_iters: int = 10000, grad_weight: float = 0.1
    ):
        """Fit the neural height field to the given points with gradient regularization.

        Args:
            xyz: Input points of shape (N, 3) where N is number of points
            tol: Convergence tolerance for loss change
            max_iters: Maximum number of iterations
            grad_weight: Weight for gradient regularization loss (higher values = smoother surface)
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        xy = xyz.cuda()[:, :2].float()
        z_train = xyz.cuda()[:, 2].float()
        prev_height_loss = float("inf")

        for i in range(max_iters):
            # Forward pass with gradients
            z_pred, grad_xy = self.forward_with_grad(xy)

            # Compute losses
            losses = {
                "height": torch.nn.functional.mse_loss(z_pred, z_train),
                "grad": torch.mean(torch.sum(grad_xy**2, dim=-1)),
            }

            # Total loss with regularization
            losses["total"] = losses["height"] + grad_weight * losses["grad"]

            # Check for convergence on height loss
            if abs(prev_height_loss - losses["height"].item()) < tol:
                print(f"Converged at iteration {i}")
                print(
                    f"Height Loss: {losses['height'].item():.6f}, "
                    f"Gradient Loss: {losses['grad'].item():.6f}"
                )
                break

            if i % 100 == 0:
                print(
                    f"Iteration {i} | "
                    f"Height Loss: {losses['height'].item():.6f}, "
                    f"Gradient Loss: {losses['grad'].item():.6f}"
                )

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            prev_height_loss = losses["height"].item()
