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
        elif encoding_type == "sinusoidal":
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

    def fit(self, xyz: torch.Tensor, tol: float = 1e-6, max_iters: int = 10000):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        xy = xyz.cuda()[:, :2].float()
        z_train = xyz.cuda()[:, 2].float()
        prev_loss = float("inf")
        for i in range(max_iters):
            z_pred = self.forward(xy)
            loss = torch.nn.functional.mse_loss(z_pred, z_train)

            # Check for convergence
            if abs(prev_loss - loss.item()) < tol:
                print(f"Converged at iteration {i} with loss: {loss.item()}")
                break

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prev_loss = loss.item()
