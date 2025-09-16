import math
import numpy as np
import torch
import gpytorch
import os
from datetime import datetime
from pathlib import Path
from nemo.dem import DEM


# ----------------------------
# 0) Helpers & synthetic DEM
# ----------------------------
def make_synthetic_dem(nx=512, ny=512, seed=0):
    """
    Make a smooth terrain-like function on a regular grid.
    Returns x(1D), y(1D), Z(2D).
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 10_000.0, nx)  # meters
    y = np.linspace(0.0, 10_000.0, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Sum of smooth radial bumps + gentle slope
    Z = (
        80 * np.exp(-((X - 2500) ** 2 + (Y - 3000) ** 2) / (2 * (1200**2)))
        + 40 * np.exp(-((X - 7000) ** 2 + (Y - 7000) ** 2) / (2 * (900**2)))
        + 0.002 * X
        + 0.003 * Y
    ).astype(np.float32)

    # tiny noise
    Z += rng.normal(0, 0.2, size=Z.shape).astype(np.float32)
    return x.astype(np.float32), y.astype(np.float32), Z


def prepare_data():
    tif_path = Path("data/Site01_final_adj_5mpp_surf.tif")
    dem = DEM.from_file(tif_path)
    XYZ = dem.data
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2]
    x = X[0, :]
    y = Y[:, 0]
    return x, y, Z


def subsample_training(x, y, Z, max_train=20000, seed=0):
    """
    Uniform random subsample from a regular grid to keep GP training tractable.
    Returns train_x (N,2), train_y (N,)
    """
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    XY = np.column_stack([Xg.ravel(), Yg.ravel()]).astype(np.float32)
    z = Z.ravel().astype(np.float32)

    n = XY.shape[0]
    if n > max_train:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_train, replace=False)
        XY = XY[idx]
        z = z[idx]
    return torch.from_numpy(XY), torch.from_numpy(z)


# --------------------------------
# 1) SKI/KISS-GP model definition
# --------------------------------
class SKIExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid_bounds, grid_size=(128, 128), nu=1.5):
        """
        grid_bounds: [(x_min,x_max), (y_min,y_max)]
        grid_size: number of inducing points per dimension (e.g., (128,128) or (256,256))
        nu: Matern smoothness (1.5 or 2.5 are common)
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        base_kernel = gpytorch.kernels.MaternKernel(nu=nu)
        base_kernel.lengthscale = 0.05 * math.hypot(
            grid_bounds[0][1] - grid_bounds[0][0], grid_bounds[1][1] - grid_bounds[1][0]
        )

        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(base_kernel),
            grid_size=grid_size,
            num_dims=2,
            grid_bounds=torch.tensor(grid_bounds, dtype=train_x.dtype),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ----------------------------
# 2) Train the SKI GP
# ----------------------------
def train_ski_gp(
    train_x, train_y, grid_bounds, grid_size=(128, 128), iters=300, lr=0.1, use_gpu=True
):
    device = (
        torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
    )
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SKIExactGP(
        train_x, train_y, likelihood, grid_bounds=grid_bounds, grid_size=grid_size
    ).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(1, iters + 1):
        optimizer.zero_grad()
        with gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True):
            output = model(train_x)
            loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if (i % 50) == 0 or i == 1:
            print(
                f"Iter {i:4d}/{iters} | loss {loss.item():.4f} | "
                f"noise {likelihood.noise.item():.4f} | ls {model.covar_module.base_kernel.base_kernel.lengthscale.item():.2f}"
            )
    return model.eval(), likelihood.eval(), device


# ----------------------------
# 3) Chunked grid prediction
# ----------------------------
@torch.no_grad()
def predict_grid_chunked(
    model, likelihood, x, y, rows_per_chunk=256, device=None, return_var=False
):
    """
    Predict mean (and optionally variance) on a full grid, in row chunks.
    x, y: 1D numpy arrays (float32 recommended). Returns numpy arrays.
    """
    if device is None:
        device = next(model.parameters()).device
    Nx, Ny = len(x), len(y)
    Z_mean = np.empty((Ny, Nx), dtype=np.float32)
    Z_var = np.empty_like(Z_mean) if return_var else None

    x_t = torch.from_numpy(x).to(device)
    for y0 in range(0, Ny, rows_per_chunk):
        y1 = min(y0 + rows_per_chunk, Ny)
        Xg, Yg = np.meshgrid(x, y[y0:y1], indexing="xy")
        XY = np.column_stack([Xg.ravel(), Yg.ravel()]).astype(np.float32)
        Xq = torch.from_numpy(XY).to(device)

        with (
            gpytorch.settings.fast_pred_var(True),
            gpytorch.settings.max_root_decomposition_size(100),
        ):
            pred = likelihood(model(Xq))
            mean = pred.mean.detach().reshape(y1 - y0, Nx).float().cpu().numpy()
            Z_mean[y0:y1, :] = mean
            if return_var:
                var = pred.variance.detach().reshape(y1 - y0, Nx).float().cpu().numpy()
                Z_var[y0:y1, :] = var

    return (Z_mean, Z_var) if return_var else Z_mean


# ----------------------------
# 4) End-to-end demo
# ----------------------------
if __name__ == "__main__":
    # Synthetic DEM (replace with your real x, y, Z)
    # x, y, Z = make_synthetic_dem(nx=512, ny=512, seed=0)
    x, y, Z = prepare_data()
    grid_bounds = [(float(x.min()), float(x.max())), (float(y.min()), float(y.max()))]

    # Subsample training points (e.g., 20k). For big DEMs (3200x3200), you might go 30k-80k.
    train_x, train_y = subsample_training(x, y, Z, max_train=20000, seed=42)

    # Train SKI GP (choose grid_size by memory/accuracy; 256x256 is a strong starting point)
    model, likelihood, device = train_ski_gp(
        train_x, train_y, grid_bounds, grid_size=(128, 128), iters=300, lr=0.1, use_gpu=True
    )

    # Predict on full grid (can be 3200x3200; just adjust rows_per_chunk)
    Z_pred, Z_var = predict_grid_chunked(
        model, likelihood, x, y, rows_per_chunk=128, device=device, return_var=True
    )

    # Example metric against the synthetic ground truth
    rmse = float(np.sqrt(np.mean((Z_pred - Z) ** 2)))
    mae = float(np.mean(np.abs(Z_pred - Z)))
    print(f"RMSE vs ground truth: {rmse:.3f} m")
    print(f"MAE vs ground truth: {mae:.3f} m")
    print(f"Max error: {np.max(np.abs(Z_pred - Z)):.3f} m")

    # Save the trained model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/gpytorch_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save model state dict
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "grid_bounds": grid_bounds,
            "grid_size": (128, 128),
            "training_points": len(train_x),
            "rmse": rmse,
            "timestamp": timestamp,
        },
        os.path.join(results_dir, "model.pth"),
    )

    # Save prediction results
    np.save(os.path.join(results_dir, "prediction.npy"), Z_pred)
    np.save(os.path.join(results_dir, "variance.npy"), Z_var)
    np.save(os.path.join(results_dir, "ground_truth.npy"), Z)
    np.save(os.path.join(results_dir, "x_coords.npy"), x)
    np.save(os.path.join(results_dir, "y_coords.npy"), y)

    print(f"\nModel and results saved to: {results_dir}")
    print(f"Model file: {os.path.join(results_dir, 'model.pth')}")
