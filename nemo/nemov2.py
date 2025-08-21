import torch
import torch.nn as nn
import tinycudann as tcnn
from typing import Dict, List, Optional, Union
from rich import print
from tqdm import tqdm
import gc
import time

from nemo.logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NEMoV2(nn.Module):
    """
    NEMoV2: Neural Elevation Model for fitting to existing DEM data.

    This class is specifically designed for fitting TCNN-based neural networks
    to Digital Elevation Model (DEM) data with improved numerical stability.

    Key features:
    - Automatic input/output scaling for numerical stability
    - Gradient clipping to prevent explosion
    - Learning rate scheduling
    - Early stopping
    - Comprehensive regularization options
    """

    def __init__(
        self,
        input_bounds: Optional[tuple] = None,  # (x_min, x_max, y_min, y_max)
        output_bounds: Optional[tuple] = None,  # (z_min, z_max)
        encoding_config: Optional[Dict[str, Union[str, int, float]]] = None,
        network_config: Optional[Dict[str, Union[str, int, float]]] = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device

        # Store bounds for scaling
        self.input_bounds = input_bounds
        self.output_bounds = output_bounds

        # Default encoding configuration (multiresolution hash grid)
        if encoding_config is None:
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": 6,  # Moderate complexity
                "n_features_per_level": 2,  # Keep small for stability
                "log2_hashmap_size": 18,  # Moderate size
                "base_resolution": 16,  # Good base resolution
                "per_level_scale": 1.5,  # Moderate scaling
                "interpolation": "Linear",  # More stable than Smoothstep
            }

        # Ensure n_features_per_level is valid for TCNN
        if encoding_config.get("n_features_per_level") not in [1, 2, 4, 8]:
            print(
                f"Warning: n_features_per_level={encoding_config.get('n_features_per_level')} is not valid for TCNN."
            )
            print("Setting to 8 (closest valid value).")
            encoding_config["n_features_per_level"] = 8

        # Default network configuration
        if network_config is None:
            network_config = {
                "otype": "FullyFusedMLP",  # More stable than CutlassMLP
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,  # Smaller for half precision stability
                "n_hidden_layers": 3,  # Moderate depth
            }

        # Create encoding
        self.encoding = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config)

        # Create height network
        tot_out_dims = self.encoding.n_output_dims
        self.height_net = tcnn.Network(
            n_input_dims=tot_out_dims, n_output_dims=1, network_config=network_config
        )

        # Move to device and use default dtype (half precision for TCNN)
        self.encoding.to(device)
        self.height_net.to(device)

        # Scaling parameters (will be computed during fit)
        self.input_scale = None
        self.input_offset = None
        self.output_scale = None
        self.output_offset = None

        # Training history
        self.training_history = {"losses": [], "best_loss": float("inf"), "best_epoch": 0}

    def compute_scaling_parameters(self, xy: torch.Tensor, z: torch.Tensor):
        """Compute input and output scaling parameters for numerical stability."""

        # Input scaling (scale to [-1, 1] like original NEMo)
        xy_np = xy.detach().cpu().numpy()
        x_min, x_max = xy_np[:, 0].min(), xy_np[:, 0].max()
        y_min, y_max = xy_np[:, 1].min(), xy_np[:, 1].max()

        # Scale to [-1, 1] range
        x_center = (x_max + x_min) / 2.0
        y_center = (y_max + y_min) / 2.0
        x_range = (x_max - x_min) / 2.0
        y_range = (y_max - y_min) / 2.0

        self.input_scale = torch.tensor([x_range, y_range], device=self.device, dtype=torch.float16)
        self.input_offset = torch.tensor(
            [x_center, y_center], device=self.device, dtype=torch.float16
        )

        # No output scaling - keep z in original range like original NEMo
        self.output_scale = 1.0
        self.output_offset = 0.0

    def normalize_input(self, xy: torch.Tensor) -> torch.Tensor:
        """Normalize input coordinates to [-1, 1] range like original NEMo."""
        if self.input_scale is None:
            raise ValueError("Input scaling not computed. Call fit() first.")

        xy = xy.to(self.device)
        xy_norm = (xy - self.input_offset) / self.input_scale  # [-1, 1]
        xy_norm = torch.clamp(xy_norm, -1.0, 1.0)
        return xy_norm

    def denormalize_output(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize output back to original scale."""
        if self.output_scale is None:
            return z_norm

        return z_norm * self.output_scale + self.output_offset

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict heights from normalized coordinates."""
        # Normalize input if scaling is available
        if self.input_scale is not None:
            xy_norm = self.normalize_input(xy)
        else:
            xy_norm = xy

        encodings = self.encoding(xy_norm)
        heights_norm = self.height_net(encodings)

        # Denormalize output if scaling is available
        if self.output_scale is not None:
            heights = self.denormalize_output(heights_norm)
        else:
            heights = heights_norm

        return heights.squeeze(-1)

    def fit(
        self,
        xy: torch.Tensor,
        z: torch.Tensor,
        lr: float = 1e-3,
        max_epochs: int = 1000,
        batch_size: int = 10000,
        grad_clip: float = 1.0,
        patience: int = 100,
        verbose: bool = False,
        early_stopping: bool = True,
        logger: Optional[Logger] = None,
    ) -> List[float]:
        """Fit the model to the given data.

        Args:
            xy: Input coordinates (N, 2)
            z: Target heights (N,) or (N, 1)
            lr: Learning rate
            max_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            grad_clip: Gradient clipping value
            grad_weight: Weight for spatial smoothness regularization
            laplacian_weight: Weight for height variance regularization
            patience: Patience for early stopping
            verbose: Whether to print training progress
            early_stopping: Whether to use early stopping
            spatial_method: Method for spatial regularization ("finite_diff", "gradients", "disabled")
            enable_spatial: Whether to enable spatial regularization
            logger: Optional NEMoLogger instance for wandb logging

        Returns:
            List of training losses
        """
        # Move data to device
        xy = xy.to(self.device)
        z = z.to(self.device)

        # Compute scaling parameters
        self.compute_scaling_parameters(xy, z)

        # Convert to half precision
        xy = xy.half()
        z = z.half()

        # Setup optimizer with separate learning rates
        optimizer = torch.optim.Adam(
            [
                {"params": self.encoding.parameters(), "lr": lr},
                {"params": self.height_net.parameters(), "lr": lr},
            ]
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=500
        )
        criterion = nn.MSELoss()

        # Training variables
        losses = []
        best_loss = float("inf")
        patience_counter = 0

        # Initialize logger if provided
        if logger is not None:
            logger.start_training(max_epochs, xy.shape[0], batch_size)
            logger.log_model_parameters(self)

        # Training loop with tqdm progress bar (always active)
        pbar = tqdm(range(max_epochs), desc="Training", unit="epoch")
        for epoch in pbar:
            # Start epoch logging
            if logger is not None:
                logger.start_epoch(epoch)
            # Create batches
            if batch_size < xy.shape[0]:
                indices = torch.randperm(xy.shape[0])
                xy_batch = xy[indices[:batch_size]]
                z_batch = z[indices[:batch_size]]
            else:
                xy_batch = xy
                z_batch = z

            # Forward pass
            pred = self.forward(xy_batch)
            height_loss = criterion(pred, z_batch)

            losses = {
                "height_loss": height_loss,
                "total_loss": height_loss,
            }
            total_loss = losses["total_loss"]

            # Log metrics with wandb if logger is provided
            if logger is not None:
                from .logger import TrainingMetrics

                # Calculate epoch time
                epoch_time = (
                    time.time() - logger.epoch_start_time if logger.epoch_start_time else None
                )

                # Calculate memory usage if available
                memory_usage = None
                if torch.cuda.is_available():
                    memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB

                # Create training metrics
                training_metrics = TrainingMetrics(
                    epoch=epoch,
                    losses=losses,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    training_time=epoch_time,
                    memory_usage=memory_usage,
                )

                logger.log_training_metrics(training_metrics)

                # Log gradient norms if enabled
                if logger.log_gradients:
                    logger.log_gradient_norms(self, epoch)

                # Log memory usage if enabled
                if logger.log_memory:
                    logger.log_memory_usage(epoch)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoding.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.height_net.parameters(), grad_clip)

            optimizer.step()

            # Update learning rate
            scheduler.step(total_loss)

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{total_loss.item():.6f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            )

            # Early stopping
            if early_stopping:
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                    self.training_history["best_loss"] = best_loss
                    self.training_history["best_epoch"] = epoch
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if logger is not None:
                        logger.log_early_stopping(epoch, patience)
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            # Memory cleanup every 100 epochs to prevent CUDA OOM
            if epoch % 100 == 0 and epoch > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

        # Close progress bar
        pbar.close()

        # Log training completion with wandb if logger is provided
        if logger is not None:
            final_loss = (
                total_loss.item()
                if "total_loss" in locals()
                else self.training_history.get("final_loss", float("inf"))
            )
            logger.log_training_completion(final_loss, len(losses))

        # Compute final loss over all data
        with torch.no_grad():
            final_pred = self.forward(xy)
            final_loss = criterion(final_pred, z).item()
            print(f"Total loss over all data: {final_loss:.6f}")

    def save_model(self, filepath: str, logger=None):
        """Save the trained model."""
        torch.save(
            {
                "encoding_state_dict": self.encoding.state_dict(),
                "height_net_state_dict": self.height_net.state_dict(),
                "input_scale": self.input_scale,
                "input_offset": self.input_offset,
                "output_scale": self.output_scale,
                "output_offset": self.output_offset,
                "training_history": self.training_history,
            },
            filepath,
        )

        # Log model saving with wandb if logger is provided
        if logger is not None:
            import os

            model_size_mb = (
                os.path.getsize(filepath) / 1024**2 if os.path.exists(filepath) else None
            )
            logger.log_model_save(filepath, model_size_mb)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str, logger=None):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.encoding.load_state_dict(checkpoint["encoding_state_dict"])
        self.height_net.load_state_dict(checkpoint["height_net_state_dict"])
        self.input_scale = checkpoint["input_scale"]
        self.input_offset = checkpoint["input_offset"]
        self.output_scale = checkpoint["output_scale"]
        self.output_offset = checkpoint["output_offset"]
        self.training_history = checkpoint["training_history"]

        # Log model loading with wandb if logger is provided
        if logger is not None:
            logger.log_model_load(filepath)

        print(f"Model loaded from {filepath}")

    def __repr__(self):
        return (
            f"NEMoV2(encoding={self.encoding.__class__.__name__}, "
            f"network={self.height_net.__class__.__name__}, "
            f"device={self.device})"
        )
