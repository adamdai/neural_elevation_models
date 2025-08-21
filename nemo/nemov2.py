import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from rich import print
from tqdm import tqdm
import wandb
import gc
import time

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

    def compute_scaling_parameters(self, xy: torch.Tensor, z: torch.Tensor, verbose: bool = False):
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

        if verbose:
            print(
                f"[green]Input scaling:[/green] scale={self.input_scale}, offset={self.input_offset}"
            )
            print(
                f"[green]Output scaling:[/green] scale={self.output_scale:.6f}, offset={self.output_offset:.6f}"
            )

    def normalize_input(self, xy: torch.Tensor) -> torch.Tensor:
        """Normalize input coordinates to [-1, 1] range like original NEMo."""
        if self.input_scale is None:
            raise ValueError("Input scaling not computed. Call fit() first.")

        # Ensure input is on the correct device
        xy = xy.to(self.device)

        # Normalize to [-1, 1]
        xy_norm = (xy - self.input_offset) / self.input_scale

        # Clamp to [-1, 1] to prevent out-of-bounds access
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

        # Encode coordinates
        encodings = self.encoding(xy_norm)

        # Predict heights
        heights_norm = self.height_net(encodings)

        # Denormalize output if scaling is available
        if self.output_scale is not None:
            heights = self.denormalize_output(heights_norm)
        else:
            heights = heights_norm

        return heights.squeeze(-1)

    def compute_regularization_loss(
        self,
        xy: torch.Tensor,
        heights: torch.Tensor,
        spatial_method: str = "finite_diff",
        enable_spatial: bool = True,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute regularization losses for smoothness.

        Args:
            xy: Input coordinates (N, 2)
            heights: Predicted heights (N,)
            spatial_method: "finite_diff", "gradients", or "disabled"
            enable_spatial: Whether to enable spatial smoothness regularization
        """
        losses = {}

        # Spatial smoothness regularization
        if enable_spatial and xy.shape[0] > 1:
            try:
                if spatial_method == "finite_diff":
                    # Method 1: Finite differences over x and y individually (grid-based)
                    losses["spatial_smoothness"] = self._compute_finite_diff_smoothness(xy, heights)
                elif spatial_method == "gradients":
                    # Method 2: Gradient-based smoothness (requires gradients)
                    losses["spatial_smoothness"] = self._compute_gradient_smoothness(
                        xy, heights, verbose=verbose
                    )
                elif spatial_method == "disabled":
                    # Method 3: Disabled
                    losses["spatial_smoothness"] = torch.tensor(0.0, device=xy.device)
                    if verbose:
                        print("    [yellow]Spatial smoothness regularization disabled[/yellow]")
                else:
                    if verbose:
                        print(
                            f"[red]Unknown spatial_method: {spatial_method}, using finite_diff[/red]"
                        )
                    losses["spatial_smoothness"] = self._compute_finite_diff_smoothness(
                        xy, heights, verbose=verbose
                    )

            except Exception as e:
                if verbose:
                    print(
                        f"[red]Warning: Could not compute spatial smoothness regularization: {e}[/red]"
                    )
                losses["spatial_smoothness"] = torch.tensor(0.0, device=xy.device)
        else:
            losses["spatial_smoothness"] = torch.tensor(0.0, device=xy.device)

        # Height variance regularization: penalize excessive height variation
        try:
            height_variance = torch.var(heights)
            losses["height_variance"] = height_variance

            # Debug height variance
            if verbose:
                print(f"    [cyan]DEBUG height_variance:[/cyan]")
                print(f"      heights range: [{heights.min():.6f}, {heights.max():.6f}]")
                print(f"      height_variance: {height_variance.item():.6f}")

        except Exception as e:
            if verbose:
                print(f"[red]Warning: Could not compute height variance regularization: {e}[/red]")
            losses["height_variance"] = torch.tensor(0.0, device=xy.device)

        return losses

    def _compute_finite_diff_smoothness(
        self, xy: torch.Tensor, heights: torch.Tensor, verbose: bool = False
    ) -> torch.Tensor:
        """Compute spatial smoothness using finite differences over x and y individually."""
        if verbose:
            print(f"    [cyan]DEBUG spatial_smoothness (finite_diff):[/cyan]")

        # Try to detect grid structure
        unique_x = torch.unique(xy[:, 0])
        unique_y = torch.unique(xy[:, 1])

        if verbose:
            print(
                f"      unique_x: {len(unique_x)} values, range: [{unique_x.min():.6f}, {unique_x.max():.6f}]"
            )
            print(
                f"      unique_y: {len(unique_y)} values, range: [{unique_y.min():.6f}, {unique_y.max():.6f}]"
            )

        if len(unique_x) > 1 and len(unique_y) > 1:
            # Grid-like structure detected
            x_step = (unique_x.max() - unique_x.min()) / (len(unique_x) - 1)
            y_step = (unique_y.max() - unique_y.min()) / (len(unique_y) - 1)

            if verbose:
                print(f"      x_step: {x_step:.8f}, y_step: {y_step:.8f}")

            # Compute finite differences in x and y directions
            x_smoothness = self._compute_directional_smoothness(
                xy, heights, direction=0, step_size=x_step
            )
            y_smoothness = self._compute_directional_smoothness(
                xy, heights, direction=1, step_size=y_step
            )

            total_smoothness = (x_smoothness + y_smoothness) / 2.0
            if verbose:
                print(
                    f"      [green]x_smoothness: {x_smoothness:.6f}, y_smoothness: {y_smoothness:.6f}[/green]"
                )
                print(f"      [green]total_smoothness: {total_smoothness:.6f}[/green]")

            return total_smoothness
        else:
            # Fallback to simple approach
            if verbose:
                print(f"      [yellow]No clear grid structure, using fallback method[/yellow]")
            return torch.tensor(0.0, device=xy.device)

    def _compute_directional_smoothness(
        self, xy: torch.Tensor, heights: torch.Tensor, direction: int, step_size: float
    ) -> torch.Tensor:
        """Compute smoothness in a specific direction using finite differences."""
        # Find points that are step_size apart in the given direction
        direction_coords = xy[:, direction]
        other_coords = xy[:, 1 - direction]

        # Group by the other coordinate
        unique_other = torch.unique(other_coords)

        smoothness_terms = []
        for other_val in unique_other:
            # Find points along this line
            mask = other_coords == other_val
            line_coords = direction_coords[mask]
            line_heights = heights[mask]

            if len(line_coords) > 1:
                # Sort by direction coordinate
                sorted_indices = torch.argsort(line_coords)
                sorted_coords = line_coords[sorted_indices]
                sorted_heights = line_heights[sorted_indices]

                # Compute finite differences
                coord_diffs = torch.diff(sorted_coords)
                height_diffs = torch.diff(sorted_heights)

                # Only use differences close to expected step size
                expected_step_mask = torch.abs(coord_diffs - step_size) < step_size * 0.1
                if expected_step_mask.any():
                    valid_height_diffs = height_diffs[expected_step_mask]
                    valid_coord_diffs = coord_diffs[expected_step_mask]

                    # Compute normalized differences
                    normalized_diffs = valid_height_diffs / valid_coord_diffs
                    smoothness_terms.append(torch.mean(normalized_diffs**2))

        if smoothness_terms:
            return torch.stack(smoothness_terms).mean()
        else:
            return torch.tensor(0.0, device=xy.device)

    def _compute_gradient_smoothness(
        self, xy: torch.Tensor, heights: torch.Tensor, verbose: bool = False
    ) -> torch.Tensor:
        """Compute spatial smoothness using gradients (requires gradients)."""
        if verbose:
            print(f"    [cyan]DEBUG spatial_smoothness (gradients):[/cyan]")
            print(
                f"      [yellow]Gradient-based smoothness not yet implemented, using finite_diff[/yellow]"
            )
        return self._compute_finite_diff_smoothness(xy, heights, verbose=verbose)

    def fit(
        self,
        xy: torch.Tensor,
        z: torch.Tensor,
        lr: float = 1e-3,
        max_epochs: int = 1000,
        batch_size: int = 10000,
        grad_clip: float = 1.0,
        grad_weight: float = 0.01,
        laplacian_weight: float = 0.001,
        patience: int = 100,
        verbose: bool = False,
        early_stopping: bool = True,
        spatial_method: str = "finite_diff",
        enable_spatial: bool = True,
        logger=None,
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
        # Ensure z is 2D
        # if z.dim() == 1:
        #     z = z.unsqueeze(1)

        # Move data to device
        xy = xy.to(self.device)
        z = z.to(self.device)

        # Compute scaling parameters
        self.compute_scaling_parameters(xy, z, verbose=verbose)

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

        # Loss function
        criterion = nn.MSELoss()

        # Training variables
        losses = []
        best_loss = float("inf")
        patience_counter = 0

        # Initialize logger if provided
        if logger is not None:
            logger.start_training(max_epochs, xy.shape[0], batch_size)
            logger.log_model_parameters(self)

        if verbose:
            print(f"[bold green]Starting training with {max_epochs} epochs...[/bold green]")
            print(f"[green]Data size:[/green] {xy.shape[0]} points")
            print(f"[green]Batch size:[/green] {batch_size}")
            print(f"[green]Learning rate:[/green] {lr}")

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

            # Skip if we get invalid predictions
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                pbar.set_postfix({"loss": "NaN/Inf", "status": "skipped"})
                continue

            if verbose:
                print(f"[cyan]Pred:[/cyan] {pred.min():.6f}, {pred.max():.6f}")
                print(f"[cyan]Z_batch:[/cyan] {z_batch.min():.6f}, {z_batch.max():.6f}")

                # Compute height loss with detailed debugging
                print("[bold blue]DEBUG - Loss calculation:[/bold blue]")
                print(f"  [cyan]pred shape:[/cyan] {pred.shape}, dtype: {pred.dtype}")
                print(f"  [cyan]z_batch shape:[/cyan] {z_batch.shape}, dtype: {z_batch.dtype}")
                print(f"  [cyan]pred range:[/cyan] [{pred.min():.6f}, {pred.max():.6f}]")
                print(f"  [cyan]z_batch range:[/cyan] [{z_batch.min():.6f}, {z_batch.max():.6f}]")

            # Check for NaN/inf in predictions and targets
            if verbose:
                if torch.isnan(pred).any():
                    print("  [bold red]ERROR: NaN detected in predictions![/bold red]")
                if torch.isinf(pred).any():
                    print("  [bold red]ERROR: Inf detected in predictions![/bold red]")
                if torch.isnan(z_batch).any():
                    print("  [bold red]ERROR: NaN detected in targets![/bold red]")
                if torch.isinf(z_batch).any():
                    print("  [bold red]ERROR: Inf detected in targets![/bold red]")

            # Compute height loss
            height_loss = criterion(pred, z_batch.squeeze())
            if verbose:
                print(f"  [green]height_loss:[/green] {height_loss.item():.6f}")

            # Skip if loss is invalid
            if torch.isnan(height_loss) or torch.isinf(height_loss):
                pbar.set_postfix({"loss": "NaN/Inf", "status": "skipped"})
                if verbose:
                    print(
                        f"[bold red]Warning: Invalid loss at epoch {epoch}, skipping...[/bold red]"
                    )
                continue

            # Check if height loss is reasonable (not too large)
            if height_loss.item() > 1e6:
                pbar.set_postfix({"loss": f"{height_loss.item():.2e}", "status": "too_large"})
                if verbose:
                    print(
                        f"[bold red]Warning: Loss too large at epoch {epoch}: {height_loss.item():.2e}, skipping...[/bold red]"
                    )
                continue

            # Compute regularization losses with debugging
            reg_losses = self.compute_regularization_loss(
                xy_batch,
                pred,
                spatial_method=spatial_method,
                enable_spatial=enable_spatial,
                verbose=verbose,
            )
            if verbose:
                print("  [yellow]Regularization losses:[/yellow]")
                print(
                    f"    [cyan]spatial_smoothness:[/cyan] {reg_losses['spatial_smoothness'].item():.6f}"
                )
                print(
                    f"    [cyan]height_variance:[/cyan] {reg_losses['height_variance'].item():.6f}"
                )

            # Total loss with debugging
            spatial_term = grad_weight * reg_losses["spatial_smoothness"]
            variance_term = laplacian_weight * reg_losses["height_variance"]
            total_loss = height_loss + spatial_term + variance_term

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
                    total_loss=total_loss.item(),
                    height_loss=height_loss.item(),
                    spatial_smoothness_loss=reg_losses["spatial_smoothness"].item(),
                    height_variance_loss=reg_losses["height_variance"].item(),
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

            if verbose:
                print("  [yellow]Total loss components:[/yellow]")
                print(f"    [green]height_loss:[/green] {height_loss.item():.6f}")
                print(f"    [cyan]spatial_term:[/cyan] {spatial_term.item():.6f}")
                print(f"    [cyan]variance_term:[/cyan] {variance_term.item():.6f}")
                print(f"    [bold green]total_loss:[/bold green] {total_loss.item():.6f}")

            # Check if total loss is reasonable
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
                pbar.set_postfix({"loss": "NaN/Inf", "status": "skipped"})
                if verbose:
                    print(
                        f"[bold red]Warning: Invalid total loss at epoch {epoch}: {total_loss.item()}, skipping...[/bold red]"
                    )
                continue

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoding.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.height_net.parameters(), grad_clip)

            optimizer.step()

            # Update learning rate
            scheduler.step(total_loss)

            # Store loss
            losses.append(total_loss.item())

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

        # Update training history
        self.training_history["losses"] = losses
        self.training_history["final_loss"] = losses[-1] if losses else float("inf")

        # Log training completion with wandb if logger is provided
        if logger is not None:
            final_loss = (
                total_loss.item()
                if "total_loss" in locals()
                else self.training_history.get("final_loss", float("inf"))
            )
            logger.log_training_completion(final_loss, len(losses))

        if verbose:
            print(
                f"[bold green]Training completed! Final loss: {total_loss.item():.6f}[/bold green]"
            )
            print(
                f"Best loss: {self.training_history['best_loss']:.6f} at epoch {self.training_history['best_epoch']}"
            )
            print(f"Final loss: {self.training_history['final_loss']:.6f}")

        # Compute final loss over all data
        with torch.no_grad():
            final_pred = self.forward(xy)
            final_loss = criterion(final_pred, z.squeeze()).item()
            print(f"Total loss over all data: {final_loss:.6f}")

        return losses

    def evaluate(
        self, xy: torch.Tensor, z: torch.Tensor, logger=None, split: str = "test"
    ) -> Dict[str, float]:
        """Evaluate the model on test data."""
        with torch.no_grad():
            # Ensure inputs are on the correct device and use half precision
            xy = xy.to(self.device).half()
            z = z.to(self.device).half()

            pred = self.forward(xy)

            # Compute metrics
            mse = torch.mean((pred - z) ** 2)
            mae = torch.mean(torch.abs(pred - z))
            rmse = torch.sqrt(mse)

            # Convert to Python scalars
            mse_val = mse.item()
            mae_val = mae.item()
            rmse_val = rmse.item()

            # Relative errors
            z_range = z.max() - z.min()
            rel_mse = mse / (z_range**2)
            rel_mae = mae / z_range

            metrics = {
                "mse": mse_val,
                "mae": mae_val,
                "rmse": rmse_val,
                "rel_mse": rel_mse.item(),
                "rel_mae": rel_mae.item(),
            }

            # Log evaluation metrics with wandb if logger is provided
            if logger is not None:
                from .logger import EvaluationMetrics

                eval_metrics = EvaluationMetrics(
                    mse=mse_val,
                    mae=mae_val,
                    rmse=rmse_val,
                    rel_mse=rel_mse.item(),
                    rel_mae=rel_mae.item(),
                    dataset_size=xy.shape[0],
                )

                logger.log_evaluation_metrics(eval_metrics, split=split)

            return metrics

    def plot_training_history(self):
        """Plot training loss history."""
        try:
            import matplotlib.pyplot as plt

            losses = self.training_history["losses"]
            if not losses:
                print("No training history available.")
                return

            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.title("Training Loss History")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.grid(True)
            plt.show()

        except ImportError:
            print("matplotlib not available for plotting")

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
