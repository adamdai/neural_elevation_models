import wandb
import torch
from typing import Dict, Optional, Union
from dataclasses import dataclass
import time


@dataclass
class TrainingMetrics:
    """Container for training metrics to be logged."""

    epoch: int
    losses: Dict[str, float]  # Flexible dictionary of loss names and values
    learning_rate: float
    gradient_norm: Optional[float] = None
    memory_usage: Optional[float] = None
    training_time: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics to be logged."""

    mse: float
    mae: float
    rmse: float
    rel_mse: float
    rel_mae: float
    dataset_size: int


class Logger:
    """
    Logger class for NEMoV2 training and evaluation using Weights & Biases.

    This class handles all wandb logging operations, providing a clean interface
    for tracking training progress, metrics, and model artifacts.
    """

    def __init__(
        self,
        project_name: str = "neural-elevation-models",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_gradients: bool = False,
        log_memory: bool = True,
        log_timing: bool = True,
    ):
        """
        Initialize the logger.

        Args:
            project_name: Name of the wandb project
            run_name: Name for this specific run (auto-generated if None)
            config: Configuration dictionary to log
            log_gradients: Whether to log gradient norms
            log_memory: Whether to log memory usage
            log_timing: Whether to log timing information
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.log_gradients = log_gradients
        self.log_memory = log_memory
        self.log_timing = log_timing

        # Initialize wandb run
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=["nemo", "elevation", "neural-network"],
        )

        # Training state
        self.training_start_time = None
        self.epoch_start_time = None
        self.best_loss = float("inf")
        self.best_epoch = 0

        # Log initial configuration
        self._log_config()

    def _log_config(self):
        """Log the initial configuration."""
        # Only update config if it hasn't been set yet
        if not hasattr(wandb.config, "_initialized") or not wandb.config._initialized:
            try:
                wandb.config.update(self.config)
                wandb.config._initialized = True
            except Exception as e:
                print(f"Warning: Could not update wandb config: {e}")
                # Mark as initialized to avoid repeated attempts
                wandb.config._initialized = True

    def start_training(self, max_epochs: int, data_size: int, batch_size: int):
        """Log the start of training."""
        self.training_start_time = time.time()

        wandb.log(
            {
                "training/start_time": self.training_start_time,
                "training/max_epochs": max_epochs,
                "training/data_size": data_size,
                "training/batch_size": batch_size,
            },
            step=0,
        )

    def start_epoch(self, epoch: int):
        """Log the start of an epoch."""
        self.epoch_start_time = time.time()

        wandb.log(
            {
                "training/epoch": epoch,
                "training/epoch_start_time": self.epoch_start_time,
            },
            step=epoch,
        )

    def log_training_metrics(self, metrics: TrainingMetrics):
        """
        Log training metrics for an epoch.

        Args:
            metrics: TrainingMetrics object containing all metrics to log
        """
        # Start with learning rate
        log_dict = {"training/learning_rate": metrics.learning_rate}

        # Dynamically log all losses from the losses dictionary
        for loss_name, loss_value in metrics.losses.items():
            log_dict[f"training/{loss_name}"] = loss_value

        # Add optional metrics if available
        if metrics.gradient_norm is not None:
            log_dict["training/gradient_norm"] = metrics.gradient_norm

        if metrics.memory_usage is not None:
            log_dict["training/memory_usage_mb"] = metrics.memory_usage

        if metrics.training_time is not None:
            log_dict["training/epoch_time"] = metrics.training_time

        # Log to wandb
        wandb.log(log_dict, step=metrics.epoch)

        # Update best loss tracking using total_loss if it exists, otherwise use the first loss
        total_loss = metrics.losses.get("total_loss", next(iter(metrics.losses.values())))
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_epoch = metrics.epoch

            wandb.log(
                {
                    "training/best_loss": self.best_loss,
                    "training/best_epoch": self.best_epoch,
                },
                step=metrics.epoch,
            )

    def log_evaluation_metrics(self, metrics: EvaluationMetrics, split: str = "test"):
        """
        Log evaluation metrics.

        Args:
            metrics: EvaluationMetrics object containing all metrics to log
            split: Dataset split name (e.g., "train", "test", "val")
        """
        log_dict = {
            f"evaluation/{split}/mse": metrics.mse,
            f"evaluation/{split}/mae": metrics.mae,
            f"evaluation/{split}/rmse": metrics.rmse,
            f"evaluation/{split}/relative_mse": metrics.rel_mse,
            f"evaluation/{split}/relative_mae": metrics.rel_mae,
            f"evaluation/{split}/dataset_size": metrics.dataset_size,
        }

        wandb.log(log_dict)

    def log_model_parameters(self, model):
        """Log model architecture and parameter information."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb.log(
            {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
            }
        )

        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)

    def log_gradient_norms(self, model, step: int):
        """Log gradient norms for model parameters."""
        if not self.log_gradients:
            return

        total_norm = 0.0
        param_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_norms[f"gradients/{name}"] = param_norm.item()

        total_norm = total_norm**0.5

        # Log total gradient norm
        wandb.log({"training/total_gradient_norm": total_norm, **param_norms}, step=step)

    def log_memory_usage(self, step: int):
        """Log GPU memory usage if available."""
        if not self.log_memory or not torch.cuda.is_available():
            return

        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB

        wandb.log(
            {
                "system/gpu_memory_allocated_mb": memory_allocated,
                "system/gpu_memory_reserved_mb": memory_reserved,
            },
            step=step,
        )

    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping information."""
        wandb.log(
            {
                "training/early_stopping_triggered": True,
                "training/early_stopping_epoch": epoch,
                "training/early_stopping_patience": patience,
            },
            step=epoch,
        )

    def log_training_completion(self, final_loss: float, total_epochs: int):
        """Log training completion information."""
        if self.training_start_time is not None:
            total_training_time = time.time() - self.training_start_time

            wandb.log(
                {
                    "training/completed": True,
                    "training/final_loss": final_loss,
                    "training/total_epochs": total_epochs,
                    "training/total_training_time": total_training_time,
                    "training/epochs_per_second": total_epochs / total_training_time,
                }
            )

    def log_model_save(self, filepath: str, model_size_mb: Optional[float] = None):
        """Log model saving information."""
        log_dict = {
            "model/saved": True,
            "model/save_path": filepath,
        }

        if model_size_mb is not None:
            log_dict["model/size_mb"] = model_size_mb

        wandb.log(log_dict)

    def log_model_load(self, filepath: str):
        """Log model loading information."""
        wandb.log(
            {
                "model/loaded": True,
                "model/load_path": filepath,
            }
        )

    def log_custom_metric(
        self, name: str, value: Union[float, int, str], step: Optional[int] = None
    ):
        """Log a custom metric."""
        if step is not None:
            wandb.log({name: value}, step=step)
        else:
            wandb.log({name: value})

    def finish(self):
        """Finish the wandb run."""
        if self.run:
            self.run.finish()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
