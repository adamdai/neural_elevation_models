#!/usr/bin/env python3
"""
Simple NEMoV2 Training Sweep Script with Hydra

This script performs hyperparameter sweeps over different encoding and MLP configurations
using Hydra's built-in sweep functionality. It saves trained models to the results folder
and logs all results to Weights & Biases.

Usage:
    # Run a single training job
    python scripts/train_sweep.py

    # Run a sweep over all combinations
    python scripts/train_sweep.py -m encoding.n_levels=4,6,8 encoding.n_features_per_level=2,4,8 network.n_neurons=32,64,128

    # Run with Optuna sweeper for more intelligent sampling
    python scripts/train_sweep.py --multirun --config-name=sweep_optuna
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from datetime import datetime

from nemo.util.plotting import plot_surface
from nemo.dem import DEM
from nemo.nemov2 import NEMoV2
from nemo.logger import Logger
from nemo.util.paths import data_dir


def load_dem_data(dem_data: str, device: str) -> tuple:
    """
    Load DEM data based on the specified type.

    Args:
        dem_data: Type of DEM data ("LAC" or "LDEM")
        device: Device to load data on

    Returns:
        Tuple of (xy, z) tensors
    """
    if dem_data == "LAC":
        dem = DEM.from_file(data_dir() / "Moon_Map_01_0_rep0.dat")
    else:  # LDEM
        tif_path = Path(data_dir() / "Site01_final_adj_5mpp_surf.tif")
        dem = DEM.from_file(tif_path)

    xyz = dem.get_xyz_combined()
    xy = torch.from_numpy(xyz[:, :2]).float().to(device)
    z = torch.from_numpy(xyz[:, 2]).float().to(device)

    return xy, z


def get_model_size(model: NEMoV2) -> Dict[str, int]:
    """
    Calculate the size of the model in terms of parameters and memory.

    Args:
        model: NEMoV2 model instance

    Returns:
        Dictionary with model size information
    """
    total_params = 0
    trainable_params = 0

    # Count encoding parameters
    for param in model.encoding.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # Count network parameters
    for param in model.height_net.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # Estimate memory usage (rough approximation)
    memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "memory_mb": memory_mb,
    }


def create_encoding_config(encoding_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create encoding configuration from config parameters.

    Args:
        encoding_config: Encoding configuration dictionary

    Returns:
        Encoding configuration dictionary for TCNN
    """

    # Helper function to extract single values from lists
    def extract_single_value(value, default):
        if isinstance(value, list):
            return value[0]
        return value

    # Handle different encoding types
    if encoding_config.get("otype") == "HashGrid":
        return {
            "otype": "HashGrid",
            "n_levels": extract_single_value(encoding_config.get("n_levels"), 6),
            "n_features_per_level": extract_single_value(
                encoding_config.get("n_features_per_level"), 2
            ),
            "log2_hashmap_size": extract_single_value(encoding_config.get("log2_hashmap_size"), 18),
            "base_resolution": extract_single_value(encoding_config.get("base_resolution"), 16),
            "per_level_scale": extract_single_value(encoding_config.get("per_level_scale"), 1.5),
            "interpolation": "Linear",
        }
    elif encoding_config.get("otype") == "SphericalHarmonics":
        return {
            "otype": "SphericalHarmonics",
            "degree": extract_single_value(encoding_config.get("degree"), 4),
        }
    elif encoding_config.get("otype") == "Frequency":
        return {
            "otype": "Frequency",
            "n_frequencies": extract_single_value(encoding_config.get("n_frequencies"), 8),
        }
    else:
        # Default HashGrid
        return {
            "otype": "HashGrid",
            "n_levels": 6,
            "n_features_per_level": 2,
            "log2_hashmap_size": 18,
            "base_resolution": 16,
            "per_level_scale": 1.5,
            "interpolation": "Linear",
        }


def create_network_config(network_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create network configuration from config parameters.

    Args:
        network_config: Network configuration dictionary

    Returns:
        Network configuration dictionary for TCNN
    """

    # Helper function to extract single values from lists
    def extract_single_value(value, default):
        if isinstance(value, list):
            return value[0]
        return value

    return {
        "otype": extract_single_value(network_config.get("otype"), "FullyFusedMLP"),
        "activation": extract_single_value(network_config.get("activation"), "ReLU"),
        "output_activation": extract_single_value(network_config.get("output_activation"), "None"),
        "n_neurons": extract_single_value(network_config.get("n_neurons"), 64),
        "n_hidden_layers": extract_single_value(network_config.get("n_hidden_layers"), 3),
    }


def save_sweep_results(
    config: DictConfig,
    model: NEMoV2,
    final_loss: float,
    training_history: list,
    model_size: Dict[str, int],
    output_dir: Path,
) -> None:
    """
    Save sweep results including model, config, and metrics.

    Args:
        config: Hydra configuration
        model: Trained NEMoV2 model
        final_loss: Final training loss
        training_history: Training history
        model_size: Model size information
        output_dir: Output directory for results
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.pth"
    model.save_model(str(model_path))

    # Save configuration
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(OmegaConf.to_container(config, resolve=True), f, default_flow_style=False)

    # Save training results
    results = {
        "final_loss": float(final_loss),
        "model_size": model_size,
        "timestamp": datetime.now().isoformat(),
        "hydra_job_num": None,  # Simplified for now
    }

    results_path = output_dir / "results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"Results saved to {output_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="sweep_config")
def main(config: DictConfig) -> None:
    """
    Main training function for the sweep.

    Args:
        config: Hydra configuration object
    """
    print(f"Starting training with configuration:")
    print(OmegaConf.to_yaml(config))

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DEM data
    print("Loading DEM data...")
    xy, z = load_dem_data("LAC", device)  # Fixed to LAC for now
    print(f"Loaded {xy.shape[0]} data points")

    # Create encoding and network configurations
    print(f"Raw encoding config: {config.encoding}")
    print(f"Raw network config: {config.network}")

    encoding_config = create_encoding_config(config.encoding)
    network_config = create_network_config(config.network)

    print(f"Processed encoding config: {encoding_config}")
    print(f"Processed network config: {network_config}")

    # Create model
    print("Creating NEMoV2 model...")

    # Final check: ensure no list values remain in configs
    def clean_config(config_dict):
        """Remove any remaining list values from config dict"""
        cleaned = {}
        for key, value in config_dict.items():
            if isinstance(value, list):
                print(f"Warning: {key} still contains list {value}, using first value")
                cleaned[key] = value[0]
            else:
                cleaned[key] = value
        return cleaned

    encoding_config = clean_config(encoding_config)
    network_config = clean_config(network_config)

    print(f"Final encoding config: {encoding_config}")
    print(f"Final network config: {network_config}")

    model = NEMoV2(encoding_config=encoding_config, network_config=network_config, device=device)

    # Get model size information
    model_size = get_model_size(model)
    print(f"Model size: {model_size}")

    # Set up wandb logging
    print("Setting up wandb logging...")
    run_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project="neural-elevation-models",
        name=run_name,
        config={
            "encoding": encoding_config,
            "network": network_config,
            "training": config.training,
            "model_size": model_size,
            "dem_data": "LAC",
            "device": device,
        },
    )

    # Training parameters
    training_config = config.training

    # Helper function to extract single values from lists
    def extract_single_value(value, default):
        if isinstance(value, list):
            return value[0]
        return value

    # Extract single values from lists if they exist
    lr = extract_single_value(training_config.learning_rate, 1e-3)
    max_epochs = extract_single_value(training_config.max_epochs, 2000)
    batch_size = extract_single_value(training_config.batch_size, 10000)

    # Convert config to a simple dict to avoid wandb serialization issues
    # Use OrderedDict to ensure consistent key ordering
    from collections import OrderedDict

    config_dict = OrderedDict(
        [
            ("encoding", encoding_config),
            ("network", network_config),
            (
                "training",
                OrderedDict(
                    [("learning_rate", lr), ("max_epochs", max_epochs), ("batch_size", batch_size)]
                ),
            ),
            ("model_size", model_size),
            ("dem_data", "LAC"),
            ("device", device),
        ]
    )

    logger = Logger(project_name="neural-elevation-models", run_name=run_name, config=config_dict)

    # Train the model
    print("Starting training...")
    try:
        training_history = model.fit(
            xy=xy,
            z=z,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            verbose=False,
            early_stopping=True,
            patience=100,
            logger=logger,
        )

        # Get final loss
        if training_history is not None:
            final_loss = training_history[-1] if training_history else float("inf")
            print(f"Training completed. Final loss: {final_loss:.6f}")

            # Log training history
            for i, loss in enumerate(training_history):
                wandb.log({"training_loss": loss, "epoch": i})
        else:
            # If training_history is None, try to get final loss from the model
            final_loss = float("inf")
            print("Training completed but no training history returned")

            # Try to get the final loss from the model's training history if available
            if hasattr(model, "training_history") and model.training_history:
                final_loss = model.training_history[-1]
                print(f"Final loss from model: {final_loss:.6f}")

        # Log final results to wandb
        wandb.log({"final_loss": final_loss, "model_size": model_size, "training_completed": True})

        # Save results
        # Check if we're actually in a sweep (Hydra creates sweep.dir only when -m is used)
        if (
            hasattr(config, "hydra")
            and hasattr(config.hydra, "sweep")
            and hasattr(config.hydra.sweep, "dir")
        ):
            # We're in a sweep - use the sweep directory
            sweep_dir = Path(config.hydra.sweep.dir)

            # If this is the first job in the sweep, rename the directory to be more descriptive
            if config.hydra.job.num == 0:
                # Create a more descriptive sweep directory name
                sweep_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                new_sweep_dir = Path("results") / sweep_name
                if sweep_dir.exists() and sweep_dir != new_sweep_dir:
                    sweep_dir.rename(new_sweep_dir)
                sweep_dir = new_sweep_dir

            # Create descriptive subdirectory name based on swept parameters
            job_name_parts = []
            if hasattr(config, "encoding") and hasattr(config.encoding, "n_levels"):
                job_name_parts.append(f"levels_{config.encoding.n_levels}")
            if hasattr(config, "encoding") and hasattr(config.encoding, "n_features_per_level"):
                job_name_parts.append(f"features_{config.encoding.n_features_per_level}")
            if hasattr(config, "network") and hasattr(config.network, "n_neurons"):
                job_name_parts.append(f"neurons_{config.network.n_neurons}")
            if hasattr(config, "network") and hasattr(config.network, "n_hidden_layers"):
                job_name_parts.append(f"layers_{config.network.n_hidden_layers}")
            if hasattr(config, "training") and hasattr(config.training, "learning_rate"):
                job_name_parts.append(f"lr_{config.training.learning_rate}")

            if job_name_parts:
                # Include W&B run ID for easy cross-referencing
                wandb_run_id = wandb.run.id if wandb.run else "unknown"
                job_dir = sweep_dir / f"{'_'.join(job_name_parts)}_wandb_{wandb_run_id}"
            else:
                job_dir = sweep_dir / str(config.hydra.job.num)
        else:
            # Single job - use timestamp
            sweep_dir = Path("results") / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            job_dir = sweep_dir / "0"

        save_sweep_results(
            config=config,
            model=model,
            final_loss=final_loss,
            training_history=training_history,
            model_size=model_size,
            output_dir=job_dir,
        )

        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed with error: {e}")
        wandb.log({"training_completed": False, "error": str(e)})
        raise

    finally:
        # Clean up wandb
        wandb.finish()


if __name__ == "__main__":
    main()
