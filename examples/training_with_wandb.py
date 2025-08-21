#!/usr/bin/env python3
"""
Example script demonstrating how to use NEMoV2 with Weights & Biases logging.

This script shows how to:
1. Set up wandb logging
2. Train a NEMoV2 model with comprehensive logging
3. Evaluate the model and log metrics
4. Save the model with logging
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import nemo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nemo.nemov2 import NEMoV2
from nemo.logger import NEMoLogger


def generate_sample_data(n_points: int = 10000, noise_std: float = 0.1):
    """Generate sample elevation data for demonstration."""
    # Generate grid of coordinates
    x = np.linspace(-10, 10, int(np.sqrt(n_points)))
    y = np.linspace(-10, 10, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)

    # Create a synthetic elevation surface (mountain-like)
    Z = 5 * np.exp(-(X**2 + Y**2) / 20) + 2 * np.exp(-((X - 3) ** 2 + (Y - 2) ** 2) / 5)

    # Add some noise
    Z += np.random.normal(0, noise_std, Z.shape)

    # Flatten and convert to tensors
    xy = torch.tensor(np.column_stack([X.flatten(), Y.flatten()]), dtype=torch.float32)
    z = torch.tensor(Z.flatten(), dtype=torch.float32)

    return xy, z


def main():
    """Main training function with wandb logging."""

    # Generate sample data
    print("Generating sample elevation data...")
    xy, z = generate_sample_data(n_points=10000, noise_std=0.1)

    # Split data into train/test
    train_size = int(0.8 * len(xy))
    train_indices = torch.randperm(len(xy))[:train_size]
    test_indices = torch.randperm(len(xy))[train_size:]

    xy_train, z_train = xy[train_indices], z[train_indices]
    xy_test, z_test = xy[test_indices], z[test_indices]

    print(f"Training set: {len(xy_train)} points")
    print(f"Test set: {len(xy_test)} points")

    # Initialize the model
    print("Initializing NEMoV2 model...")
    model = NEMoV2(
        input_bounds=(-10, 10, -10, 10),
        output_bounds=(0, 10),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Set up wandb logging
    print("Setting up wandb logging...")
    config = {
        "model": "NEMoV2",
        "encoding": "HashGrid",
        "network": "FullyFusedMLP",
        "learning_rate": 1e-3,
        "max_epochs": 500,
        "batch_size": 5000,
        "grad_clip": 1.0,
        "grad_weight": 0.01,
        "laplacian_weight": 0.001,
        "patience": 50,
        "early_stopping": True,
        "spatial_method": "finite_diff",
        "enable_spatial": True,
    }

    # Initialize logger
    logger = NEMoLogger(
        project_name="nemo-elevation-demo",
        run_name="nemo-training-example",
        config=config,
        log_gradients=True,
        log_memory=True,
        log_timing=True,
    )

    try:
        # Train the model with logging
        print("Starting training with wandb logging...")
        losses = model.fit(
            xy=xy_train,
            z=z_train,
            lr=config["learning_rate"],
            max_epochs=config["max_epochs"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            grad_weight=config["grad_weight"],
            laplacian_weight=config["laplacian_weight"],
            patience=config["patience"],
            verbose=True,
            early_stopping=config["early_stopping"],
            spatial_method=config["spatial_method"],
            enable_spatial=config["enable_spatial"],
            logger=logger,
        )

        print(f"Training completed! Final loss: {losses[-1]:.6f}")

        # Evaluate on test set with logging
        print("Evaluating model on test set...")
        test_metrics = model.evaluate(xy_test, z_test, logger=logger, split="test")

        print("Test set metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.6f}")

        # Save the model with logging
        print("Saving model...")
        model.save_model("nemo_trained_model.pth", logger=logger)

        # Log custom metrics
        logger.log_custom_metric("training/final_loss", losses[-1])
        logger.log_custom_metric("training/best_loss", min(losses))
        logger.log_custom_metric("training/epochs_trained", len(losses))

        print("Training and logging completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # Always finish the wandb run
        logger.finish()


if __name__ == "__main__":
    main()
