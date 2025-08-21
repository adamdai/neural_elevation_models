#!/usr/bin/env python3
"""
Example demonstrating the flexible losses logging system.

This shows how to use the updated logger with different loss configurations.
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import nemo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nemo.nemov2 import NEMoV2
from nemo.logger import Logger, TrainingMetrics


def generate_sample_data(n_points: int = 1000):
    """Generate sample elevation data."""
    x = np.linspace(-5, 5, int(np.sqrt(n_points)))
    y = np.linspace(-5, 5, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)

    # Simple elevation surface
    Z = 2.0 + 0.5 * np.sin(X / 2) + 0.3 * np.cos(Y / 3)

    # Flatten and convert to tensors
    xy = torch.tensor(np.column_stack([X.flatten(), Y.flatten()]), dtype=torch.float32)
    z = torch.tensor(Z.flatten(), dtype=torch.float32)

    return xy, z


def example_with_height_loss_only():
    """Example with only height loss (no regularization)."""
    print("=== Example 1: Height Loss Only ===")

    # Initialize logger
    logger = Logger(
        project_name="nemo-flexible-losses",
        run_name="height-loss-only",
        config={"loss_type": "height_only"},
    )

    try:
        # Generate data
        xy, z = generate_sample_data(1000)

        # Create model
        model = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

        # Start logging
        logger.start_training(max_epochs=100, data_size=len(xy), batch_size=500)
        logger.log_model_parameters(model)

        # Simulate training loop
        for epoch in range(10):
            logger.start_epoch(epoch)

            # Simulate loss computation
            height_loss = 0.1 * np.exp(-epoch / 5) + 0.01 * np.random.random()

            # Create flexible losses dictionary
            losses = {
                "height_loss": height_loss,
                "total_loss": height_loss,  # In this case, height_loss is the total
            }

            # Create training metrics
            metrics = TrainingMetrics(
                epoch=epoch, losses=losses, learning_rate=1e-3, training_time=0.1
            )

            # Log metrics
            logger.log_training_metrics(metrics)

            print(f"Epoch {epoch}: height_loss={height_loss:.6f}")

        # Log completion
        logger.log_training_completion(final_loss=height_loss, total_epochs=10)

    finally:
        logger.finish()


def example_with_multiple_losses():
    """Example with multiple loss components."""
    print("\n=== Example 2: Multiple Loss Components ===")

    # Initialize logger
    logger = Logger(
        project_name="nemo-flexible-losses",
        run_name="multiple-losses",
        config={"loss_type": "multiple_components"},
    )

    try:
        # Generate data
        xy, z = generate_sample_data(1000)

        # Create model
        model = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

        # Start logging
        logger.start_training(max_epochs=100, data_size=len(xy), batch_size=500)
        logger.log_model_parameters(model)

        # Simulate training loop
        for epoch in range(10):
            logger.start_epoch(epoch)

            # Simulate different loss components
            height_loss = 0.1 * np.exp(-epoch / 5) + 0.01 * np.random.random()
            smoothness_loss = 0.05 * np.exp(-epoch / 3) + 0.005 * np.random.random()
            variance_loss = 0.02 * np.exp(-epoch / 4) + 0.002 * np.random.random()

            # Create flexible losses dictionary
            losses = {
                "height_loss": height_loss,
                "smoothness_loss": smoothness_loss,
                "variance_loss": variance_loss,
                "total_loss": height_loss + smoothness_loss + variance_loss,
            }

            # Create training metrics
            metrics = TrainingMetrics(
                epoch=epoch, losses=losses, learning_rate=1e-3, training_time=0.1
            )

            # Log metrics
            logger.log_training_metrics(metrics)

            print(
                f"Epoch {epoch}: total={losses['total_loss']:.6f}, "
                f"height={height_loss:.6f}, smooth={smoothness_loss:.6f}, "
                f"var={variance_loss:.6f}"
            )

        # Log completion
        final_total_loss = losses["total_loss"]
        logger.log_training_completion(final_loss=final_total_loss, total_epochs=10)

    finally:
        logger.finish()


def example_with_custom_loss_names():
    """Example with custom loss names."""
    print("\n=== Example 3: Custom Loss Names ===")

    # Initialize logger
    logger = Logger(
        project_name="nemo-flexible-losses",
        run_name="custom-loss-names",
        config={"loss_type": "custom_names"},
    )

    try:
        # Generate data
        xy, z = generate_sample_data(1000)

        # Create model
        model = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

        # Start logging
        logger.start_training(max_epochs=100, data_size=len(xy), batch_size=500)
        logger.log_model_parameters(model)

        # Simulate training loop
        for epoch in range(10):
            logger.start_epoch(epoch)

            # Simulate losses with custom names
            reconstruction_loss = 0.1 * np.exp(-epoch / 5) + 0.01 * np.random.random()
            perceptual_loss = 0.03 * np.exp(-epoch / 6) + 0.003 * np.random.random()
            adversarial_loss = 0.02 * np.exp(-epoch / 7) + 0.002 * np.random.random()

            # Create flexible losses dictionary with custom names
            losses = {
                "reconstruction": reconstruction_loss,
                "perceptual": perceptual_loss,
                "adversarial": adversarial_loss,
                "total": reconstruction_loss + perceptual_loss + adversarial_loss,
            }

            # Create training metrics
            metrics = TrainingMetrics(
                epoch=epoch, losses=losses, learning_rate=1e-3, training_time=0.1
            )

            # Log metrics
            logger.log_training_metrics(metrics)

            print(
                f"Epoch {epoch}: total={losses['total']:.6f}, "
                f"recon={reconstruction_loss:.6f}, "
                f"perceptual={perceptual_loss:.6f}, "
                f"adv={adversarial_loss:.6f}"
            )

        # Log completion
        final_total_loss = losses["total"]
        logger.log_training_completion(final_loss=final_total_loss, total_epochs=10)

    finally:
        logger.finish()


def main():
    """Run all examples."""
    print("Flexible Losses Logging Examples")
    print("=" * 50)

    # Run examples with different loss configurations
    example_with_height_loss_only()
    example_with_multiple_losses()
    example_with_custom_loss_names()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)
    print("\nKey benefits of the flexible losses system:")
    print("1. No hardcoded loss names - works with any loss configuration")
    print("2. Automatically logs all losses in the dictionary")
    print("3. Easy to add/remove loss components without changing the logger")
    print("4. Maintains backward compatibility")


if __name__ == "__main__":
    main()
