#!/usr/bin/env python3
"""
Test script for NEMoV2 class - specifically designed for fitting to DEM data.
"""

import torch
from rich import print
import argparse

from nemo.nemov2 import NEMoV2
from nemo.logger import NEMoLogger


def test_nemov2_synthetic(logger=None):
    """Test NEMoV2 with synthetic data."""

    print("Testing NEMoV2 with synthetic data...")

    # Create synthetic terrain data
    N = 1000
    x = torch.linspace(-10, 10, N)
    y = torch.linspace(-10, 10, N)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    # Create smooth terrain with some features
    Z = 2.0 + 0.5 * torch.sin(X / 2) + 0.3 * torch.cos(Y / 3) + 0.1 * torch.randn_like(X)

    # Flatten and create training data
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    z = Z.flatten()

    print(f"Data shape: xy={xy.shape}, z={z.shape}")
    print(
        f"Coordinate ranges: X({xy[:, 0].min():.2f}, {xy[:, 0].max():.2f}), Y({xy[:, 1].min():.2f}, {xy[:, 1].max():.2f})"
    )
    print(f"Elevation range: Z({z.min():.2f}, {z.max():.2f})")

    # Create NEMoV2 instance
    nemov2 = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

    # Test fitting with different configurations
    configs = [
        {"lr": 1e-4, "grad_weight": 0.1, "laplacian_weight": 0.05},
        {"lr": 1e-3, "grad_weight": 0.2, "laplacian_weight": 0.1},
        {"lr": 1e-2, "grad_weight": 0.5, "laplacian_weight": 0.2},
    ]

    for i, config in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"Test configuration {i + 1}: {config}")
        print(f"{'=' * 60}")

        try:
            # Fit the model
            losses = nemov2.fit(
                xy,
                z,
                lr=config["lr"],
                max_epochs=1000,
                grad_weight=config["grad_weight"],
                laplacian_weight=config["laplacian_weight"],
                verbose=True,
                logger=logger,
            )

            if losses:
                print(f"✓ Training completed successfully!")
                print(f"  Final loss: {losses[-1]:.6f}")
                print(f"  Best loss: {min(losses):.6f}")
                print(f"  No inf/NaN losses encountered")

                # Test evaluation
                metrics = nemov2.evaluate(xy, z, logger=logger, split="synthetic")
                print(f"  Evaluation metrics:")
                for key, value in metrics.items():
                    print(f"    {key}: {value:.6f}")

                # Test prediction
                with torch.no_grad():
                    pred = nemov2.forward(xy)
                    print(f"  Prediction range: {pred.min():.3f} to {pred.max():.3f}")

            else:
                print("⚠ Training completed but no loss history returned")

        except Exception as e:
            print(f"✗ Training failed with error: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("Synthetic data test completed!")
    print(f"{'=' * 60}")


def test_nemov2_real_ldem(logger=None):
    """Test NEMoV2 with real LDEM data."""

    print("\nTesting NEMoV2 with real LDEM data...")

    try:
        # Load LDEM data using the enhanced DEM class
        from nemo.dem import DEM

        # Load the lunar DEM data
        dem = DEM.from_file("data/Moon_Map_01_0_rep0.dat")

        # Downsample to manageable size for testing
        dem_ds = dem.downsample(4)  # Reduce from 180x180 to 45x45

        # Get XYZ data
        xyz = dem_ds.get_xyz_combined()

        # Convert to torch tensors and move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        xy = torch.from_numpy(xyz[:, :2]).float().to(device)
        z = torch.from_numpy(xyz[:, 2]).float().to(device)

        print(f"Real LDEM data shape: xy={xy.shape}, z={z.shape}")
        print(
            f"Coordinate ranges: X({xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}), Y({xy[:, 1].min():.3f}, {xy[:, 1].max():.3f})"
        )
        print(f"Elevation range: Z({z.min():.3f}, {z.max():.3f})")

        # Create new NEMoV2 instance
        nemov2 = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

        # Test fitting with conservative settings
        print("\nFitting NEMoV2 to real LDEM data...")
        losses = nemov2.fit(
            xy,
            z,
            lr=1e-4,
            max_epochs=2000,
            grad_weight=0.2,  # Higher regularization for real data
            laplacian_weight=0.1,
            verbose=True,
            logger=logger,
        )

        if losses:
            print(f"✓ Real LDEM training completed successfully!")
            print(f"  Final loss: {losses[-1]:.6f}")
            print(f"  Best loss: {min(losses):.6f}")

            # Test evaluation
            metrics = nemov2.evaluate(xy, z, logger=logger, split="real_ldem")
            print(f"\nEvaluation metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")

            # Test prediction
            print("\nTesting predictions...")
            with torch.no_grad():
                pred = nemov2.forward(xy)
                mse = torch.mean((pred - z) ** 2)
                print(f"  Test MSE: {mse:.6f}")
                print(f"  Prediction range: {pred.min():.3f} to {pred.max():.3f}")

            # Save the trained model
            nemov2.save_model("ldem_nemov2_model.pth", logger=logger)

        else:
            print("⚠ Real LDEM training completed but no loss history returned")

    except Exception as e:
        print(f"✗ Real LDEM test failed with error: {e}")
        import traceback

        traceback.print_exc()


def test_nemov2_architectures(logger=None):
    """Test different NEMoV2 architectures."""

    print("\nTesting different NEMoV2 architectures...")

    # Create small test data
    N = 100
    xy = torch.rand(N, 2) * 10 - 5  # Random coordinates in [-5, 5]
    z = 1.0 + 0.5 * torch.sin(xy[:, 0]) + 0.3 * torch.cos(xy[:, 1])  # Simple function

    # Test different encoding configurations
    encoding_configs = [
        {
            "name": "HashGrid (default)",
            "config": None,  # Use default
        },
        {
            "name": "HashGrid (smaller)",
            "config": {
                "otype": "HashGrid",
                "n_levels": 4,
                "n_features_per_level": 4,  # Valid value
                "log2_hashmap_size": 16,
                "base_resolution": 8,
                "per_level_scale": 2.0,
                "interpolation": "Smoothstep",
            },
        },
        {
            "name": "HashGrid (larger)",
            "config": {
                "otype": "HashGrid",
                "n_levels": 12,
                "n_features_per_level": 8,  # Valid value
                "log2_hashmap_size": 20,
                "base_resolution": 32,
                "per_level_scale": 1.2,
                "interpolation": "Smoothstep",
            },
        },
    ]

    for config_info in encoding_configs:
        print(f"\n{'=' * 50}")
        print(f"Testing architecture: {config_info['name']}")
        print(f"{'=' * 50}")

        try:
            # Create NEMoV2 with specific configuration
            nemov2 = NEMoV2(
                encoding_config=config_info["config"],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Quick training test
            losses = nemov2.fit(xy, z, lr=1e-3, max_epochs=500, verbose=False, logger=logger)

            if losses:
                print(f"✓ Architecture test successful!")
                print(f"  Final loss: {losses[-1]:.6f}")
                print(f"  Best loss: {min(losses):.6f}")
            else:
                print("⚠ Architecture test completed but no loss history returned")

        except Exception as e:
            print(f"✗ Architecture test failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEMoV2 tests with optional wandb logging")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--project", type=str, default="neural-elevation-models", help="wandb project name"
    )
    parser.add_argument("--run-name", type=str, default="test-nemov2", help="wandb run name")
    args = parser.parse_args()

    logger = None
    if args.wandb:
        logger = NEMoLogger(
            project_name=args.project, run_name=args.run_name, config={"script": "test_nemov2"}
        )

    try:
        print("NEMoV2 Testing Suite")
        print("=" * 80)

        # Test with synthetic data first
        test_nemov2_synthetic(logger=logger)

        # Test different architectures
        test_nemov2_architectures(logger=logger)

        # Test with real LDEM data
        test_nemov2_real_ldem(logger=logger)

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
    finally:
        if logger is not None:
            logger.finish()
