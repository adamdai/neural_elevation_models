#!/usr/bin/env python3
"""
Simple example of using NEMoV2 to fit to DEM data.
This shows the basic workflow for fitting a neural height field to existing DEM data.
"""

import torch
import numpy as np
from nemo.dem import DEM
from nemo.nemov2 import NEMoV2


def basic_dem_fitting_example():
    """Basic example of fitting NEMoV2 to DEM data."""

    print("NEMoV2 DEM Fitting Example")
    print("=" * 50)

    # 1. Load DEM data
    print("1. Loading DEM data...")
    dem = DEM.from_file("data/Moon_Map_01_0_rep0.dat")

    # Downsample for faster training
    dem_ds = dem.downsample(2)  # From 180x180 to 90x90

    # Get XYZ coordinates
    xyz = dem_ds.get_xyz_combined()

    # Convert to torch tensors
    xy = torch.from_numpy(xyz[:, :2]).float()
    z = torch.from_numpy(xyz[:, 2]).float()

    print(f"   Data shape: {xy.shape}")
    print(
        f"   Coordinate ranges: X({xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}), Y({xy[:, 1].min():.3f}, {xy[:, 1].max():.3f})"
    )
    print(f"   Elevation range: Z({z.min():.3f}, {z.max():.3f})")

    # 2. Create NEMoV2 instance
    print("\n2. Creating NEMoV2 instance...")
    nemov2 = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {nemov2.device}")

    # 3. Fit the model
    print("\n3. Fitting NEMoV2 to DEM data...")
    losses = nemov2.fit(
        xy,
        z,
        lr=1e-4,  # Learning rate
        max_epochs=2000,  # Maximum training epochs
        grad_weight=0.2,  # Gradient regularization weight
        laplacian_weight=0.1,  # Laplacian regularization weight
        verbose=True,  # Print training progress
    )

    # 4. Evaluate the model
    print("\n4. Evaluating fitted model...")
    metrics = nemov2.evaluate(xy, z)

    print("   Evaluation metrics:")
    for key, value in metrics.items():
        print(f"     {key}: {value:.6f}")

    # 5. Test predictions
    print("\n5. Testing predictions...")
    with torch.no_grad():
        pred = nemov2.forward(xy)
        print(f"   Prediction range: {pred.min():.3f} to {pred.max():.3f}")
        print(f"   Target range: {z.min():.3f} to {z.max():.3f}")

        # Compare with original data
        mse = torch.mean((pred - z) ** 2)
        mae = torch.mean(torch.abs(pred - z))
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")

    # 6. Save the model
    print("\n6. Saving trained model...")
    nemov2.save_model("fitted_nemov2_model.pth")

    print("\n✓ Example completed successfully!")
    return nemov2


def advanced_fitting_example():
    """Advanced example with different configurations and comparison."""

    print("\nAdvanced NEMoV2 Fitting Example")
    print("=" * 50)

    # Load smaller dataset for faster comparison
    dem = DEM.from_file("data/Moon_Map_01_0_rep0.dat")
    dem_ds = dem.downsample(4)  # 45x45
    xyz = dem_ds.get_xyz_combined()

    xy = torch.from_numpy(xyz[:, :2]).float()
    z = torch.from_numpy(xyz[:, 2]).float()

    print(f"Data shape: {xy.shape}")

    # Test different regularization settings
    configs = [
        {"name": "Low regularization", "grad_weight": 0.05, "laplacian_weight": 0.02},
        {"name": "Medium regularization", "grad_weight": 0.2, "laplacian_weight": 0.1},
        {"name": "High regularization", "grad_weight": 0.5, "laplacian_weight": 0.2},
    ]

    results = {}

    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 30)

        # Create new instance for each test
        nemov2 = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

        # Fit with specific configuration
        losses = nemov2.fit(
            xy,
            z,
            lr=1e-4,
            max_epochs=1000,
            grad_weight=config["grad_weight"],
            laplacian_weight=config["laplacian_weight"],
            verbose=False,
        )

        if losses:
            # Evaluate
            metrics = nemov2.evaluate(xy, z)
            results[config["name"]] = {
                "final_loss": losses[-1],
                "best_loss": min(losses),
                "metrics": metrics,
            }

            print(f"  Final loss: {losses[-1]:.6f}")
            print(f"  Best loss: {min(losses):.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  Relative MAE: {metrics['rel_mae']:.6f}")

    # Compare results
    print(f"\n{'=' * 50}")
    print("Configuration Comparison")
    print(f"{'=' * 50}")

    for name, result in results.items():
        print(f"{name}:")
        print(f"  Best loss: {result['best_loss']:.6f}")
        print(f"  RMSE: {result['metrics']['rmse']:.6f}")
        print(f"  Relative MAE: {result['metrics']['rel_mae']:.6f}")
        print()


if __name__ == "__main__":
    # Run basic example
    basic_dem_fitting_example()

    # Run advanced example
    advanced_fitting_example()

    print("\n🎉 All examples completed!")
    print("You can now use NEMoV2 in your surface fitting notebook!")



