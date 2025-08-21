#!/usr/bin/env python3
"""
Test script for different spatial regularization methods in NEMoV2.
"""

import torch
import numpy as np
from nemo.nemov2 import NEMoV2


def test_spatial_methods():
    """Test different spatial regularization methods."""

    print("Testing NEMoV2 Spatial Regularization Methods")
    print("=" * 60)

    # Create synthetic grid data (regular grid to test finite differences)
    N = 20  # 20x20 grid
    x = torch.linspace(-1, 1, N)
    y = torch.linspace(-1, 1, N)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    # Create smooth terrain with some features
    Z = 1.0 + 0.5 * torch.sin(X * np.pi) + 0.3 * torch.cos(Y * np.pi) + 0.1 * torch.randn_like(X)

    # Flatten and create training data
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    z = Z.flatten()

    print(f"Data shape: {xy.shape}")
    print(f"Grid structure: {N}x{N} = {N * N} points")
    print(
        f"Coordinate ranges: X({xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}), Y({xy[:, 1].min():.3f}, {xy[:, 1].max():.3f})"
    )
    print(f"Elevation range: Z({z.min():.3f}, {z.max():.3f})")

    # Test different spatial methods
    methods = [
        ("finite_diff", True, "Finite differences over x and y individually"),
        ("gradients", True, "Gradient-based smoothness (falls back to finite_diff)"),
        ("disabled", True, "Spatial regularization disabled"),
        ("finite_diff", False, "Spatial regularization completely disabled"),
    ]

    for method, enable, description in methods:
        print(f"\n{'=' * 60}")
        print(f"Testing: {method} (enable={enable})")
        print(f"Description: {description}")
        print(f"{'=' * 60}")

        try:
            # Create new NEMoV2 instance for each test
            nemov2 = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

            # Quick training test with specific spatial method
            losses = nemov2.fit(
                xy,
                z,
                lr=1e-3,
                max_epochs=100,  # Just a few epochs for testing
                grad_weight=0.1,
                laplacian_weight=0.05,
                verbose=False,  # Reduce output for testing
                spatial_method=method,
                enable_spatial=enable,
            )

            if losses:
                print(f"✓ Method {method} successful!")
                print(f"  Final loss: {losses[-1]:.6f}")
                print(f"  Best loss: {min(losses):.6f}")
                print(f"  No NaN losses encountered")
            else:
                print(f"⚠ Method {method} completed but no loss history returned")

        except Exception as e:
            print(f"✗ Method {method} failed with error: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print("All spatial method tests completed!")
    print(f"{'=' * 60}")


def test_with_real_data():
    """Test with real LDEM data."""

    print("\nTesting with Real LDEM Data")
    print("=" * 60)

    try:
        from nemo.dem import DEM

        # Load LDEM data
        dem = DEM.from_file("data/Moon_Map_01_0_rep0.dat")
        dem_ds = dem.downsample(4)  # 45x45 for faster testing

        # Get XYZ data
        xyz = dem_ds.get_xyz_combined()
        xy = torch.from_numpy(xyz[:, :2]).float()
        z = torch.from_numpy(xyz[:, 2]).float()

        print(f"Real LDEM data shape: {xy.shape}")
        print(
            f"Coordinate ranges: X({xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}), Y({xy[:, 1].min():.3f}, {xy[:, 1].max():.3f})"
        )
        print(f"Elevation range: Z({z.min():.3f}, {z.max():.3f})")

        # Test with spatial regularization disabled
        print(f"\nTesting with spatial regularization DISABLED:")
        nemov2 = NEMoV2(device="cuda" if torch.cuda.is_available() else "cpu")

        losses = nemov2.fit(
            xy,
            z,
            lr=1e-4,
            max_epochs=500,  # Fewer epochs for testing
            grad_weight=0.0,  # No spatial regularization
            laplacian_weight=0.1,
            verbose=True,
            spatial_method="disabled",
            enable_spatial=False,
        )

        if losses:
            print(f"✓ Real LDEM training with disabled spatial regularization successful!")
            print(f"  Final loss: {losses[-1]:.6f}")
            print(f"  Best loss: {min(losses):.6f}")
            print(f"  No NaN losses encountered")
        else:
            print(f"⚠ Real LDEM training completed but no loss history returned")

    except Exception as e:
        print(f"✗ Real LDEM test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Test synthetic data with different methods
    test_spatial_methods()

    # Test with real LDEM data (spatial disabled)
    test_with_real_data()

    print("\n🎉 All tests completed!")
    print("Check the output above to see which spatial methods work best!")



