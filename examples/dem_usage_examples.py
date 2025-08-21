#!/usr/bin/env python3
"""
Examples of using the enhanced DEM class for different use cases.
This file demonstrates the flexibility of the DEM class for various input formats.
The DEM.data field now contains the concatenated NxMx3 XYZ coordinates.
"""

import numpy as np
from pathlib import Path
from nemo.dem import DEM


def example_1_load_dat_file():
    """Example 1: Loading a .dat file and getting standardized X, Y, Z arrays."""
    print("Example 1: Loading .dat file")
    print("-" * 40)

    # Load the lunar DEM data
    dat_path = Path("data/Moon_Map_01_0_rep0.dat")
    if dat_path.exists():
        dem = DEM.from_file(dat_path)
        print(f"Loaded DEM: {dem}")
        print(f"Data shape: {dem.data.shape}")  # Now (N, M, 3)
        print("Data structure: data[:,:,0] = X, data[:,:,1] = Y, data[:,:,2] = Z")

        # Get X, Y, Z arrays in different formats
        x, y, z = dem.get_xyz_arrays()  # Each is (N, M) shape
        print(f"X, Y, Z arrays shape: {x.shape}")

        # Get flattened arrays
        x_flat, y_flat, z_flat = dem.get_xyz_flat()  # Each is 1D
        print(f"Flattened arrays length: {len(x_flat)}")

        # Get combined array of shape (N*M, 3)
        xyz_combined = dem.get_xyz_combined()
        print(f"Combined XYZ shape: {xyz_combined.shape}")
        print(f"First 5 points:\n{xyz_combined[:5]}")

        # Access data directly - now it's the XYZ coordinates!
        print(f"Direct access to data:")
        print(f"  X range: {dem.data[:, :, 0].min():.3f} to {dem.data[:, :, 0].max():.3f}")
        print(f"  Y range: {dem.data[:, :, 1].min():.3f} to {dem.data[:, :, 1].max():.3f}")
        print(f"  Z range: {dem.data[:, :, 2].min():.3f} to {dem.data[:, :, 2].max():.3f}")

        return dem
    else:
        print(f"File not found: {dat_path}")
        return None


def example_2_load_geotiff():
    """Example 2: Loading a GeoTIFF file with geospatial information."""
    print("\nExample 2: Loading GeoTIFF file")
    print("-" * 40)

    tif_path = Path("data/Site01_final_adj_5mpp_surf.tif")
    if tif_path.exists():
        dem = DEM.from_file(tif_path)
        print(f"Loaded GeoTIFF: {dem}")
        print(f"Data shape: {dem.data.shape}")  # Now (N, M, 3)
        print(f"Metadata: {dem.metadata}")

        # Get coordinate arrays
        x, y, z = dem.get_xyz_arrays()
        print(f"Coordinate ranges:")
        print(f"  X: {x.min():.1f} to {x.max():.1f} meters")
        print(f"  Y: {y.min():.1f} to {y.max():.1f} meters")
        print(f"  Z: {z.min():.1f} to {z.max():.1f} meters")

        # Direct access to XYZ data
        print(f"Direct data access:")
        print(f"  data[:,:,0] (X): {dem.data[:, :, 0].min():.1f} to {dem.data[:, :, 0].max():.1f}")
        print(f"  data[:,:,1] (Y): {dem.data[:, :, 1].min():.1f} to {dem.data[:, :, 1].max():.1f}")
        print(f"  data[:,:,2] (Z): {dem.data[:, :, 2].min():.1f} to {dem.data[:, :, 2].max():.1f}")

        return dem
    else:
        print(f"File not found: {tif_path}")
        return None


def example_3_create_from_array():
    """Example 3: Creating DEM from numpy array with custom extent."""
    print("\nExample 3: Creating DEM from numpy array")
    print("-" * 40)

    # Create synthetic terrain data
    N, M = 50, 50
    x = np.linspace(0, 100, M)
    y = np.linspace(0, 100, N)
    X, Y = np.meshgrid(x, y)

    # Create elevation with multiple peaks
    Z = (
        10
        + 5 * np.sin(X / 10)
        + 3 * np.cos(Y / 15)
        + 2 * np.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 200)
    )

    # Create DEM with custom extent (100m x 100m)
    dem = DEM.from_array(Z, extent=(100, 100), metadata={"source": "synthetic_terrain"})
    print(f"Created DEM: {dem}")
    print(f"Data shape: {dem.data.shape}")  # Now (N, M, 3)

    # Get coordinate arrays
    x_coords, y_coords, z_coords = dem.get_xyz_arrays()
    print(f"Terrain features:")
    print(f"  Peak elevation: {z_coords.max():.1f}m")
    print(f"  Valley elevation: {z_coords.min():.1f}m")
    print(f"  Elevation range: {z_coords.max() - z_coords.min():.1f}m")

    # Show that data now contains XYZ coordinates
    print(f"Data structure verification:")
    print(f"  data[:,:,0] == x_coords: {np.allclose(dem.data[:, :, 0], x_coords)}")
    print(f"  data[:,:,1] == y_coords: {np.allclose(dem.data[:, :, 1], y_coords)}")
    print(f"  data[:,:,2] == z_coords: {np.allclose(dem.data[:, :, 2], z_coords)}")

    return dem


def example_4_query_and_interpolation(dem):
    """Example 4: Elevation querying and interpolation."""
    print("\nExample 4: Elevation querying and interpolation")
    print("-" * 40)

    if dem is None:
        print("No DEM available for this example")
        return

    # Query elevations at specific points
    query_points = [
        (dem.extent[0] * 0.1, dem.extent[1] * 0.1),  # Near corner
        (dem.extent[0] * 0.5, dem.extent[1] * 0.5),  # Center
        (dem.extent[0] * 0.8, dem.extent[1] * 0.8),  # Near opposite corner
    ]

    print("Elevation queries:")
    for x, y in query_points:
        elevation = dem.query(x, y)
        print(f"  At ({x:.1f}, {y:.1f}): {elevation:.2f}m")

    # Show that we can also get elevation directly from data
    print(f"Direct elevation access from data[:,:,2]:")
    print(f"  Min elevation: {dem.data[:, :, 2].min():.2f}m")
    print(f"  Max elevation: {dem.data[:, :, 2].max():.2f}m")


def example_5_downsampling_and_analysis(dem):
    """Example 5: Downsampling and analyzing DEM data."""
    print("\nExample 5: Downsampling and analysis")
    print("-" * 40)

    if dem is None:
        print("No DEM available for this example")
        return

    print(f"Original DEM shape: {dem.data.shape}")

    # Downsample by different factors
    for factor in [2, 4, 8]:
        dem_ds = dem.downsample(factor)
        print(f"Downsampled by {factor}: {dem_ds.data.shape}")

        # Get statistics from the downsampled data
        x_ds = dem_ds.data[:, :, 0]
        y_ds = dem_ds.data[:, :, 1]
        z_ds = dem_ds.data[:, :, 2]
        print(
            f"  Coordinate ranges: X({x_ds.min():.1f}, {x_ds.max():.1f}), "
            f"Y({y_ds.min():.1f}, {y_ds.max():.1f})"
        )
        print(
            f"  Elevation stats: min={z_ds.min():.2f}, max={z_ds.max():.2f}, mean={z_ds.mean():.2f}"
        )


def example_6_export_xyz_data(dem, output_file="dem_xyz_export.txt"):
    """Example 6: Exporting DEM data to XYZ format."""
    print("\nExample 6: Exporting to XYZ format")
    print("-" * 40)

    if dem is None:
        print("No DEM available for this example")
        return

    # Get combined XYZ data
    xyz_data = dem.get_xyz_combined()

    # Save to text file
    np.savetxt(output_file, xyz_data, fmt="%.6f", header="X Y Z", comments="# ")

    print(f"Exported {xyz_data.shape[0]} points to {output_file}")
    print(f"File format: X Y Z (space-separated)")
    print(f"Sample data:\n{xyz_data[:5]}")

    # Alternative: export directly from data array
    # Reshape the 3D data to 2D for export
    data_reshaped = dem.data.reshape(-1, 3)
    print(f"Direct export from data array shape: {data_reshaped.shape}")
    print(f"Data arrays are identical: {np.allclose(xyz_data, data_reshaped)}")


def example_7_neural_model_compatibility():
    """Example 7: Demonstrating compatibility with neural elevation models."""
    print("\nExample 7: Neural model compatibility")
    print("-" * 40)

    # Create a simple DEM
    N, M = 32, 32
    X, Y = np.meshgrid(np.linspace(0, 10, M), np.linspace(0, 10, N))
    Z = 1 + 0.5 * np.sin(X) * np.cos(Y)

    dem = DEM.from_array(Z, extent=(10, 10), metadata={"source": "neural_test"})
    print(f"Created test DEM: {dem}")

    # Perfect for neural models - data is already in NxMx3 format!
    print(f"Data ready for neural models:")
    print(f"  Shape: {dem.data.shape}")  # (32, 32, 3)
    print(f"  X coordinates: {dem.data[:, :, 0].min():.2f} to {dem.data[:, :, 0].max():.2f}")
    print(f"  Y coordinates: {dem.data[:, :, 1].min():.2f} to {dem.data[:, :, 1].max():.2f}")
    print(f"  Z coordinates: {dem.data[:, :, 2].min():.2f} to {dem.data[:, :, 2].max():.2f}")

    # Easy to use in neural networks
    print(f"Neural model usage:")
    print(f"  Input shape: {dem.data.shape}")
    print(f"  Flattened for MLPs: {dem.data.reshape(-1, 3).shape}")
    print(f"  Batch processing: {dem.data.reshape(1, -1, 3).shape}")

    return dem


def main():
    """Run all examples."""
    print("Enhanced DEM Class Usage Examples")
    print("=" * 60)
    print("NEW: Data field now contains NxMx3 XYZ coordinates!")
    print("=" * 60)

    # Run examples
    dem1 = example_1_load_dat_file()
    dem2 = example_2_load_geotiff()
    dem3 = example_3_create_from_array()
    dem7 = example_7_neural_model_compatibility()

    # Test functionality with different DEMs
    if dem1:
        example_4_query_and_interpolation(dem1)
        example_5_downsampling_and_analysis(dem1)
        example_6_export_xyz_data(dem1, "lunar_dem_xyz.txt")

    if dem2:
        example_4_query_and_interpolation(dem2)
        example_5_downsampling_and_analysis(dem2)
        example_6_export_xyz_data(dem2, "geotiff_dem_xyz.txt")

    if dem3:
        example_4_query_and_interpolation(dem3)
        example_5_downsampling_and_analysis(dem3)
        example_6_export_xyz_data(dem3, "synthetic_dem_xyz.txt")

    print("\n" + "=" * 60)
    print("Examples complete! Check the generated XYZ files.")
    print("The DEM.data field now contains the concatenated NxMx3 XYZ coordinates!")


if __name__ == "__main__":
    main()
