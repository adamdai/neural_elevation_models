# Enhanced DEM Class Documentation

## Overview

The enhanced `DEM` class in `nemo/dem.py` provides a flexible interface for loading and working with Digital Elevation Models (DEMs) from various file formats. It automatically handles different input types and provides standardized X, Y, Z coordinate arrays as output.

**NEW: The `data` field now contains the concatenated NxMx3 XYZ coordinates!**

## Features

- **Multiple Input Formats**: Supports GeoTIFF (.tif), NumPy arrays (.npy, .dat), and image files (.png, .jpg)
- **Standardized Output**: Always provides X, Y, Z coordinate arrays in consistent formats
- **Flexible Coordinate Systems**: Handles both simple grid coordinates and geospatial coordinates
- **Built-in Functionality**: Includes downsampling, elevation querying, and data export
- **Metadata Support**: Preserves source information and coordinate system details
- **Neural Model Ready**: Data field is already in NxMx3 format perfect for neural networks

## Installation Requirements

The enhanced DEM class requires the following optional dependencies:

```bash
# For GeoTIFF support
pip install rasterio

# For image file support  
pip install Pillow

# Core dependencies (already included)
numpy
```

## Basic Usage

### Loading DEMs from Files

```python
from nemo.dem import DEM

# Load from different file types
dem_tif = DEM.from_file("data/elevation.tif")      # GeoTIFF
dem_dat = DEM.from_file("data/elevation.dat")      # NumPy .dat file
dem_npy = DEM.from_file("data/elevation.npy")      # NumPy .npy file
dem_img = DEM.from_file("data/heightmap.png")      # Image file
```

### Creating DEMs from Arrays

```python
import numpy as np

# Create elevation data
elevation_data = np.random.rand(100, 100) * 1000  # 100x100 grid, 0-1000m

# Create DEM with custom extent
dem = DEM.from_array(elevation_data, extent=(1000, 1000), metadata={'source': 'synthetic'})
```

## Data Structure

### The `data` Field

The most important change is that `dem.data` now contains the concatenated XYZ coordinates:

```python
# Before: dem.data was just elevation (N, M)
# Now: dem.data contains XYZ coordinates (N, M, 3)

dem.data.shape  # (N, M, 3)
dem.data[:,:,0]  # X coordinates
dem.data[:,:,1]  # Y coordinates  
dem.data[:,:,2]  # Z coordinates (elevation)
```

This makes it perfect for neural elevation models that expect NxMx3 input!

## Output Formats

The DEM class provides several ways to access X, Y, Z coordinate data:

### 1. Direct Access via `data` Field
```python
# Access XYZ coordinates directly
x_coords = dem.data[:,:,0]  # Shape: (N, M)
y_coords = dem.data[:,:,1]  # Shape: (N, M)
z_coords = dem.data[:,:,2]  # Shape: (N, M)

# Perfect for neural networks!
neural_input = dem.data  # Shape: (N, M, 3)
```

### 2. Grid Arrays (N, M)
```python
x, y, z = dem.get_xyz_arrays()
# x.shape = (N, M), y.shape = (N, M), z.shape = (N, M)
```

### 3. Flattened Arrays (N*M)
```python
x_flat, y_flat, z_flat = dem.get_xyz_flat()
# Each is a 1D array of length N*M
```

### 4. Combined Array (N*M, 3)
```python
xyz_combined = dem.get_xyz_combined()
# Shape: (N*M, 3) where each row is [x, y, z]
```

### 5. Elevation Only
```python
elevation = dem.get_elevation()  # Shape: (N, M) - just Z coordinates
```

## File Format Support

### GeoTIFF (.tif, .tiff)
- **Requirements**: `rasterio` library
- **Features**: 
  - Preserves geospatial information (CRS, transform, bounds)
  - Automatic extent calculation from geospatial bounds
  - Metadata preservation
- **Output**: Standardized X, Y, Z arrays in meters
- **Data field**: Contains generated coordinate grids + elevation

### NumPy Arrays (.npy, .npz, .dat)
- **Requirements**: `numpy` (already included)
- **Features**:
  - Handles 2D elevation arrays
  - Handles 3D arrays with [x, y, z] structure
  - Handles 4D arrays with [x, y, z, ?] structure (like .dat files)
  - Automatic extent calculation from coordinate ranges
- **Output**: Standardized X, Y, Z arrays
- **Data field**: Contains the XYZ coordinates directly

### Image Files (.png, .jpg, .jpeg)
- **Requirements**: `PIL/Pillow` library
- **Features**:
  - Converts grayscale images to elevation data
  - Configurable elevation range
  - Customizable spatial extent
- **Output**: Standardized X, Y, Z arrays
- **Data field**: Contains generated coordinate grids + elevation

## Advanced Features

### Elevation Querying
```python
# Query elevation at specific coordinates with bilinear interpolation
elevation = dem.query(x_coord, y_coord)
```

### Downsampling
```python
# Downsample by factor of 2
dem_ds = dem.downsample(2)
# New shape: (N//2, M//2, 3)
# New extent: (extent_x/2, extent_y/2)
```

### Missing Data Handling
```python
# Handle missing data values (default: -32767.0)
dem.handle_missing_data(missing_value=-9999.0)
```

### Data Export
```python
# Export to XYZ format
xyz_data = dem.get_xyz_combined()
np.savetxt("dem_xyz.txt", xyz_data, fmt='%.6f', header='X Y Z')

# Or export directly from data field
data_reshaped = dem.data.reshape(-1, 3)
np.savetxt("dem_xyz_direct.txt", data_reshaped, fmt='%.6f', header='X Y Z')
```

## Neural Model Compatibility

The new data structure is perfect for neural elevation models:

```python
# Perfect input shape for neural networks
dem.data.shape  # (N, M, 3)

# For MLPs that expect flattened input
mlp_input = dem.data.reshape(-1, 3)  # Shape: (N*M, 3)

# For batch processing
batch_input = dem.data.reshape(1, -1, 3)  # Shape: (1, N*M, 3)

# For convolutional networks
conv_input = dem.data  # Shape: (N, M, 3) - ready for conv layers!
```

## Examples

### Example 1: Loading Lunar DEM Data
```python
from nemo.dem import DEM

# Load lunar DEM from .dat file
dem = DEM.from_file("data/Moon_Map_01_0_rep0.dat")
print(f"DEM shape: {dem.data.shape}")  # Now (N, M, 3)!
print(f"Spatial extent: {dem.extent}")

# Access XYZ coordinates directly
x_coords = dem.data[:,:,0]
y_coords = dem.data[:,:,1]
z_coords = dem.data[:,:,2]

print(f"Coordinate ranges: X({x_coords.min():.1f}, {x_coords.max():.1f})")
```

### Example 2: Working with GeoTIFF
```python
# Load GeoTIFF with geospatial information
dem = DEM.from_file("data/elevation.tif")
print(f"CRS: {dem.metadata.get('crs')}")
print(f"Data shape: {dem.data.shape}")  # (N, M, 3)

# Query elevation at specific coordinates
elevation = dem.query(1000.0, 2000.0)  # meters
print(f"Elevation at (1000, 2000): {elevation:.1f}m")

# Direct access to coordinates
x_range = dem.data[:,:,0].max() - dem.data[:,:,0].min()
y_range = dem.data[:,:,1].max() - dem.data[:,:,1].min()
print(f"Coordinate ranges: X={x_range:.1f}m, Y={y_range:.1f}m")
```

### Example 3: Creating Synthetic Terrain
```python
import numpy as np

# Create synthetic terrain
N, M = 100, 100
X, Y = np.meshgrid(np.linspace(0, 100, M), np.linspace(0, 100, N))
Z = 50 + 20 * np.sin(X / 10) + 15 * np.cos(Y / 15)

# Create DEM
dem = DEM.from_array(Z, extent=(100, 100), metadata={'source': 'synthetic'})

# Data is already in perfect format for neural networks!
print(f"Data shape: {dem.data.shape}")  # (100, 100, 3)

# Export to XYZ format
xyz = dem.get_xyz_combined()
np.savetxt("synthetic_terrain.xyz", xyz)

# Or export directly from data field
data_export = dem.data.reshape(-1, 3)
np.savetxt("synthetic_terrain_direct.xyz", data_export)
```

## Class Attributes

- **`data`**: 3D XYZ coordinate array (N, M, 3) where data[:,:,0] = X, data[:,:,1] = Y, data[:,:,2] = Z
- **`N, M`**: Dimensions of the DEM
- **`extent`**: Spatial extent in meters (width, height)
- **`x_coords, y_coords, z_coords`**: Coordinate arrays (N, M) - extracted from data field
- **`metadata`**: Dictionary containing source information and additional data

## Class Methods

- **`from_file(file_path)`**: Load DEM from file (class method)
- **`from_array(data, extent, metadata)`**: Create DEM from array (class method)
- **`get_xyz_arrays()`**: Get X, Y, Z coordinate arrays
- **`get_elevation()`**: Get elevation data (Z coordinates)
- **`get_xyz_flat()`**: Get flattened coordinate arrays
- **`get_xyz_combined()`**: Get combined XYZ array
- **`query(x, y)`**: Query elevation at coordinates
- **`downsample(factor)`**: Downsample DEM data
- **`handle_missing_data(missing_value)`**: Handle missing data
- **`show()`**: Display DEM visualization

## Error Handling

The class provides informative error messages for common issues:

- **File not found**: Clear file path information
- **Unsupported format**: Lists supported formats
- **Missing dependencies**: Installation instructions for optional libraries
- **Invalid data structure**: Detailed information about expected array shapes

## Performance Considerations

- **Large GeoTIFFs**: Use downsampling for very large files
- **Memory usage**: Coordinate arrays are stored in memory for fast access
- **Interpolation**: Bilinear interpolation provides smooth elevation queries
- **File I/O**: GeoTIFF loading uses rasterio for efficient reading
- **Neural models**: Data is already in the right format - no reshaping needed!

## Troubleshooting

### Common Issues

1. **ImportError for rasterio**: Install with `pip install rasterio`
2. **ImportError for PIL**: Install with `pip install Pillow`
3. **File format not supported**: Check file extension and ensure it's in the supported list
4. **Memory issues with large files**: Use downsampling or load smaller regions

### Getting Help

- Check the examples in `examples/dem_usage_examples.py`
- Run the examples: `python examples/dem_usage_examples.py`
- Verify file formats and dependencies
- Check the metadata for coordinate system information

## Migration from Old Version

If you were using the old DEM class:

```python
# Old way
dem = DEM(elevation_data)
elevation = dem.data  # Shape: (N, M)

# New way
dem = DEM.from_array(elevation_data)
elevation = dem.get_elevation()  # Shape: (N, M)
xyz_coords = dem.data  # Shape: (N, M, 3) - NEW!
```
