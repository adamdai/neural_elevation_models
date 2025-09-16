"""Digital Elevation Model (DEM) class and functions."""

import numpy as np
import os
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any

# Try to import optional dependencies
try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from nemo.util.plotting import plot_heatmap, plot_surface


class DEM:
    """
    Enhanced Digital Elevation Model (DEM) class supporting multiple input formats.

    This class can load DEMs from various sources:
    - GeoTIFF files (.tif, .tiff)
    - NumPy arrays (.npy, .dat)
    - PNG/JPEG images
    - Raw numpy arrays

    Attributes
    ----------
    data : np.array (N, M, 3)
        3D array of XYZ coordinates where data[:,:,0] = X, data[:,:,1] = Y, data[:,:,2] = Z.
    N, M : int
        Dimensions of DEM data.
    extent : tuple (x, y)
        Extent in x and y of DEM data (meters).
    x_coords : np.array (N, M)
        X coordinates for each grid point.
    y_coords : np.array (N, M)
        Y coordinates for each grid point.
    z_coords : np.array (N, M)
        Z coordinates (elevation) for each grid point.
    metadata : dict
        Additional metadata about the DEM (coordinate system, units, etc.)

    Methods
    -------
    from_file
        Load DEM from a file (class method).
    from_array
        Create DEM from numpy array (class method).
    get_xyz_arrays
        Get X, Y, Z coordinate arrays.
    get_elevation
        Get elevation data (Z coordinates).
    downsample
        Downsample the DEM data by a factor.
    query
        Query elevation at given xy point.
    handle_missing_data
        Handle missing data entries.
    show
        Display a heatmap visualization of the DEM.
    """

    def __init__(
        self,
        data: np.ndarray,
        extent: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize DEM from elevation data or XYZ coordinates.

        Parameters
        ----------
        data : np.array
            If 2D (N, M): elevation data (Z coordinates)
            If 3D (N, M, 3): XYZ coordinates where data[:,:,0] = X, data[:,:,1] = Y, data[:,:,2] = Z
        extent : tuple (x, y), optional
            Extent in x and y of DEM data (meters).
        metadata : dict, optional
            Additional metadata about the DEM.
        """
        self.metadata = metadata or {}

        if data.ndim == 2:
            # 2D elevation data provided
            self.N, self.M = data.shape
            if extent is None:
                # Assume one cell is 1 unit in x and y
                self.extent = (self.M, self.N)
            else:
                self.extent = extent

            # Generate coordinate grids and create 3D XYZ array
            self._generate_coordinate_grids()
            self.z_coords = data  # Set z_coords from input data
            self.data = np.stack([self.x_coords, self.y_coords, self.z_coords], axis=2)

        elif data.ndim == 3 and data.shape[2] == 3:
            # 3D XYZ coordinates provided
            self.N, self.M, _ = data.shape
            self.data = data

            # Extract coordinate arrays
            self.x_coords = data[:, :, 0]
            self.y_coords = data[:, :, 1]
            self.z_coords = data[:, :, 2]

            if extent is None:
                # Calculate extent from coordinate ranges
                self.extent = (
                    self.x_coords.max() - self.x_coords.min(),
                    self.y_coords.max() - self.y_coords.min(),
                )
            else:
                self.extent = extent
        else:
            raise ValueError(f"Data must be 2D (N, M) or 3D (N, M, 3), got shape {data.shape}")

    def _generate_coordinate_grids(self):
        """Generate X, Y coordinate grids based on extent."""
        x_range = np.linspace(0, self.extent[0], self.M)
        y_range = np.linspace(0, self.extent[1], self.N)
        self.x_coords, self.y_coords = np.meshgrid(x_range, y_range)
        # Note: z_coords will be set from the input data in __init__

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs) -> "DEM":
        """
        Load DEM from a file.

        Parameters
        ----------
        file_path : str or Path
            Path to the DEM file.
        **kwargs
            Additional arguments passed to specific loaders.

        Returns
        -------
        DEM
            Loaded DEM object.

        Raises
        ------
        ValueError
            If file format is not supported or file cannot be loaded.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and load accordingly
        suffix = file_path.suffix.lower()

        if suffix in [".tif", ".tiff"]:
            return cls._load_geotiff(file_path, **kwargs)
        elif suffix in [".npy", ".npz"]:
            return cls._load_numpy(file_path, **kwargs)
        elif suffix == ".dat":
            return cls._load_dat(file_path, **kwargs)
        elif suffix in [".png", ".jpg", ".jpeg"]:
            return cls._load_image(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @classmethod
    def _load_geotiff(cls, file_path: Path, **kwargs) -> "DEM":
        """Load DEM from GeoTIFF file."""
        if not HAS_RASTERIO:
            raise ImportError(
                "rasterio is required to load GeoTIFF files. Install with: pip install rasterio"
            )

        with rasterio.open(file_path) as src:
            # Read elevation data
            elevation_data = src.read(1)  # Read first band

            # Get geospatial information
            bounds = src.bounds
            transform = src.transform

            # Calculate extent in meters
            width_m = bounds.right - bounds.left
            height_m = bounds.top - bounds.bottom

            # Create metadata
            metadata = {
                "crs": src.crs,
                "transform": transform,
                "bounds": bounds,
                "units": "meters",
                "source": "geotiff",
            }

            return cls(elevation_data, extent=(width_m, height_m), metadata=metadata)

    @classmethod
    def _load_numpy(cls, file_path: Path, **kwargs) -> "DEM":
        """Load DEM from NumPy file."""
        data = np.load(file_path, allow_pickle=True)

        # Handle different numpy array shapes
        if data.ndim == 2:
            # Simple 2D elevation array
            return cls(data, metadata={"source": "numpy"})
        elif data.ndim == 3:
            # 3D array - need to determine structure
            if data.shape[2] == 3:
                # Assume [x, y, z] structure - perfect for our 3D XYZ format
                return cls(data, metadata={"source": "numpy_3d"})
            elif data.shape[2] == 4:
                # Assume [x, y, z, ?] structure (like .dat files)
                # Extract just the XYZ coordinates
                xyz_data = data[:, :, :3]
                return cls(xyz_data, metadata={"source": "numpy_4d"})
            else:
                raise ValueError(f"Unexpected 3D array shape: {data.shape}")
        else:
            raise ValueError(f"Unexpected array dimensions: {data.ndim}")

    @classmethod
    def _load_dat(cls, file_path: Path, **kwargs) -> "DEM":
        """Load DEM from .dat file (assumed to be numpy array)."""
        return cls._load_numpy(file_path, **kwargs)

    @classmethod
    def _load_image(cls, file_path: Path, **kwargs) -> "DEM":
        """Load DEM from image file (PNG, JPEG, etc.)."""
        if not HAS_PIL:
            raise ImportError(
                "PIL/Pillow is required to load image files. Install with: pip install Pillow"
            )

        # Load image and convert to grayscale
        image = Image.open(file_path).convert("L")
        elevation_data = np.array(image, dtype=np.float32)

        # Normalize to reasonable elevation range (0-1000m by default)
        elevation_range = kwargs.get("elevation_range", (0, 1000))
        elevation_data = (elevation_data - elevation_data.min()) / (
            elevation_data.max() - elevation_data.min()
        ) * (elevation_range[1] - elevation_range[0]) + elevation_range[0]

        # Assume image dimensions represent spatial extent
        width_m = kwargs.get("width_m", elevation_data.shape[1])
        height_m = kwargs.get("height_m", elevation_data.shape[0])

        metadata = {
            "source": "image",
            "original_image_size": image.size,
            "elevation_range": elevation_range,
        }

        return cls(elevation_data, extent=(width_m, height_m), metadata=metadata)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        extent: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DEM":
        """
        Create DEM from numpy array.

        Parameters
        ----------
        data : np.array
            Elevation data array (2D) or XYZ coordinates array (3D).
        extent : tuple, optional
            Spatial extent of the data.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        DEM
            DEM object.
        """
        return cls(data, extent, metadata)

    def get_xyz_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get X, Y, Z coordinate arrays.

        Returns
        -------
        tuple
            (x_coords, y_coords, z_coords) where each is a (N, M) array.
        """
        return self.x_coords, self.y_coords, self.z_coords

    def get_elevation(self) -> np.ndarray:
        """
        Get elevation data (Z coordinates).

        Returns
        -------
        np.ndarray
            Elevation data array of shape (N, M).
        """
        return self.z_coords

    def get_xyz_flat(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get flattened X, Y, Z coordinate arrays.

        Returns
        -------
        tuple
            (x_coords, y_coords, z_coords) where each is a 1D array of length N*M.
        """
        return (self.x_coords.flatten(), self.y_coords.flatten(), self.z_coords.flatten())

    def get_xyz_combined(self) -> np.ndarray:
        """
        Get combined X, Y, Z array of shape (N*M, 3).

        Returns
        -------
        np.ndarray
            Array of shape (N*M, 3) containing [x, y, z] coordinates.
        """
        return np.stack(
            [self.x_coords.flatten(), self.y_coords.flatten(), self.z_coords.flatten()], axis=1
        )

    def handle_missing_data(self, missing_value: float = -32767.0):
        """
        Handle missing data entries in the DEM.

        Parameters
        ----------
        missing_value : float
            Value representing missing data.
        """
        missing = self.z_coords == missing_value
        if np.any(missing):
            min_val = np.min(self.z_coords[~missing])
            self.z_coords[missing] = min_val
            # Update the data array as well
            self.data[:, :, 2] = self.z_coords

    def downsample(self, factor: int) -> "DEM":
        """
        Downsample the DEM data by a factor.

        Parameters
        ----------
        factor : int
            Factor to downsample the DEM data by.

        Returns
        -------
        DEM
            Downsampled DEM object.
        """
        data_ds = self.data[::factor, ::factor, :]
        new_extent = (self.extent[0] / factor, self.extent[1] / factor)
        dem = DEM(data_ds, extent=new_extent, metadata=self.metadata)
        return dem

    def query(self, x: float, y: float) -> float:
        """
        Query elevation at given xy point using bilinear interpolation.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.

        Returns
        -------
        float
            Interpolated elevation value.
        """
        # Find grid indices
        x_idx = (x / self.extent[0]) * (self.M - 1)
        y_idx = (y / self.extent[1]) * (self.N - 1)

        # Bilinear interpolation
        x0, y0 = int(x_idx), int(y_idx)
        x1, y1 = min(x0 + 1, self.M - 1), min(y0 + 1, self.N - 1)

        # Get corner values from z_coords
        z00 = self.z_coords[y0, x0]
        z01 = self.z_coords[y0, x1]
        z10 = self.z_coords[y1, x0]
        z11 = self.z_coords[y1, x1]

        # Interpolation weights
        wx = x_idx - x0
        wy = y_idx - y0

        # Bilinear interpolation
        z = z00 * (1 - wx) * (1 - wy) + z01 * wx * (1 - wy) + z10 * (1 - wx) * wy + z11 * wx * wy

        return z

    def show(self):
        """
        Display a heatmap visualization of the DEM.
        """
        fig = plot_heatmap(self.z_coords)
        return fig

    def surface_plot(self):
        """
        Plot the surface of the DEM.
        """
        if self.N > 500:
            downsample_factor = self.N // 500
            downsampled_data = self.data[::downsample_factor, ::downsample_factor]
            print(f"Downsampling DEM from {self.shape} to {downsampled_data.shape} for plotting")
            fig = plot_surface(downsampled_data)
        else:
            fig = plot_surface(self.data)
        return fig

    def __repr__(self):
        return f"DEM(shape=({self.N}, {self.M}, 3), extent={self.extent}, source={self.metadata.get('source', 'unknown')})"
