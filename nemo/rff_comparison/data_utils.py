import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path


class HeightFieldData:
    """
    Data loading and preprocessing for height field evaluation.
    
    Handles loading moon DEM data, normalization, and train/test splitting.
    """
    
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        Initialize data loader.
        
        Args:
            data_path (str): Path to the moon DEM data file
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.normalization_params = {}
        
    def load_moon_data(self):
        """
        Load and preprocess moon DEM data.
        
        Returns:
            tuple: (train_xyz, test_xyz, normalization_params)
                - train_xyz: torch.Tensor of shape (N_train, 3) with (x, y, z) coordinates
                - test_xyz: torch.Tensor of shape (N_test, 3) with (x, y, z) coordinates  
                - normalization_params: dict with normalization parameters for denormalization
        """
        print(f"Loading moon data from: {self.data_path}")
        
        # Load the DEM data
        dem = np.load(self.data_path, allow_pickle=True)
        print(f"DEM shape: {dem.shape}")
        print(f"DEM height range: [{dem[:,:,2].min():.3f}, {dem[:,:,2].max():.3f}]")
        
        # Reshape from (H, W, 3) to (N, 3) where N = H * W
        dem_reshaped = dem[:, :, :3].reshape(-1, 3)  # Flatten spatial dimensions
        print(f"Reshaped to: {dem_reshaped.shape}")
        
        # Split into coordinates and heights
        xy = dem_reshaped[:, :2]  # (x, y) coordinates
        z = dem_reshaped[:, 2]    # height values
        
        # Normalize coordinates to [0, 1] range
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        xy_normalized = (xy - xy_min) / (xy_max - xy_min)
        
        # Normalize heights to zero mean and unit variance
        z_mean = z.mean()
        z_std = z.std()
        z_normalized = (z - z_mean) / z_std
        
        # Store normalization parameters for later denormalization
        self.normalization_params = {
            'xy_min': xy_min,
            'xy_max': xy_max,
            'z_mean': z_mean,
            'z_std': z_std
        }
        
        # Combine normalized coordinates and heights
        xyz_normalized = np.column_stack([xy_normalized, z_normalized])
        
        # Split into train and test sets
        train_xyz, test_xyz = train_test_split(
            xyz_normalized, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Convert to torch tensors
        train_xyz = torch.from_numpy(train_xyz).float()
        test_xyz = torch.from_numpy(test_xyz).float()
        
        print(f"Train set: {train_xyz.shape[0]} points")
        print(f"Test set: {test_xyz.shape[0]} points")
        print(f"Normalized height range: [{train_xyz[:, 2].min():.3f}, {train_xyz[:, 2].max():.3f}]")
        
        return train_xyz, test_xyz, self.normalization_params
    
    def denormalize_heights(self, normalized_heights):
        """
        Denormalize height predictions back to original scale.
        
        Args:
            normalized_heights (torch.Tensor): Height predictions in normalized scale
            
        Returns:
            torch.Tensor: Heights in original scale
        """
        return normalized_heights * self.normalization_params['z_std'] + self.normalization_params['z_mean']
    
    def denormalize_coordinates(self, normalized_xy):
        """
        Denormalize coordinates back to original scale.
        
        Args:
            normalized_xy (torch.Tensor): Coordinates in [0, 1] range
            
        Returns:
            torch.Tensor: Coordinates in original scale
        """
        xy_min = self.normalization_params['xy_min']
        xy_max = self.normalization_params['xy_max']
        return normalized_xy * (xy_max - xy_min) + xy_min
    
    def get_data_info(self):
        """
        Get information about the loaded data.
        
        Returns:
            dict: Information about the data
        """
        return {
            'data_path': str(self.data_path),
            'test_size': self.test_size,
            'random_state': self.random_state,
            'normalization_params': self.normalization_params
        }


def load_moon_data_for_comparison(data_path="data/Moon_Map_01_0_rep0.dat", test_size=0.2):
    """
    Convenience function to load moon data for comparison experiments.
    
    Args:
        data_path (str): Path to moon DEM data
        test_size (float): Fraction of data for testing
        
    Returns:
        tuple: (train_xyz, test_xyz, normalization_params)
    """
    data_loader = HeightFieldData(data_path, test_size=test_size)
    return data_loader.load_moon_data()


if __name__ == "__main__":
    # Test the data loading
    train_xyz, test_xyz, norm_params = load_moon_data_for_comparison()
    
    print("\nData loading test completed successfully!")
    print(f"Train set shape: {train_xyz.shape}")
    print(f"Test set shape: {test_xyz.shape}")
    print(f"Normalization params: {norm_params}")
