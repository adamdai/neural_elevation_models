import torch
import torch.nn as nn
import numpy as np


class FourierEncoding(nn.Module):
    """
    Fourier features encoding from "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    (https://arxiv.org/abs/2006.10739)
    
    This encoding creates fixed random frequencies and applies sine/cosine transformations.
    The B matrix is fixed (not trainable) to maintain the frequency structure.
    
    Args:
        in_dim (int): Input dimension (default: 2 for x,y coordinates)
        num_frequencies (int): Number of Fourier features (default: 256)
        scale (float): Scale factor for the random frequencies (default: 10.0)
    
    Example:
        # Basic usage
        encoder = FourierEncoding(in_dim=2, num_frequencies=256, scale=10.0)
        
        # With custom parameters
        encoder = FourierEncoding(in_dim=2, num_frequencies=512, scale=20.0)
    """
    
    def __init__(self, in_dim=2, num_frequencies=256, scale=10.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.scale = scale
        
        # Create fixed B matrix for Fourier features - this should NOT change during training
        # B matrix of shape [num_frequencies, in_dim]
        self.register_buffer('B', scale * torch.randn(num_frequencies, in_dim))
    
    def forward(self, inputs):
        """
        Apply Fourier features encoding to input coordinates.
        
        Args:
            inputs: Input tensor of shape [..., in_dim]
            
        Returns:
            Encoded tensor of shape [..., 2*num_frequencies]
        """
        # Compute 2π * inputs @ B.T using the fixed B matrix
        projected = 2 * np.pi * inputs @ self.B.T
        
        # Return [sin(projected), cos(projected)]
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
    
    def get_out_dim(self):
        """Get the output dimension of the encoding."""
        return 2 * self.num_frequencies
