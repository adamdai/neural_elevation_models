"""Digital Elevation Model (DEM) class and functions.

"""

import numpy as np

from nemo.plotting import plot_heatmap


class DEM:
    """
    
    Attributes
    ----------
    data : np.array (N, M)
        2D array of Elevation data.
    N, M : int
        Dimensions of DEM data.
    extent : tuple (x, y)
        Extent in x and y of DEM data (meters).

    Methods
    -------
    downsample
        Downsample the DEM data by a factor.
    query
        Query elevation at given xy point.

    """
    def __init__(self, data, extent=None):
        """
        
        data : np.array (N, M)
            2D array of Elevation data.

        """
        self.data = data
        self.N, self.M = data.shape

        if extent is None:
            # Assume one cell is 1 unit in x and y
            self.extent = (self.M, self.N)
        else:
            self.extent = extent


    def handle_missing_data(self):
        """
        Handle missing data entries in the DEM tif file.

        """
        missing = (self.data == -32767.0)
        min_val = np.min(self.data[~missing])
        self.data[missing] = min_val


    def downsample(self, factor):
        """
        Downsample the DEM data by a factor.

        Parameters
        ----------
        factor : int
            Factor to downsample the DEM data by.

        Returns
        -------
        DEM
            downsampled DEM object.

        """
        data = self.data[::factor, ::factor]
        return DEM(data)
    

    def show(self):
        """
        Display a heatmap visualization of the DEM.

        """
        fig = plot_heatmap(self.data)
        fig.show()
    
