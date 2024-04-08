"""Geometry utils

"""

import numpy as np


def fit_plane(points):
    """
    Fit a plane to a set of points

    Parameters
    ----------
    points: np.ndarray (N, 3)

    Returns
    -------
    plane: np.ndarray (4,)
    """
    assert points.shape[1] == 3
    A = np.c_[points[:, :2], np.ones(points.shape[0])]
    B = points[:, 2]
    plane = np.linalg.lstsq(A, B, rcond=None)[0]
    return plane