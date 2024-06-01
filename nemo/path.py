import numpy as np
import scipy.interpolate as spi


class PolyPath:
    def __init__(self, d, n, start, end, free_coeffs=None):
        """
        Constrain endpoints, zero velocity at endpoints.

        d : int
            Dimension of the path
        n : int
            Degree of the polynomial
        free_coeffs : np.ndarray (n+1-4, d)
            Free coefficients. The first 4 coefficients are constrained
            
        """
        self.d = d
        self.n = n
        self.exponents = np.arange(n+1)
        
        # Compute full coefficients
        self.coeffs = np.zeros((n+1, d))
        self.coeffs[4:] = free_coeffs
        self.coeffs[3] = 2 * (start - end) - free_coeffs.T @ np.arange(2, n-1)
        self.coeffs[2] = (end - start) - self.coeffs[3] - free_coeffs.T @ np.ones(n-3)
        self.coeffs[1] = 0
        self.coeffs[0] = start
        

    def eval(self, t):
        N = len(t)
        T = t[:, None] ** self.exponents
        return T @ self.coeffs