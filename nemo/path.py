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
    


class CubicSplinePath:
    def __init__(self, waypoints, init_heading):
        """
        """
        self.waypoints = waypoints
        self.init_heading = init_heading
        self.n = len(waypoints) - 1
        self.h = np.diff(waypoints[:, 0])
        self.b = np.diff(waypoints[:, 1]) / self.h
        self.coeffs = self.compute_coeffs()

    def eval(self, t):
        """
        """
        spline_idx = np.searchsorted(self.waypoints[:, 0], t, side='right') - 1
        spline_idx = np.clip(spline_idx, 0, self.n-1)
        t = (t - self.waypoints[spline_idx, 0]) / self.h[spline_idx]
        return self.coeffs[spline_idx] @ np.array([1, t, t**2, t**3])

    def compute_coeffs(self):
        A, B = self.compute_AB()
        c = np.linalg.solve(A, B)
        a = self.waypoints[:-1, 1]
        b = self.b - self.h * (2*c[:-1] + c[1:]) / 3
        d = (c[1:] - c[:-1]) / (3*self.h)
        c = c[:-1]
        return np.vstack((a, b, c, d)).T
    
    def compute_AB(self):
        A = np.zeros((self.n+1, self.n+1))
        B = np.zeros(self.n+1)

        # Initial heading condition
        initial_slope = np.tan(self.init_heading)
        A[0, 0] = 2 * self.h[0]
        A[0, 1] = self.h[0]
        B[0] = 3 * (self.b[0] - initial_slope)

        # Continuity conditions
        for i in range(1, self.n):
            A[i, i-1] = self.h[i-1]
            A[i, i] = 2 * (self.h[i-1] + self.h[i])
            A[i, i+1] = self.h[i]
            B[i] = 3 * (self.b[i] - self.b[i-1])

        # Not-a-knot end condition at the last point
        A[self.n, self.n-1] = self.h[-1]
        A[self.n, self.n] = 2 * self.h[-1]
        B[self.n] = 0

        return A, B

def cubic_spline_coefficients(waypoints, initial_heading):
    n = len(waypoints) - 1  # Number of splines

    h = np.diff(waypoints[:, 0])
    b = np.diff(waypoints[:, 1]) / h

    A = np.zeros((n+1, n+1))
    B = np.zeros(n+1)

    # Initial heading condition
    initial_slope = np.tan(initial_heading)
    A[0, 0] = 2 * h[0]
    A[0, 1] = h[0]
    B[0] = 3 * (b[0] - initial_slope)

    # Continuity conditions
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 3 * (b[i] - b[i-1])

    # Not-a-knot end condition at the last point
    A[n, n-1] = h[-1]
    A[n, n] = 2 * h[-1]
    B[n] = 0  # Natural spline end condition

    # Solve for c
    c = np.linalg.solve(A, B)

    a = waypoints[:-1, 1]
    b = b - h * (2*c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3*h)
    c = c[:-1]

    coefficients = np.vstack((a, b, c, d)).T
    return coefficients