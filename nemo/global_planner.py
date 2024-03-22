"""Global planners

"""

import numpy as np

from nemo.astar import AStar



class AStarPlanner(AStar):
    """Basic A* planner for a costmap
    
    - Neighbors are 4-connected
    
    """

    def __init__(self, costmat):
        """
        Parameters
        ----------
        costmap : CostMap

        
        """
        self.costmat = costmat
        self.width = self.costmat.shape[0]
        self.height = self.costmat.shape[1]

        self.path = None

    def neighbors(self, node):
        x, y = node
        return [(nx, ny) for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] 
                if 0 <= nx < self.width and 0 <= ny < self.height]

    def distance_between(self, node1, node2):
        return self.costmat[node2]
    
    def heuristic_cost_estimate(self, node1, node2):
        """Straight line distance"""
        return np.linalg.norm(np.array(node1) - np.array(node2))
    
    def plan(self, start, goal):
        """Find a path from start to goal
        
        Start and goal are in image coordinates
        Output path is in spatial index coordinates (indices of XY grid)
        
        """
        path = list(self.astar(start, goal))  # path is in image coordinates
        dt = np.dtype('int32','int32')
        path = np.array(path, dtype=dt)[:,[1,0]]  # flip x and y to get back to spatial coordinates
        self.path = path  
        return path
    


class AStarGradPlanner(AStar):
    """A* planner for a heightmap with gradient-based costs
    
    - Neighbors are 8-connected
    
    """

    def __init__(self, heightmat, bounds):
        """
        Parameters
        ----------
        heightmat : np.ndarray
            2D array of heights
        bounds : tuple
            (xmin, xmax, ymin, ymax)

        
        """
        self.heightmat = heightmat
        self.width = self.heightmat.shape[0]
        self.height = self.heightmat.shape[1]
        self.bounds = bounds

    def spatial_to_grid(self, pos):
        """Convert spatial coordinates to grid coordinates
        
        Parameters
        ----------
        pos : tuple or np.ndarray
            Single spatial coordinate or array of spatial coordinates
        
        """
        xmin, xmax, ymin, ymax = self.bounds
        if isinstance(pos, tuple):
            pos = np.array(pos).reshape(-1, 2)
        x, y = pos[:,0], pos[:,1]
        i = ((y - ymin) / (ymax - ymin) * self.width).astype(int)
        j = ((x - xmin) / (xmax - xmin) * self.height).astype(int)
        return np.vstack((i, j)).T
    
    def grid_to_spatial(self, idx):
        """Convert grid coordinates to spatial coordinates
        
        Parameters
        ----------
        idx : tuple or np.ndarray
            Single grid coordiante or array of grid coordinates
        
        """
        xmin, xmax, ymin, ymax = self.bounds
        if isinstance(idx, tuple):
            idx = np.array(idx).reshape(-1, 2)
        i, j = idx[:,0], idx[:,1]
        x = i / self.width * (xmax - xmin) + xmin
        y = j / self.height * (ymax - ymin) + ymin
        return np.vstack((x, y)).T

    def in_bounds(self, pos):
        """Check if a position is in bounds"""
        x, y = pos
        xmin, xmax, ymin, ymax = self.bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def neighbors(self, node):
        x, y = node
        return [(nx, ny) for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1), 
                                        (x+1, y+1), (x-1, y+1), (x+1, y-1), (x-1, y-1)] 
                if 0 <= nx < self.width and 0 <= ny < self.height]

    def distance_between(self, node1, node2):
        return (self.heightmat[node2] - self.heightmat[node1])**2
    
    def heuristic_cost_estimate(self, node1, node2):
        """Straight line distance"""
        return np.linalg.norm(np.array(node1) - np.array(node2))
    
    def grid_plan(self, start_idx, goal_idx):
        """Find a path from start to goal in grid coordinates

        Parameters
        ----------
        start_idx : tuple
            Start index
        goal_idx : tuple
            Goal index
        
        Returns
        -------
        path_idxs : np.ndarray
            (N, 2) array of path indices
        
        """
        path_idxs = list(self.astar(start_idx, goal_idx))  # path is in image coordinates
        dt = np.dtype('int32','int32')
        path_idxs = np.array(path_idxs, dtype=dt)[:,[1,0]]  # flip x and y to get back to grid coordinates
        return path_idxs


    def spatial_plan(self, start, goal):
        """Find a path from start to goal in spatial coordinates
        
        Parameters
        ----------
        start : np.ndarray
            (2,) array of start position
        goal : np.ndarray
            (2,) array of goal position
        
        Returns
        -------
        path : np.ndarray
            (N, 2) array of path positions
        
        """
        # Check if start and goal are in bounds
        assert self.in_bounds(start), "Start position not in bounds"
        assert self.in_bounds(goal), "Goal position not in bounds"

        # Convert and plan in grid coordinates
        start_idx = self.spatial_to_grid(start)
        start_idx = tuple(start_idx.flatten())
        goal_idx = self.spatial_to_grid(goal)
        goal_idx = tuple(goal_idx.flatten())
        path_idxs = self.grid_plan(start_idx, goal_idx)

        # Convert back to spatial coordinates
        path = self.grid_to_spatial(path_idxs)
        return path



