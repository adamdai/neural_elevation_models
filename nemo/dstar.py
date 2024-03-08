"""Implementation of Field D* algorithm for path planning in 2D grid maps.

"""

"""Global planners

"""

import numpy as np

from nemo.astar import AStar



class DStar():

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
        
        """
        path = list(self.astar(start, goal))  # path is in image coordinates
        dt = np.dtype('int32','int32')
        path = np.array(path, dtype=dt)[:,[1,0]]  # flip x and y to get back to spatial coordinates
        self.path = path  
        return path
    

    def plot(self, ax):
        """Plot costmap and path"""
        im = ax.imshow(self.costmat, cmap='viridis', origin='lower')
        if self.path is not None:
            ax.plot(self.path[:,0], self.path[:,1], 'r')
            ax.scatter(self.path[:,0], self.path[:,1], c='r', s=10, marker='*')
        # ax.set_xticks(np.arange(xmin, xmax, 50))
        # ax.set_yticks(np.arange(ymin, ymax, 50))
        # Invert x axis
        #ax.invert_xaxis()
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        return im

