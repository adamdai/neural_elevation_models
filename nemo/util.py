import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_PI = torch.tensor(np.pi, device=device)

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def wrap_angle_torch(angle):
    """Wrap angle to [-pi, pi] range"""
    return ((angle + TORCH_PI) % (2 * TORCH_PI)) - TORCH_PI


def path_metrics(path):
    """
    Parameters
    ---------
    path: torch tensor (N, 3)
    """
    print(f'Path length: {len(path)}')

    # Calculate the length of the path
    path_length_2d = 0
    path_length_3d = 0
    path_xy = path[:,:2]
    for i in range(len(path)-1):
        path_length_2d += torch.norm(path_xy[i+1] - path_xy[i])
        path_length_3d += torch.norm(path[i+1] - path[i])
    print(f'2D distance: {path_length_2d}')
    print(f'3D distance: {path_length_3d}')

    # Calculate the sum of absolute height deltas along the path
    height_diff = 0
    path_zs = path[:,2]
    for i in range(len(path)-1):
        height_diff += torch.abs(path_zs[i+1] - path_zs[i])
    print(f'Height delta sum: {height_diff.item()}')