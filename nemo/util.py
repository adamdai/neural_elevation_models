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