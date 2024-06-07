import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

g = 9.81  # gravity

# Dynamics model
def dubins_traj(x0, U):
    vdt = 0.01
    traj = torch.zeros(len(U), 3)
    x = x0
    for i, u in enumerate(U):
        x[0] += vdt * torch.cos(x[2])
        x[1] += vdt * torch.sin(x[2])
        x[2] += u
        traj[i] = x
    return traj



def compute_slopes_sample(xy, theta, nemo, l):
    """Compute slopes through sampling"""
    dl = l/2 * torch.stack((torch.cos(theta), torch.sin(theta))).T 
    z_front = nemo.get_heights(xy + dl).float()
    z_back = nemo.get_heights(xy - dl).float()
    phi = torch.arctan2(z_front - z_back, torch.tensor(l))
    return phi


def compute_slopes_plane_fit(xy, nemo):
    """Compute slopes through plane fitting"""
    pass


# TODO: test version with cos(\phi) term


def diff_flatness(xy, nemo, dt):
    x = xy[:, 0]
    y = xy[:, 1]
    epsilon = torch.tensor(1e-5, device=device, requires_grad=True)
    xdot = torch.hstack((epsilon, torch.diff(x) / dt))
    ydot = torch.hstack((epsilon, torch.diff(y) / dt))
    xddot = torch.hstack((epsilon, torch.diff(xdot) / dt))
    yddot = torch.hstack((epsilon, torch.diff(ydot) / dt))
    v = torch.sqrt(xdot**2 + ydot**2)
    theta = torch.arctan2(ydot, xdot)
    
    _, grad = nemo.get_heights_with_grad(xy.clone().requires_grad_(True))
    psi = torch.atan2(grad[:,1], grad[:,0])
    alpha = torch.atan(grad.norm(dim=1))

    phi = alpha * torch.cos(theta - psi)
    g_eff = 9.81 * torch.sin(phi)

    u = torch.zeros(len(x), 2)
    for i in range(len(x)):
        J = torch.tensor([[torch.cos(theta[i]), -v[i] * torch.sin(theta[i])],
                          [torch.sin(theta[i]), v[i] * torch.cos(theta[i])]], device=device, requires_grad=True)
        b = torch.tensor([[xddot[i] + g_eff[i] * torch.cos(theta[i])],
                          [yddot[i] + g_eff[i] * torch.sin(theta[i])]], device=device, requires_grad=True)
        u[i] = torch.linalg.solve(J, b).flatten()

    return u