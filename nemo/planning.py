import torch

from nemo.util import wrap_angle_torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# TODO TODO TODO
def integrate_path(path):
    """Given path and dynamics, integrate to get full trajectory"""
    pass


# Initial Dubin's version - optimize over (x,y), estimate theta
def path_optimization(nemo, path_xy, iterations=500, lr=1e-3):
    """
    Refine a path by adding points in between the original points

    nemo : height field
    path_xy : initial 2D path (torch tensor)
    iterations : number of optimization iterations
    lr : learning rate
    (TODO) tol : tolerance for convergence 

    """
    path_start = path_xy[0]
    path_end = path_xy[-1]
    path_opt = path_xy[1:-1].clone().detach().requires_grad_(True)  # portion of the path to optimize
    path = torch.cat((path_start[None], path_opt, path_end[None]), dim=0)  # full path

    # Dubin's based cost
    def cost(path, dt=0.1):
        thetas = torch.atan2(path[1:,1] - path[:-1,1], path[1:,0] - path[:-1,0])  # Compute path thetas
        omegas = wrap_angle_torch(thetas[1:] - thetas[:-1]) / dt  # Omegas as wrapped difference
        # Path Vs
        path_dxy = torch.diff(path, dim=0)
        Vs = torch.norm(path_dxy, dim=1) / dt
        controls_cost = 10 * torch.mean(Vs**2) + torch.mean(omegas**2)
        # Slope cost
        path_zs = 10 * nemo.get_heights(path)
        slope_cost = torch.mean(torch.abs(path_zs[1:] - path_zs[:-1]))
        return controls_cost + slope_cost

    # Optimize path
    opt = torch.optim.Adam([path_opt], lr=lr)

    for it in range(iterations):
        opt.zero_grad()
        path = torch.cat((path_start[None], path_opt, path_end[None]), dim=0)
        c = cost(path)
        c.backward()
        opt.step()
        if it % 50 == 0:
            print(f'it: {it},  Cost: {c.item()}')

    print(f'Finished optimization - final cost: {c.item()}')

    # Compute final heights
    path_zs = nemo.get_heights(path)

    # Full 3D path
    path_3d = torch.cat((path, path_zs), dim=1)

    return path_3d