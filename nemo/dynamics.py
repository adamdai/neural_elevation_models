import torch


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



# x0 = torch.zeros(3)
# U = 0.1 * (torch.rand(1000) - 0.5)
# traj = dubins_traj(x0, U)

# # Plot the traj
# fig = go.Figure(data=go.Scatter(x=traj[:,0].cpu().numpy(), y=traj[:,1].cpu().numpy()))
# fig.show()