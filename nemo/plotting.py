"""Plotting functions

"""


import plotly.graph_objects as go


##### ------------------- PLOTTING ------------------- #####

def plot_surface(fig, x, y, z, colorscale='Viridis', no_axes=False):
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=colorscale))
    fig.update_layout(width=1200, height=900, scene_aspectmode='data')
    if no_axes:
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    return fig


def plot_path_3d(fig, x, y, z, color='red', markersize=3, linewidth=3):
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=markersize, color=color),
                           line=dict(color=color, width=linewidth)))
    return fig


def plot_3d_points(fig, x, y, z, color='blue', markersize=3):
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=markersize, color=color)))
    fig.update_layout(width=1200, height=900, scene_aspectmode='data')
    return fig