"""Plotting functions

"""

import plotly.graph_objects as go
import plotly.express as px

##### ------------------- 2D ------------------- #####

def plot_heatmap(data, fig=None, colorscale='Viridis', no_axes=False):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Heatmap(z=data, colorscale=colorscale))
    fig.update_layout(width=1200, height=900)
    if no_axes:
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


##### ------------------- 3D ------------------- #####

def plot_surface(x, y, z, fig=None, colorscale='Viridis', no_axes=False, showscale=True):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=colorscale, showscale=showscale))
    fig.update_layout(width=1200, height=900, scene_aspectmode='data')
    if no_axes:
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    return fig


def plot_path_3d(x, y, z, fig=None, color='red', markersize=3, linewidth=3):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=markersize, color=color),
                           line=dict(color=color, width=linewidth)))
    return fig


def plot_3d_points(x, y, z, fig=None, color='blue', markersize=3):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=markersize, color=color)))
    fig.update_layout(width=1200, height=900, scene_aspectmode='data')
    return fig