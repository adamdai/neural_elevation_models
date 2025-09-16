"""Plotting functions"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

##### ------------------- 2D ------------------- #####


def plot_heatmap(data, fig=None, colorscale="Viridis", no_axes=False):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Heatmap(z=data, colorscale=colorscale))
    fig.update_layout(width=1200, height=900)
    if no_axes:
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


##### ------------------- 3D ------------------- #####


def plot_surface(
    grid: np.ndarray, fig=None, colorscale="Viridis", no_axes=False, showscale=True, **kwargs
):
    """
    grid is NxNx3 array representing the coordinates and elevation data for a surface plot.

    """
    if fig is None:
        fig = go.Figure()
    # Downsample large grids to 500x500 for plotting
    M, N = grid.shape[:2]
    if M > 500 or N > 500:
        downsample_factor_M = max(1, M // 500)
        downsample_factor_N = max(1, N // 500)
        grid = grid[::downsample_factor_M, ::downsample_factor_N, :]
        print(f"Downsampling grid from {(M, N)} to {grid.shape[:2]} for plotting")
    fig.add_trace(
        go.Surface(
            x=grid[:, :, 0],
            y=grid[:, :, 1],
            z=grid[:, :, 2],
            colorscale=colorscale,
            showscale=showscale,
            **kwargs,
        )
    )
    fig.update_layout(width=1200, height=800, scene_aspectmode="data")
    if no_axes:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
            )
        )
    return fig


def plot_path_3d(
    x, y, z, fig=None, color="red", markersize=3, linewidth=3, markers=True, name=None, **kwargs
):
    if fig is None:
        fig = go.Figure()
    if markers:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+lines",
                marker=dict(size=markersize, color=color),
                line=dict(color=color, width=linewidth),
                name=name,
                **kwargs,
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=color, width=linewidth),
                name=name,
                **kwargs,
            )
        )
    return fig


def plot_3d_points(x, y, z, fig=None, color="blue", markersize=3):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=markersize, color=color))
    )
    fig.update_layout(width=1200, height=900, scene_aspectmode="data")
    return fig


def plot_rotating_surface(x, y, z, fig=None):
    fig = plot_surface(x, y, z, no_axes=True, showscale=False)
    fig.update_layout(width=1600, height=900)
    fig.show()

    x_eye = 1.5
    y_eye = -1
    z_eye = 1

    fig.update_layout(
        title="Animation Test",
        width=1600,
        height=900,
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1,
                x=0.8,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=5, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
    )

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    frames = []
    for t in np.arange(0, 6.26, 0.025):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    fig.frames = frames

    fig.show()
