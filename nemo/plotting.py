from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from nemo.dem import DEM
from nemo.nemo import Nemo


def _subsample_grid(dem: DEM, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stride = max(int(stride), 1)
    x = dem.x[::stride, ::stride]
    y = dem.y[::stride, ::stride]
    z = dem.z[::stride, ::stride]
    return x, y, z


def _batched_nemo_eval(
    nemo: Nemo, xy: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    grads: list[np.ndarray] = []
    device = nemo.device
    for start in range(0, len(xy), batch_size):
        batch = torch.as_tensor(xy[start : start + batch_size], dtype=torch.float32, device=device)
        pred = nemo.field.h(batch).detach().cpu().numpy()
        grad = nemo.field.grad(batch).detach().cpu().numpy()
        preds.append(pred)
        grads.append(grad)
    return np.concatenate(preds, axis=0), np.concatenate(grads, axis=0)


def _heatmap_colorbar(fig: go.Figure, xaxis_name: str, yaxis_name: str) -> dict[str, object]:
    xaxis = getattr(fig.layout, xaxis_name)
    yaxis = getattr(fig.layout, yaxis_name)
    _, x1 = xaxis.domain
    y0, y1 = yaxis.domain
    return {
        "x": min(x1 + 0.002, 1.06),
        "y": (y0 + y1) / 2.0,
        "len": max((y1 - y0) * 0.9, 0.1),
        "thickness": 16,
        "xanchor": "left",
        "yanchor": "middle",
    }


def evaluate_dem_fit(
    dem: DEM,
    nemo: Nemo,
    *,
    stride: int = 1,
    batch_size: int = 65536,
) -> dict[str, np.ndarray]:
    x, y, z_dem = _subsample_grid(dem, stride)
    xy = np.column_stack([x.reshape(-1), y.reshape(-1)])
    z_pred_flat, grad_pred_flat = _batched_nemo_eval(nemo, xy, batch_size=batch_size)

    z_pred = z_pred_flat.reshape(z_dem.shape)
    grad_pred = grad_pred_flat.reshape(*z_dem.shape, 2)

    gx_dem = dem.gx[::stride, ::stride]
    gy_dem = dem.gy[::stride, ::stride]
    gx_pred = grad_pred[..., 0]
    gy_pred = grad_pred[..., 1]

    z_error = z_pred - z_dem
    gx_error = gx_pred - gx_dem
    gy_error = gy_pred - gy_dem
    grad_mag_error = np.sqrt(gx_error**2 + gy_error**2)

    return {
        "x": x,
        "y": y,
        "z_dem": z_dem,
        "z_pred": z_pred,
        "gx_dem": gx_dem,
        "gy_dem": gy_dem,
        "gx_pred": gx_pred,
        "gy_pred": gy_pred,
        "z_error": z_error,
        "gx_error": gx_error,
        "gy_error": gy_error,
        "grad_mag_error": grad_mag_error,
    }


def create_dem_fit_figure(
    dem: DEM,
    nemo: Nemo,
    *,
    stride: int = 1,
    batch_size: int = 65536,
    surface_showscale: bool = False,
) -> go.Figure:
    fields = evaluate_dem_fit(dem, nemo, stride=stride, batch_size=batch_size)
    x = fields["x"]
    y = fields["y"]
    z_dem = fields["z_dem"]
    z_pred = fields["z_pred"]
    gx_dem = fields["gx_dem"]
    gy_dem = fields["gy_dem"]
    gx_pred = fields["gx_pred"]
    gy_pred = fields["gy_pred"]
    z_error = fields["z_error"]
    gx_error = fields["gx_error"]
    gy_error = fields["gy_error"]
    grad_mag_dem = np.sqrt(gx_dem**2 + gy_dem**2)
    grad_mag_pred = np.sqrt(gx_pred**2 + gy_pred**2)
    grad_mag_error = fields["grad_mag_error"]

    z_absmax = float(np.nanmax(np.abs(z_error))) if np.isfinite(z_error).any() else 1.0
    grad_mag_absmax = float(np.nanmax(grad_mag_error)) if np.isfinite(grad_mag_error).any() else 1.0
    gx_absmax = float(np.nanmax(np.abs(gx_error))) if np.isfinite(gx_error).any() else 1.0
    gy_absmax = float(np.nanmax(np.abs(gy_error))) if np.isfinite(gy_error).any() else 1.0
    z_absmax = max(z_absmax, 1e-8)
    grad_mag_absmax = max(grad_mag_absmax, 1e-8)
    gx_absmax = max(gx_absmax, 1e-8)
    gy_absmax = max(gy_absmax, 1e-8)

    fig = make_subplots(
        rows=5,
        cols=3,
        specs=[
            [{"type": "surface"}, {"type": "surface"}, None],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
        ],
        subplot_titles=(
            "DEM Surface",
            "NEMo Surface",
            "DEM Height",
            "NEMo Height",
            "Height Error",
            "DEM Gradient Magnitude",
            "NEMo Gradient Magnitude",
            "Gradient Error Magnitude",
            "DEM X Gradient",
            "NEMo X Gradient",
            "X Gradient Error",
            "DEM Y Gradient",
            "NEMo Y Gradient",
            "Y Gradient Error",
        ),
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
    )

    fig.add_trace(
        go.Surface(x=x, y=y, z=z_dem, colorscale="Viridis", showscale=surface_showscale),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(x=x, y=y, z=z_pred, colorscale="Viridis", showscale=surface_showscale),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=z_dem,
            x=x[0, :],
            y=y[:, 0],
            colorscale="Viridis",
            colorbar=_heatmap_colorbar(fig, "xaxis", "yaxis"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=z_pred,
            x=x[0, :],
            y=y[:, 0],
            colorscale="Viridis",
            colorbar=_heatmap_colorbar(fig, "xaxis2", "yaxis2"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=z_error,
            x=x[0, :],
            y=y[:, 0],
            colorscale="RdBu",
            zmid=0.0,
            zmin=-z_absmax,
            zmax=z_absmax,
            colorbar=_heatmap_colorbar(fig, "xaxis3", "yaxis3"),
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Heatmap(
            z=grad_mag_dem,
            x=x[0, :],
            y=y[:, 0],
            colorscale="Magma",
            colorbar=_heatmap_colorbar(fig, "xaxis4", "yaxis4"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=grad_mag_pred,
            x=x[0, :],
            y=y[:, 0],
            colorscale="Magma",
            colorbar=_heatmap_colorbar(fig, "xaxis5", "yaxis5"),
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=grad_mag_error,
            x=x[0, :],
            y=y[:, 0],
            colorscale="Inferno",
            zmin=0.0,
            zmax=grad_mag_absmax,
            colorbar=_heatmap_colorbar(fig, "xaxis6", "yaxis6"),
        ),
        row=3,
        col=3,
    )
    fig.add_trace(
        go.Heatmap(
            z=gx_dem,
            x=x[0, :],
            y=y[:, 0],
            colorscale="RdBu",
            zmid=0.0,
            colorbar=_heatmap_colorbar(fig, "xaxis7", "yaxis7"),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=gx_pred,
            x=x[0, :],
            y=y[:, 0],
            colorscale="RdBu",
            zmid=0.0,
            colorbar=_heatmap_colorbar(fig, "xaxis8", "yaxis8"),
        ),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=gx_error,
            x=x[0, :],
            y=y[:, 0],
            colorscale="RdBu",
            zmid=0.0,
            zmin=-gx_absmax,
            zmax=gx_absmax,
            colorbar=_heatmap_colorbar(fig, "xaxis9", "yaxis9"),
        ),
        row=4,
        col=3,
    )
    fig.add_trace(
        go.Heatmap(
            z=gy_dem,
            x=x[0, :],
            y=y[:, 0],
            colorscale="RdBu",
            zmid=0.0,
            colorbar=_heatmap_colorbar(fig, "xaxis10", "yaxis10"),
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=gy_pred,
            x=x[0, :],
            y=y[:, 0],
            colorscale="RdBu",
            zmid=0.0,
            colorbar=_heatmap_colorbar(fig, "xaxis11", "yaxis11"),
        ),
        row=5,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=gy_error,
            x=x[0, :],
            y=y[:, 0],
            colorscale="RdBu",
            zmid=0.0,
            zmin=-gy_absmax,
            zmax=gy_absmax,
            colorbar=_heatmap_colorbar(fig, "xaxis12", "yaxis12"),
        ),
        row=5,
        col=3,
    )

    row1_y_domain = fig.layout.scene.domain.y
    fig.update_layout(
        height=2100,
        width=1750,
        margin={"r": 180},
        title="DEM vs NEMo Fit",
        plot_bgcolor="white",
        paper_bgcolor="white",
        scene={"domain": {"x": [0.0, 0.47], "y": row1_y_domain}},
        scene2={"domain": {"x": [0.53, 1.0], "y": row1_y_domain}},
    )
    fig.update_scenes(aspectmode="data")
    for xaxis_name, yaxis_name in (
        ("xaxis", "yaxis"),
        ("xaxis2", "yaxis2"),
        ("xaxis3", "yaxis3"),
        ("xaxis4", "yaxis4"),
        ("xaxis5", "yaxis5"),
        ("xaxis6", "yaxis6"),
        ("xaxis7", "yaxis7"),
        ("xaxis8", "yaxis8"),
        ("xaxis9", "yaxis9"),
        ("xaxis10", "yaxis10"),
        ("xaxis11", "yaxis11"),
        ("xaxis12", "yaxis12"),
    ):
        if hasattr(fig.layout, xaxis_name) and hasattr(fig.layout, yaxis_name):
            getattr(fig.layout, xaxis_name).update(showgrid=False, zeroline=False, showline=False)
            getattr(fig.layout, yaxis_name).update(
                scaleanchor=xaxis_name.replace("axis", ""),
                scaleratio=1,
                showgrid=False,
                zeroline=False,
                showline=False,
            )
    return fig
