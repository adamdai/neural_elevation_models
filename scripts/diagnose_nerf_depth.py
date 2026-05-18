"""Compare NeRF depth rendering methods with unbiased 3D surfaces and geometry metrics."""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-14_121524/config.yml"))
    parser.add_argument("--data", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked"), help="Override data path in config")
    parser.add_argument("--dem-pc", type=Path, default=Path("/home/shared/data_nerfstudio/moon_spiral_2_masked/dem_pc.npy"))
    parser.add_argument("--nerfstudio-root", type=Path, default=Path("/home/addai/NeRF/nerfstudio"))
    parser.add_argument("--res", type=int, default=128, help="Resolution for grid")
    parser.add_argument("--bound", type=float, default=900.0, help="Bounds (+/-) for x and y")
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_diagnostics/latest_geometry.html"))
    return parser.parse_args()


def load_dem_interpolator(path: Path) -> tuple[RegularGridInterpolator, dict]:
    dem = np.load(path)
    order = np.lexsort((dem[:, 0], dem[:, 1]))
    sorted_dem = dem[order]
    xs = np.unique(sorted_dem[:, 0])
    ys = np.unique(sorted_dem[:, 1])
    z_grid = sorted_dem[:, 2].reshape(len(ys), len(xs))
    bounds = {
        "x_min": xs.min(), "x_max": xs.max(),
        "y_min": ys.min(), "y_max": ys.max(),
        "z_min": sorted_dem[:, 2].min(), "z_max": sorted_dem[:, 2].max(),
    }
    return RegularGridInterpolator((ys, xs), z_grid, bounds_error=False, fill_value=np.nan), bounds


def compute_surface_normals(height: np.ndarray, resolution: float) -> np.ndarray:
    """Compute surface normals from height field."""
    h_y, h_x = np.gradient(height.astype(np.float64), resolution)
    # n = [-h_x, -h_y, 1]
    normals = np.stack([-h_x, -h_y, np.ones_like(h_x)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / norm


def compute_slope_angles(height: np.ndarray, resolution: float) -> np.ndarray:
    """Compute slope angles in degrees."""
    h_y, h_x = np.gradient(height.astype(np.float64), resolution)
    slope = np.arctan(np.sqrt(h_x**2 + h_y**2))
    return np.degrees(slope)


def evaluate_heightfield_geometry(pred_height: np.ndarray, gt_height: np.ndarray, resolution: float) -> dict:
    """Evaluate slope and normal angular errors."""
    # Mask borders (1-pixel border)
    pred = pred_height[1:-1, 1:-1]
    gt = gt_height[1:-1, 1:-1]
    
    if not (np.isfinite(pred).any() and np.isfinite(gt).any()):
        return {}

    # Slope evaluation
    slope_pred = compute_slope_angles(pred_height, resolution)[1:-1, 1:-1]
    slope_gt = compute_slope_angles(gt_height, resolution)[1:-1, 1:-1]
    slope_error = np.abs(slope_pred - slope_gt)
    
    # Normal evaluation
    n_pred = compute_surface_normals(pred_height, resolution)[1:-1, 1:-1]
    n_gt = compute_surface_normals(gt_height, resolution)[1:-1, 1:-1]
    dot = np.sum(n_pred * n_gt, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    normal_error = np.degrees(np.arccos(dot))

    valid = np.isfinite(slope_error) & np.isfinite(normal_error)
    if not valid.any():
        return {}
    
    se = slope_error[valid]
    ne = normal_error[valid]
    
    return {
        "slope_err_mean": float(np.mean(se)),
        "slope_err_med": float(np.median(se)),
        "slope_err_p95": float(np.percentile(se, 95)),
        "normal_err_mean": float(np.mean(ne)),
        "normal_err_med": float(np.median(ne)),
        "normal_err_p95": float(np.percentile(ne, 95)),
        "slope_map": slope_pred,
        "normal_err_map": normal_error,
        "slope_err_map": slope_error,
    }


def main() -> None:
    args = parse_args()
    
    # Make output path absolute before changing directory
    output_path = args.output.resolve()
    
    terrain_nerf_root = Path("/home/addai/NeRF/terrain-nerf").resolve()
    nerfstudio_root = args.nerfstudio_root.expanduser().resolve()
    
    sys.path.insert(0, str(terrain_nerf_root))
    sys.path.insert(0, str(nerfstudio_root))
    
    # Change to the directory where the outputs/ folder lives
    os.chdir(terrain_nerf_root)

    import terrain_nerf  # noqa: F401
    from nerfstudio.cameras.rays import RayBundle
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.utils import install_checks
    install_checks.check_ffmpeg_installed() # Needed for some ns imports

    config, pipeline, _, _ = eval_setup(args.config.expanduser(), test_mode="val")
    
    # Ensure checkpoints can be found
    config.load_dir = args.config.expanduser().parent
    
    # Override data path if it doesn\''t exist
    if not config.pipeline.datamanager.data.exists():
        config.pipeline.datamanager.data = args.data.expanduser()
        config.pipeline.datamanager.dataparser.data = args.data.expanduser()
        
    model = pipeline.model.eval()
    datamanager = pipeline.datamanager
    dem_interp, dem_bounds = load_dem_interpolator(args.dem_pc.expanduser())

    thresholds = [0.3, 0.2, 0.15, 0.1, 0.05, 0.01]
    method_names = ["expected", "median", "nemo"] + [f"threshold_{t}" for t in thresholds]
    
    grid_res = args.res
    resolution = (2.0 * args.bound) / (grid_res - 1)
    xs = np.linspace(-args.bound, args.bound, grid_res)
    ys = np.linspace(-args.bound, args.bound, grid_res)
    xv, yv = np.meshgrid(xs, ys)
    
    z_start = dem_bounds["z_max"] + 20.0
    points_raw = np.stack([xv, yv, np.full_like(xv, z_start)], axis=-1).reshape(-1, 3)
    points_nerf = datamanager.raw_to_nerf_points(torch.from_numpy(points_raw).float().to(model.device))
    
    p_down_raw = points_raw.copy()
    p_down_raw[:, 2] -= 1.0
    p_down_nerf = datamanager.raw_to_nerf_points(torch.from_numpy(p_down_raw).float().to(model.device))
    dirs_nerf = p_down_nerf - points_nerf
    dirs_nerf = dirs_nerf / torch.norm(dirs_nerf, dim=-1, keepdim=True)

    rays = RayBundle(
        origins=points_nerf,
        directions=dirs_nerf,
        pixel_area=torch.ones((points_nerf.shape[0], 1), device=model.device),
        nears=torch.zeros((points_nerf.shape[0], 1), device=model.device),
        fars=torch.ones((points_nerf.shape[0], 1), device=model.device) * 10.0,
        camera_indices=torch.zeros((points_nerf.shape[0], 1), device=model.device, dtype=torch.long),
    )

    results_z = {name: np.full((grid_res, grid_res), np.nan) for name in method_names}
    results_err = {name: np.full((grid_res, grid_res), np.nan) for name in method_names}
    
    gt_z_vals = dem_interp(np.stack([yv.flatten(), xv.flatten()], axis=-1))
    gt_z_grid = gt_z_vals.reshape(grid_res, grid_res)

    with torch.no_grad():
        chunk_size = 2048
        for i in range(0, len(rays), chunk_size):
            r_chunk = rays[i : i + chunk_size]
            ray_samples, _, _ = model.proposal_sampler(r_chunk, density_fns=model.density_fns)
            field_outputs = model.field.forward(ray_samples, compute_normals=False)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            
            midpoints = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2.0
            cumulative_weights = torch.cumsum(weights, dim=1)
            accumulation = model.renderer_accumulation(weights=weights).reshape(-1).cpu().numpy()

            chunk_results = {}
            chunk_results["expected"] = model.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
            chunk_results["median"] = model._weighted_depth_quantile(weights=weights, ray_samples=ray_samples, quantile=0.5)
            
            # NEMo evaluation (if available)
            if hasattr(model, "height_field") and model.height_field is not None:
                # Get raw XY for this chunk
                # We can use r_chunk.origins since these are top-down vertical rays
                # but let's be more robust and transform back just in case
                p_raw_origin = datamanager.nerf_to_raw_points(r_chunk.origins).cpu().numpy()
                xy = torch.from_numpy(p_raw_origin[:, :2]).float().to(model.device)
                h_nemo = model.height_field(xy) # This returns normalized/residual height depending on model
                # Wait, TNerfModel has a helper for this
                if hasattr(model, "_height_at_raw_xy"):
                    h_nemo = model._height_at_raw_xy(xy)
                
                # Reshape to match results_z logic below
                # We'll handle 'nemo' specially since it doesn't need depth
                nemo_z = h_nemo.cpu().numpy().flatten()
                for j in range(len(nemo_z)):
                    idx = i + j
                    row_idx = idx // grid_res
                    col_idx = idx % grid_res
                    results_z["nemo"][row_idx, col_idx] = nemo_z[j]
                    if np.isfinite(gt_z_grid[row_idx, col_idx]):
                        results_err["nemo"][row_idx, col_idx] = nemo_z[j] - gt_z_grid[row_idx, col_idx]

            for t in thresholds:
                mask = cumulative_weights > t
                t_idx = torch.argmax(mask.to(torch.int), dim=1)
                chunk_results[f"threshold_{t}"] = torch.gather(midpoints, 1, t_idx.unsqueeze(1))

            for name, depth in chunk_results.items():
                p_nerf = r_chunk.origins + r_chunk.directions * depth.reshape(-1, 1)
                p_raw = datamanager.nerf_to_raw_points(p_nerf).cpu().numpy()
                for j in range(len(p_raw)):
                    idx = i + j
                    r = idx // grid_res
                    c = idx % grid_res
                    if accumulation[j] > 0.1:
                        results_z[name][r, c] = p_raw[j, 2]
                        if np.isfinite(gt_z_grid[r, c]):
                            results_err[name][r, c] = p_raw[j, 2] - gt_z_grid[r, c]

    # Geometry Evaluation
    geometry_stats = {}
    for name in method_names:
        geometry_stats[name] = evaluate_heightfield_geometry(results_z[name], gt_z_grid, resolution)

    # 3. Visualization
    num_m = len(method_names)
    # Rows: 1(Stats) + 1(GT) + N(Surfaces/Heatmaps) + N(Slope/NormalErr)
    rows = 2 + num_m * 2
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=(
            "Error Distributions", "Summary Statistics",
            "GT Surface (DEM)", "GT Slope Map"
        ) + tuple(sum([[f"3D Surface: {name}", f"Unbiased Heatmap: {name}", f"Slope Map: {name}", f"Normal Angular Error: {name}"] for name in method_names], [])),
        specs=[
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "surface"}, {"type": "xy"}],
        ] + [[{"type": "surface"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]] * num_m,
        vertical_spacing=0.02,
    )

    all_metrics = {}

    # A. Stats Table & Distributions
    table_data = []
    biases = {}
    for name in method_names:
        errs = results_err[name].flatten()
        errs = errs[np.isfinite(errs)]
        if errs.size > 0:
            bias = float(np.mean(errs))
            biases[name] = bias
            unbiased_errs = errs - bias
            
            rmse = float(np.sqrt(np.mean(errs**2)))
            mae = float(np.mean(np.abs(errs)))
            max_e = float(np.max(np.abs(errs)))
            
            rmse_u = float(np.sqrt(np.mean(unbiased_errs**2)))
            mae_u = float(np.mean(np.abs(unbiased_errs)))
            max_u = float(np.max(np.abs(unbiased_errs)))
            
            g = geometry_stats[name]
            
            table_data.append([
                name, f"{bias:.2f}", f"{rmse:.2f}", f"{mae:.2f}",
                f"{rmse_u:.2f}", f"{mae_u:.2f}", f"{max_u:.2f}",
                f"{g.get('slope_err_mean', 0):.1f}°", f"{g.get('normal_err_mean', 0):.1f}°"
            ])
            fig.add_trace(go.Violin(y=unbiased_errs, name=name, box_visible=True, meanline_visible=True), row=1, col=1)
            
            all_metrics[name] = {
                "bias": bias, "rmse": rmse, "mae": mae, "max_err": max_e,
                "rmse_unbiased": rmse_u, "mae_unbiased": mae_u, "max_err_unbiased": max_u,
                "slope_err_mean": g.get('slope_err_mean'), "slope_err_p95": g.get('slope_err_p95'),
                "normal_err_mean": g.get('normal_err_mean'), "normal_err_p95": g.get('normal_err_p95'),
            }

    fig.add_trace(go.Table(
        header=dict(values=["Method", "Bias", "RMSE", "MAE", "RMSE_u", "MAE_u", "Max_u", "Slope_e", "Norm_e"]),
        cells=dict(values=list(zip(*table_data)))
    ), row=1, col=2)

    # B. GT Surface & Slope
    fig.add_trace(go.Surface(x=xs, y=ys, z=gt_z_grid, colorscale="Greens", name="GT DEM", showscale=False), row=2, col=1)
    fig.update_scenes(dict(aspectmode='data'), row=2, col=1)
    
    gt_slope = compute_slope_angles(gt_z_grid, resolution)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=gt_slope, colorscale="Magma", name="GT Slope", showscale=False), row=2, col=2)

    # C. Method Surfaces, Unbiased Heatmaps, Slope, and Normal Error
    for i, name in enumerate(method_names):
        surf_row = 3 + i*2
        geo_row = 4 + i*2
        
        # 3D Surface
        fig.add_trace(go.Surface(x=xs, y=ys, z=results_z[name], colorscale="Viridis", name=name, showscale=False), row=surf_row, col=1)
        fig.update_scenes(dict(aspectmode='data'), row=surf_row, col=1)
        
        # Unbiased Heatmap
        unbiased_heatmap = results_err[name] - biases.get(name, 0.0)
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=unbiased_heatmap, 
            colorscale="RdBu", zmid=0, zmin=-10, zmax=10,
            colorbar=dict(title=f"Unbiased {name}", x=1.02)
        ), row=surf_row, col=2)
        
        g = geometry_stats[name]
        if "slope_map" in g:
            # Slope Map
            fig.add_trace(go.Heatmap(x=xs[1:-1], y=ys[1:-1], z=g["slope_map"], colorscale="Magma", showscale=False), row=geo_row, col=1)
            # Normal Error Heatmap
            fig.add_trace(go.Heatmap(x=xs[1:-1], y=ys[1:-1], z=g["normal_err_map"], colorscale="Reds", zmin=0, zmax=45, colorbar=dict(title=f"NormErr {name}", x=1.08)), row=geo_row, col=2)

    fig.update_layout(height=600 + 800 * num_m, width=1500, title_text=f"NeRF Geometry Diagnostics: {args.config.parent.name}")
    fig.write_html(output_path)
    
    # Save metrics to JSON
    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"Diagnostics saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    original_cwd = os.getcwd()
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(original_cwd)
