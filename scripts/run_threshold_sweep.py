"""Automate Terrain-Nerfacto experiments for threshold optimization."""

import subprocess
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
DATA_PATH = "/home/shared/data_nerfstudio/moon_spiral_2_masked"
TERRAIN_NERF_ROOT = "/home/addai/NeRF/terrain-nerf"
DIAGNOSTIC_SCRIPT = "/home/addai/Projects/neural_elevation_models/scripts/diagnose_nerf_depth.py"
ITERATIONS = 5000
BOUND = 900
RES = 128

# Threshold sweep: 0.3 down to 0.05 in 0.05 increments
THRESHOLDS = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

def run_command(cmd, cwd=None):
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode

def main():
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = f"threshold_sweep_{timestamp}"
    exp_root = Path("outputs/depth_diagnostics") / sweep_name
    exp_root.mkdir(parents=True, exist_ok=True)

    for t in THRESHOLDS:
        name = f"thresh_{t:.2f}"
        params = {
            "height_supervision_mode": "threshold",
            "threshold_supervision_value": t,
            "height_supervision_rays": "vertical",
            "threshold_supervision_huber_delta": 0.1,
            "height_tv_loss_mult": 0.0,
            "height_curvature_loss_mult": 0.0,
        }
        print(f"\n\n=== RUNNING EXPERIMENT: {name} ===")
        
        # 1. Training
        cmd = [
            "conda", "run", "--no-capture-output", "-n", "shadow-splat",
            "ns-train", "terrain-nerfacto",
            "--data", DATA_PATH,
            "--max_num_iterations", str(ITERATIONS),
            "--vis", "viewer",
            "--viewer.quit-on-train-completion", "True",
            "--timestamp", f"sweep_{name}"
        ]
        for p_name, p_val in params.items():
            cmd.extend([f"--pipeline.model.{p_name}", str(p_val)])
            
        ret = run_command(cmd, cwd=TERRAIN_NERF_ROOT)
        if ret != 0:
            print(f"Experiment {name} failed training.")
            continue
            
        # 2. Diagnostics
        config_path = Path(TERRAIN_NERF_ROOT) / "outputs" / "moon_spiral_2_masked" / "terrain-nerfacto" / f"sweep_{name}" / "config.yml"
        diag_output = exp_root / f"diag_{name}.html"
        diag_json = exp_root / f"diag_{name}.json"
        
        diag_cmd = [
            "conda", "run", "-n", "shadow-splat",
            "python", DIAGNOSTIC_SCRIPT,
            "--config", str(config_path),
            "--res", str(RES),
            "--bound", str(BOUND),
            "--output", str(diag_output.resolve())
        ]
        run_command(diag_cmd)
        
        # 3. Parse results
        # The diag script saves to its output path + .json
        actual_json = diag_output.with_suffix(".json")
        if actual_json.exists():
            with open(actual_json, "r") as f:
                data = json.load(f)
                t_name = f"threshold_{t}"
                if t_name in data:
                    metrics = data[t_name]
                    metrics["threshold"] = t
                    metrics["name"] = name
                    results.append(metrics)
                    print(f"Results for {name}: RMSE_u={metrics['rmse_unbiased']:.2f}, NormErr={metrics['normal_err_mean']:.2f}")

    # 4. Aggregation
    if results:
        df = pd.DataFrame(results)
        summary_path = exp_root / "summary.csv"
        df.to_csv(summary_path, index=False)
        
        # Sort by unbiased RMSE
        df = df.sort_values("rmse_unbiased")
        
        md_summary = df[[
            "threshold", "bias", "rmse_unbiased", "mae_unbiased", 
            "slope_err_mean", "normal_err_mean", "normal_err_p95"
        ]].to_markdown(index=False)
        
        with open(exp_root / "summary.md", "w") as f:
            f.write(f"# Threshold Optimization Sweep Results\n\n")
            f.write(md_summary)
            
        print("\n\n=== SWEEP COMPLETE ===")
        print(md_summary)
        
        best_rmse = df.iloc[0]
        print(f"\nBest threshold for RMSE_u: {best_rmse['threshold']} ({best_rmse['rmse_unbiased']:.2f}m)")
        
        # Also check normal error
        df_geo = df.sort_values("normal_err_mean")
        best_geo = df_geo.iloc[0]
        print(f"Best threshold for Normal Error: {best_geo['threshold']} ({best_geo['normal_err_mean']:.2f}°)")

if __name__ == "__main__":
    main()
