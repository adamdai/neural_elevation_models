"""Automate Terrain-Nerfacto experiments with a JSON status logger."""

import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
DATA_PATH = "/home/shared/data_nerfstudio/moon_spiral_2_masked"
CHECKPOINT_DIR = "/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-14_121524"
BASE_ITERATIONS = 15000
ADDITIONAL_ITERATIONS = 5000
TOTAL_ITERATIONS = BASE_ITERATIONS + ADDITIONAL_ITERATIONS
TERRAIN_NERF_ROOT = "/home/addai/NeRF/terrain-nerf"
OUTPUT_DIR = "/home/shared/outputs/addai"
STATUS_FILE = "/home/addai/Projects/neural_elevation_models/outputs/sweep_status.json"

EXPERIMENTS = [
    {
        "name": "baseline_v_t0.15_frozen",
        "params": {
            "height_supervision_mode": "threshold",
            "threshold_supervision_value": 0.15,
            "height_supervision_rays": "vertical",
            "threshold_supervision_huber_delta": 0.1,
            "train_height_field_only": True,
        }
    },
    {
        "name": "tv_0.01_frozen",
        "params": {
            "height_supervision_mode": "threshold",
            "threshold_supervision_value": 0.15,
            "height_supervision_rays": "vertical",
            "threshold_supervision_huber_delta": 0.1,
            "train_height_field_only": True,
            "height_tv_loss_mult": 0.01,
        }
    },
    {
        "name": "tv_0.1_frozen",
        "params": {
            "height_supervision_mode": "threshold",
            "threshold_supervision_value": 0.15,
            "height_supervision_rays": "vertical",
            "threshold_supervision_huber_delta": 0.1,
            "train_height_field_only": True,
            "height_tv_loss_mult": 0.1,
        }
    },
    {
        "name": "curv_0.01_frozen",
        "params": {
            "height_supervision_mode": "threshold",
            "threshold_supervision_value": 0.15,
            "height_supervision_rays": "vertical",
            "threshold_supervision_huber_delta": 0.1,
            "train_height_field_only": True,
            "height_curvature_loss_mult": 0.01,
        }
    },
]

def update_status(data):
    Path(STATUS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _ns_cli_name(name: str) -> str:
    return name.replace("_", "-")


def _parse_height_surface_metrics(path: Path) -> dict:
    metrics = {}
    if not path.exists():
        return metrics
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def _status_metrics_from_height_surface(metrics: dict) -> dict:
    required = ["RMSE [m]", "MAE [m]", "Max abs error [m]", "Bias [m]", "Valid fraction"]
    missing = [key for key in required if key not in metrics]
    if missing:
        return {
            "error": f"Missing NEMo metric keys: {missing}",
            "eval_source": "nemo_height_field_surface",
        }
    return {
        "rmse": round(metrics["RMSE [m]"], 2),
        "mae": round(metrics["MAE [m]"], 2),
        "max_abs_error": round(metrics["Max abs error [m]"], 2),
        "bias": round(metrics["Bias [m]"], 2),
        "valid_fraction": round(metrics["Valid fraction"], 4),
        "eval_source": "nemo_height_field_surface",
    }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(OUTPUT_DIR) / "moon_spiral_2_masked" / "terrain-nerfacto"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "sweep_name": f"Sweep {timestamp}",
        "experiments": [],
        "current_index": 0,
        "current_name": "Idle",
        "current_progress": 0.0,
        "is_complete": False
    }

    # Initialize experiment slots
    for exp in EXPERIMENTS:
        run_dir = sweep_dir / f"sweep_{exp['name']}"
        status["experiments"].append({
            "name": exp["name"],
            "status": "pending",
            "metrics": {},
            "run_dir": str(run_dir),
            "nemo_checkpoint": str(run_dir / "nemo_model.pt"),
            "surface_plot": str(run_dir / "height_field_surface.html"),
        })
    update_status(status)

    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        params = exp["params"]
        
        status["current_index"] = i
        status["current_name"] = name
        status["current_progress"] = 0.0
        status["experiments"][i]["status"] = "running"
        update_status(status)
        
        # 1. Training
        cmd = [
            "conda", "run", "--no-capture-output", "-n", "shadow-splat",
            "ns-train", "terrain-nerfacto",
            "--data", DATA_PATH,
            "--output-dir", OUTPUT_DIR,
            "--load-dir", str(CHECKPOINT_DIR) + "/nerfstudio_models",
            "--max-num-iterations", str(TOTAL_ITERATIONS),
            "--steps-per-save", str(ADDITIONAL_ITERATIONS),
            "--vis", "viewer",
            "--viewer.quit-on-train-completion", "True",
            "--timestamp", f"sweep_{name}"
        ]
        for p_name, p_val in params.items():
            cmd.extend([f"--pipeline.model.{_ns_cli_name(p_name)}", str(p_val)])
            
        print(f"Starting {name}...")
        process = subprocess.Popen(cmd, cwd=TERRAIN_NERF_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Parse output for progress
        # Match pattern like "15110 (75.55%)"
        prog_regex = re.compile(r"(\d+) \((\d+\.\d+)%\)")
        
        for line in process.stdout:
            match = prog_regex.search(line)
            if match:
                curr_step = int(match.group(1))
                # Normalize progress to the 5000 iterations we are actually running
                local_prog = max(0.0, (curr_step - BASE_ITERATIONS) / ADDITIONAL_ITERATIONS * 100.0)
                status["current_progress"] = local_prog
                update_status(status)
        
        process.wait()
        if process.returncode != 0:
            status["experiments"][i]["status"] = "failed"
            update_status(status)
            continue

        # 2. NEMo evaluation. Training writes this from model.height_field, not from NeRF rendered depth.
        status["experiments"][i]["status"] = "evaluating"
        update_status(status)

        run_dir = sweep_dir / f"sweep_{name}"
        metrics_path = run_dir / "height_field_surface.metrics.txt"
        metrics = _parse_height_surface_metrics(metrics_path)
        if metrics:
            status["experiments"][i]["metrics"] = _status_metrics_from_height_surface(metrics)
            status["experiments"][i]["status"] = "complete"
        else:
            status["experiments"][i]["status"] = "failed"
            status["experiments"][i]["metrics"] = {
                "error": f"Missing or empty NEMo metrics file: {metrics_path}"
            }
        update_status(status)

    status["is_complete"] = True
    status["current_name"] = "All Complete"
    status["current_progress"] = 100.0
    update_status(status)
    print("Sweep complete.")

if __name__ == "__main__":
    main()
