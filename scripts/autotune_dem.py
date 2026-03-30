from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

from nemo import DEM


@dataclass
class AutotuneDemArgs:
    dem_path: str
    study_name: str | None = None
    output_root: str = "output/autotune"
    patch_name: str | None = None
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    crop_center_x: float | None = None
    crop_center_y: float | None = None
    crop_width: float | None = None
    crop_height: float | None = None
    trials: int = 12
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_fit_points: int | None = 200000
    plot_stride: int = 8
    plot_batch_size: int = 65536
    iterations: int = 1000
    eval_every: int = 25
    early_stop: bool = False
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5
    restore_best: bool = True
    include_residual_mlp: bool = True
    include_hashgrid: bool = True
    include_tiled_residual_mlp: bool = True
    include_smooth_grid: bool = True
    batch_size_choices: tuple[int, ...] = (16384, 32768, 65536)
    lr_min: float = 1e-4
    lr_max: float = 3e-3
    weight_decay_choices: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4)
    score_height_rmse_weight: float = 1.0
    score_grad_rmse_weight: float = 1.0
    score_height_max_weight: float = 0.1
    score_grad_max_weight: float = 0.1
    score_runtime_weight: float = 0.0


@dataclass
class TrialConfig:
    model_name: str
    main_flags: dict[str, Any]
    model_flags: dict[str, Any]


def _study_name(args: AutotuneDemArgs) -> str:
    if args.study_name is not None:
        return args.study_name
    return f"{Path(args.dem_path).stem}_autotune"


def _model_slug(model_name: str) -> str:
    return model_name.replace("_", "-")


def _load_patch_registry() -> dict[str, dict[str, dict[str, object]]]:
    registry_path = Path(__file__).resolve().parent.parent / "data" / "dem_patches.json"
    if not registry_path.exists():
        return {}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _default_patch_name(dem_path: str | Path) -> str | None:
    if Path(dem_path).name == "Mt_Etna-DSM.tif":
        return "s3li_crater_dem_buffer_5"
    return None


def _resolve_crop_bounds(
    args: AutotuneDemArgs,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, str | None]:
    explicit_xlims = args.xlims
    explicit_ylims = args.ylims
    if (
        args.crop_center_x is not None
        or args.crop_center_y is not None
        or args.crop_width is not None
        or args.crop_height is not None
    ):
        if None in (args.crop_center_x, args.crop_center_y, args.crop_width, args.crop_height):
            raise ValueError(
                "Provide all of `crop_center_x`, `crop_center_y`, `crop_width`, and `crop_height`."
            )
        half_width = float(args.crop_width) / 2.0
        half_height = float(args.crop_height) / 2.0
        explicit_xlims = (float(args.crop_center_x) - half_width, float(args.crop_center_x) + half_width)
        explicit_ylims = (float(args.crop_center_y) - half_height, float(args.crop_center_y) + half_height)

    if explicit_xlims is not None or explicit_ylims is not None:
        if explicit_xlims is None or explicit_ylims is None:
            raise ValueError("Provide both `xlims` and `ylims` when cropping explicitly.")
        return explicit_xlims, explicit_ylims, None

    registry = _load_patch_registry()
    dem_name = Path(args.dem_path).name
    patch_name = args.patch_name or _default_patch_name(args.dem_path)
    if patch_name is None:
        return None, None, None

    dem_patches = registry.get(dem_name, {})
    patch = dem_patches.get(patch_name)
    if patch is None:
        available = ", ".join(sorted(dem_patches)) or "none"
        raise ValueError(f"Unknown patch '{patch_name}' for {dem_name}. Available patches: {available}.")

    xlims = tuple(float(v) for v in patch["xlims"])
    ylims = tuple(float(v) for v in patch["ylims"])
    return xlims, ylims, patch_name


def _load_dem_bounds(
    args: AutotuneDemArgs,
) -> tuple[tuple[float, float], tuple[float, float], str | None]:
    xlims, ylims, patch_name = _resolve_crop_bounds(args)
    dem = DEM.from_path(args.dem_path, xlims=xlims, ylims=ylims)
    return dem.bounds.x, dem.bounds.y, patch_name


def _sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    return 10.0 ** rng.uniform(math.log10(low), math.log10(high))


def _sample_residual_mlp(rng: random.Random) -> TrialConfig:
    return TrialConfig(
        model_name="residual-mlp",
        main_flags={},
        model_flags={
            "hidden_dim": rng.choice((64, 128, 256)),
            "depth": rng.choice((3, 4, 5, 6)),
            "baseline": rng.choice(("constant", "plane")),
        },
    )


def _sample_hashgrid(rng: random.Random) -> TrialConfig:
    return TrialConfig(
        model_name="hashgrid",
        main_flags={},
        model_flags={
            "n_levels": rng.choice((12, 16, 20)),
            "n_features_per_level": rng.choice((2, 4)),
            "log2_hashmap_size": rng.choice((18, 19, 20)),
            "base_resolution": rng.choice((8, 16, 32)),
            "per_level_scale": rng.choice((1.3, 1.5, 1.8, 2.0)),
            "n_neurons": rng.choice((32, 64, 128)),
            "n_hidden_layers": rng.choice((1, 2, 3)),
        },
    )


def _sample_tiled_residual_mlp(
    rng: random.Random,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> TrialConfig:
    width = max(bounds[0][1] - bounds[0][0], 1e-6)
    height = max(bounds[1][1] - bounds[1][0], 1e-6)
    tile_fraction = rng.choice((0.25, 0.33, 0.5, 0.75))
    tile_size_x = width * tile_fraction
    tile_size_y = height * tile_fraction
    overlap_fraction = rng.choice((0.1, 0.2, 0.25))
    return TrialConfig(
        model_name="tiled-residual-mlp",
        main_flags={},
        model_flags={
            "hidden_dim": rng.choice((32, 64, 128)),
            "depth": rng.choice((2, 3, 4)),
            "baseline": rng.choice(("constant", "plane")),
            "tile_size_x": tile_size_x,
            "tile_size_y": tile_size_y,
            "overlap_x": tile_size_x * overlap_fraction,
            "overlap_y": tile_size_y * overlap_fraction,
        },
    )


def _sample_smooth_grid(rng: random.Random) -> TrialConfig:
    return TrialConfig(
        model_name="smooth-grid",
        main_flags={},
        model_flags={
            "hidden_dim": rng.choice((32, 48, 64)),
            "depth": rng.choice((3, 4, 5)),
            "grid_resolution_x": rng.choice((64, 96, 128, 192)),
            "grid_resolution_y": rng.choice((64, 96, 128, 192)),
            "interpolation": rng.choice(("bilinear", "bicubic")),
        },
    )


def _sample_trial(
    args: AutotuneDemArgs,
    rng: random.Random,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> TrialConfig:
    candidates: list[str] = []
    if args.include_residual_mlp:
        candidates.append("residual-mlp")
    if args.include_hashgrid and args.device.startswith("cuda"):
        candidates.append("hashgrid")
    if args.include_tiled_residual_mlp:
        candidates.append("tiled-residual-mlp")
    if args.include_smooth_grid:
        candidates.append("smooth-grid")
    if not candidates:
        raise ValueError("Enable at least one model family for autotuning.")

    model_name = rng.choice(candidates)
    if model_name == "residual-mlp":
        trial = _sample_residual_mlp(rng)
    elif model_name == "hashgrid":
        trial = _sample_hashgrid(rng)
    elif model_name == "smooth-grid":
        trial = _sample_smooth_grid(rng)
    else:
        trial = _sample_tiled_residual_mlp(rng, bounds)

    trial.main_flags["lr"] = float(_sample_log_uniform(rng, args.lr_min, args.lr_max))
    trial.main_flags["batch_size"] = int(
        rng.choice(args.batch_size_choices or (65536,))
    )
    trial.main_flags["weight_decay"] = float(
        rng.choice(args.weight_decay_choices or (0.0,))
    )
    return trial


def _append_flag(cmd: list[str], flag: str, value: Any) -> None:
    if isinstance(value, bool):
        cmd.append(flag if value else flag.replace("--", "--no-"))
        return
    if isinstance(value, (tuple, list)):
        cmd.append(flag)
        cmd.extend(str(v) for v in value)
        return
    cmd.extend([flag, str(value)])


def _trial_command(
    args: AutotuneDemArgs,
    trial: TrialConfig,
    trial_output_root: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "fit_dem.py"),
        "--dem-path",
        args.dem_path,
        "--output-root",
        str(trial_output_root),
        "--device",
        args.device,
        "--plot-stride",
        str(args.plot_stride),
        "--plot-batch-size",
        str(args.plot_batch_size),
        "--iterations",
        str(args.iterations),
        "--eval-every",
        str(args.eval_every),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--seed",
        str(args.seed),
        "--no-auto-open-plot",
    ]
    if args.max_fit_points is not None:
        cmd.extend(["--max-fit-points", str(args.max_fit_points)])
    if args.patch_name is not None:
        cmd.extend(["--patch-name", args.patch_name])
    if args.xlims is not None:
        cmd.extend(["--xlims", str(args.xlims[0]), str(args.xlims[1])])
    if args.ylims is not None:
        cmd.extend(["--ylims", str(args.ylims[0]), str(args.ylims[1])])
    if args.crop_center_x is not None:
        cmd.extend(
            [
                "--crop-center-x",
                str(args.crop_center_x),
                "--crop-center-y",
                str(args.crop_center_y),
                "--crop-width",
                str(args.crop_width),
                "--crop-height",
                str(args.crop_height),
            ]
        )
    if args.early_stop:
        cmd.append("--early-stop")
    if not args.restore_best:
        cmd.append("--no-restore-best")

    for key, value in trial.main_flags.items():
        _append_flag(cmd, f"--{key.replace('_', '-')}", value)

    cmd.append(f"model:{trial.model_name}")
    for key, value in trial.model_flags.items():
        _append_flag(cmd, f"--model.{key.replace('_', '-')}", value)
    return cmd


def _score_trial(args: AutotuneDemArgs, metrics: dict[str, float], runtime_sec: float) -> float:
    return (
        args.score_height_rmse_weight * float(metrics["height_rmse"])
        + args.score_grad_rmse_weight * float(metrics["grad_rmse"])
        + args.score_height_max_weight * float(metrics["height_max_abs"])
        + args.score_grad_max_weight * float(metrics["grad_max"])
        + args.score_runtime_weight * float(runtime_sec)
    )


def _write_study_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(args: AutotuneDemArgs) -> None:
    rng = random.Random(args.seed)
    bounds = _load_dem_bounds(args)[:2]
    study_root = Path(args.output_root) / _study_name(args)
    trials_root = study_root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    summary_path = study_root / "study_results.json"
    leaderboard_path = study_root / "leaderboard.md"
    trial_records: list[dict[str, Any]] = []

    for trial_index in range(args.trials):
        trial = _sample_trial(args, rng, bounds)
        trial_root = trials_root / f"trial_{trial_index:03d}"
        trial_root.mkdir(parents=True, exist_ok=True)
        cmd = _trial_command(args, trial, trial_root)

        print(
            f"[autotune] trial={trial_index + 1}/{args.trials} "
            f"model={trial.model_name} lr={trial.main_flags['lr']:.3e}"
        )
        started = time.perf_counter()
        completed = subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parent.parent)
        runtime_sec = time.perf_counter() - started

        model_output_dir = trial_root / f"{Path(args.dem_path).stem}__{trial.model_name}"
        results_path = model_output_dir / "results.json"
        record: dict[str, Any] = {
            "trial_index": trial_index,
            "trial_dir": str(trial_root.resolve()),
            "model": trial.model_name,
            "main_flags": trial.main_flags,
            "model_flags": trial.model_flags,
            "command": cmd,
            "runtime_sec": runtime_sec,
            "returncode": int(completed.returncode),
            "results_path": str(results_path.resolve()),
        }

        if completed.returncode == 0 and results_path.exists():
            result_payload = json.loads(results_path.read_text(encoding="utf-8"))
            metrics = result_payload["metrics"]
            score = _score_trial(args, metrics, runtime_sec)
            record["metrics"] = metrics
            record["score"] = score
            record["output_dir"] = result_payload["output_dir"]
            record["plot_html"] = result_payload["plot_html"]
            record["model_checkpoint"] = result_payload["model_checkpoint"]
            print(
                f"[autotune] completed trial={trial_index + 1} "
                f"score={score:.6f} height_rmse={metrics['height_rmse']:.6f} "
                f"grad_rmse={metrics['grad_rmse']:.6f}"
            )
        else:
            record["error"] = "fit_dem failed"
            print(f"[autotune] trial={trial_index + 1} failed with return code {completed.returncode}")

        trial_records.append(record)
        successful_trials = [trial_record for trial_record in trial_records if "score" in trial_record]
        successful_trials.sort(key=lambda trial_record: float(trial_record["score"]))

        summary = {
            "dem_path": str(Path(args.dem_path).resolve()),
            "study_root": str(study_root.resolve()),
            "trials": args.trials,
            "seed": args.seed,
            "score_weights": {
                "height_rmse": args.score_height_rmse_weight,
                "grad_rmse": args.score_grad_rmse_weight,
                "height_max_abs": args.score_height_max_weight,
                "grad_max": args.score_grad_max_weight,
                "runtime_sec": args.score_runtime_weight,
            },
            "best_trial": successful_trials[0] if successful_trials else None,
            "trial_records": trial_records,
        }
        _write_study_summary(summary_path, summary)

        leaderboard_lines = [
            "# Autotune Leaderboard",
            "",
            f"- DEM: `{Path(args.dem_path).resolve()}`",
            f"- Study root: `{study_root.resolve()}`",
            f"- Trials finished: `{len(trial_records)}/{args.trials}`",
            "",
            "| Rank | Trial | Model | Score | Height RMSE | Grad RMSE | Runtime (s) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for rank, successful_trial in enumerate(successful_trials[:10], start=1):
            metrics = successful_trial["metrics"]
            leaderboard_lines.append(
                f"| {rank} | {successful_trial['trial_index']} | {successful_trial['model']} | "
                f"{successful_trial['score']:.6f} | {metrics['height_rmse']:.6f} | "
                f"{metrics['grad_rmse']:.6f} | {successful_trial['runtime_sec']:.2f} |"
            )
        leaderboard_path.write_text("\n".join(leaderboard_lines) + "\n", encoding="utf-8")

    best_trial = min(
        (trial_record for trial_record in trial_records if "score" in trial_record),
        key=lambda trial_record: float(trial_record["score"]),
        default=None,
    )
    if best_trial is None:
        print("[autotune] no successful trials")
        raise SystemExit(1)

    print("[autotune] best trial")
    print(
        f"  trial={best_trial['trial_index']} model={best_trial['model']} "
        f"score={best_trial['score']:.6f}"
    )
    print(f"  checkpoint={best_trial['model_checkpoint']}")
    print(f"  results={best_trial['results_path']}")
    print(f"  leaderboard={leaderboard_path}")


if __name__ == "__main__":
    main(tyro.cli(AutotuneDemArgs))
