"""Summarize AirSim tracker logs into defense-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_manifest(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_name = {}
    for item in data.get("paths", []):
        by_name[str(item["name"])] = item
        tracker_path = item.get("tracker_path_file")
        if tracker_path:
            by_name[Path(str(tracker_path)).name] = item
        path_file = item.get("path_file")
        if path_file:
            by_name[Path(str(path_file)).name] = item
    return by_name


def find_summary_files(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.glob("summary_*.json"))


def trajectory_for_summary(summary_path: Path) -> Path | None:
    timestamp = summary_path.stem.removeprefix("summary_")
    candidate = summary_path.with_name(f"trajectory_{timestamp}.csv")
    return candidate if candidate.exists() else None


def read_last_trajectory_row(path: Path | None) -> dict[str, str] | None:
    if path is None or not path.exists():
        return None
    last = None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    return last


def read_trajectory_stats(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    speeds = []
    energies = []
    distances = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, target in (("speed", speeds), ("energy_step", energies), ("distance_step", distances)):
                try:
                    target.append(float(row[key]))
                except (KeyError, TypeError, ValueError):
                    pass
    stats: dict[str, float] = {}
    if speeds:
        speed = np.asarray(speeds, dtype=np.float64)
        stats["mean_speed_mps"] = float(np.nanmean(speed))
        stats["max_speed_mps"] = float(np.nanmax(speed))
    if energies:
        stats["trajectory_energy_kj"] = float(np.nansum(np.asarray(energies, dtype=np.float64)) / 1000.0)
    if distances:
        stats["trajectory_distance_m"] = float(np.nansum(np.asarray(distances, dtype=np.float64)))
    return stats


def infer_path_key(summary: dict[str, object], summary_path: Path) -> str:
    for key in ("path_file", "PATH_FILE", "path", "path_name"):
        value = summary.get(key)
        if value:
            return Path(str(value)).name
    return summary_path.stem.removeprefix("summary_")


def goal_error_from_manifest(manifest_entry: dict[str, object] | None, final_row: dict[str, str] | None) -> float | None:
    if manifest_entry is None or final_row is None:
        return None
    try:
        goal = np.array(
            [
                float(manifest_entry["goal_x"]),
                float(manifest_entry["goal_y"]),
                float(manifest_entry["goal_z"]),
            ],
            dtype=np.float64,
        )
        final = np.array([float(final_row["x"]), float(final_row["y"]), float(final_row["z"])], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return float(np.linalg.norm(final - goal))


def summarize(logs_dir: Path, manifest_path: Path | None, success_threshold_m: float) -> list[dict[str, object]]:
    manifest = load_manifest(manifest_path)
    rows: list[dict[str, object]] = []
    for summary_path in find_summary_files(logs_dir):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trajectory_path = trajectory_for_summary(summary_path)
        final_row = read_last_trajectory_row(trajectory_path)
        traj_stats = read_trajectory_stats(trajectory_path)
        path_key = infer_path_key(summary, summary_path)
        manifest_entry = manifest.get(path_key)
        goal_error = goal_error_from_manifest(manifest_entry, final_row)
        row = {
            "run": summary_path.stem.removeprefix("summary_"),
            "path_key": path_key,
            "path_name": manifest_entry.get("name", "") if manifest_entry else "",
            "route": manifest_entry.get("route", "") if manifest_entry else "",
            "kind": manifest_entry.get("kind", "") if manifest_entry else "",
            "summary_file": str(summary_path),
            "trajectory_file": str(trajectory_path) if trajectory_path else "",
            "time_taken_s": float(summary.get("time_taken_s", np.nan)),
            "distance_m": float(summary.get("distance_m", np.nan)),
            "energy_kj": float(summary.get("energy_kj", np.nan)),
            "goal_error_m": goal_error if goal_error is not None else np.nan,
            "success": bool(goal_error is not None and goal_error <= success_threshold_m),
            **traj_stats,
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    cols = ["run", "path_name", "route", "kind", "success", "time_taken_s", "distance_m", "energy_kj", "goal_error_m"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append("" if np.isnan(value) else f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("data/logs"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/defense_demo/defense_paths_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/defense_demo"))
    parser.add_argument("--success-threshold-m", type=float, default=10.0)
    args = parser.parse_args()

    rows = summarize(args.logs_dir, args.manifest, float(args.success_threshold_m))
    csv_path = args.output_dir / "airsim_execution_summary.csv"
    md_path = args.output_dir / "airsim_execution_summary.md"
    json_path = args.output_dir / "airsim_execution_summary.json"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Found {len(rows)} AirSim summary logs in {args.logs_dir}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
