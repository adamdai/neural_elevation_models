# nemo_utils/paths.py
from __future__ import annotations
from pathlib import Path
import os
import subprocess
from functools import lru_cache

ANCHORS = ("pyproject.toml", "setup.cfg", ".git", "conf")


@lru_cache()
def project_root(start: Path | None = None) -> Path:
    # 1) Env override wins
    env = os.environ.get("NEMO_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # 2) Try git (fast if repo)
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        )
        return Path(out.decode().strip()).resolve()
    except Exception:
        pass

    # 3) Walk up from start (or this file) for anchors
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if any((p / a).exists() for a in ANCHORS):
            return p

    # 4) Fallback: current working directory
    return Path.cwd().resolve()


def config_dir() -> Path:
    return project_root() / "configs"


def data_dir() -> Path:
    return project_root() / "data"
