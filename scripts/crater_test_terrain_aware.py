"""Compatibility entry point for the renamed terrain-aware planner script."""

from scripts.terrain_aware_planner import *  # noqa: F401,F403
from scripts.terrain_aware_planner import main


if __name__ == "__main__":
    main()
