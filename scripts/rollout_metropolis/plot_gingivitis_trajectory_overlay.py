#!/usr/bin/env python3

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os

from scripts.gingivitis import plot_gingivitis_trajectory_overlay as base


def main():
    out_dir = os.path.join("data", "rollout_metropolis")
    base.ROLL_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob_metropolis.tsv")
    base.ENDPOINTS_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_endpoints_cache_metropolis.npz")
    base.REAL_NPZ = os.path.join("data", "gingivitis", "visionary_rollout_direction_cache_metropolis.npz")
    base.OUT_PNG = os.path.join(out_dir, "gingivitis_rollout_trajectory_overlay.png")
    base.CACHE_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_trajectory_overlay_cache_metropolis.npz")
    base.main()


if __name__ == "__main__":
    main()
