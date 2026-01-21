#!/usr/bin/env python3

import os

from scripts.gingivitis import plot_gingivitis_trajectory_overlay as base


def main():
    base.ROLL_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob_metropolis.tsv")
    base.ENDPOINTS_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_endpoints_cache_metropolis.npz")
    base.REAL_NPZ = os.path.join("data", "gingivitis", "visionary_rollout_direction_cache_metropolis.npz")
    base.OUT_PNG = os.path.join("data", "gingivitis", "gingivitis_rollout_trajectory_overlay_metropolis.png")
    base.CACHE_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_trajectory_overlay_cache_metropolis.npz")
    base.main()


if __name__ == "__main__":
    main()

