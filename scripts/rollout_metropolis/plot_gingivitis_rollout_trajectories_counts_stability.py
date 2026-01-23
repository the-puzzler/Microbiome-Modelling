#!/usr/bin/env python3

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gingivitis import plot_gingivitis_rollout_trajectories_counts_stability as base


def main():
    out_dir = os.path.join("data", "rollout_metropolis")
    base.ROLL_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob_metropolis.tsv")
    base.OUT_PNG = os.path.join(out_dir, "gingivitis_rollout_trajectories_counts_stability.png")
    base.CACHE_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_trajectories_counts_stability_cache_metropolis.npz")
    base.main()


if __name__ == "__main__":
    main()

