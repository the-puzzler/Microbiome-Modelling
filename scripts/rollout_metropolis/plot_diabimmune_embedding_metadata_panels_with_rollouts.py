#!/usr/bin/env python3

import os

from scripts.diabimmune import plot_diabimmune_embedding_metadata_panels_with_rollouts as base


def main():
    base.REAL_NPZ = os.path.join("data", "diabimmune", "diabimmune_real_embeddings_cache_metropolis.npz")
    base.ENDPOINTS_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoints_cache_metropolis.npz")
    base.ROLL_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob_metropolis.tsv")
    base.ENDPOINT_STABILITY_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoint_stability_cache_metropolis.npz")
    base.OUT_PNG = os.path.join("data", "diabimmune", "diabimmune_embedding_metadata_panels_with_rollouts_metropolis.png")
    base.main()


if __name__ == "__main__":
    main()

