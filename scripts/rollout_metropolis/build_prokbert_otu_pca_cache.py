#!/usr/bin/env python3

import argparse
import os
import sys

import h5py
import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


OUT_NPZ = os.path.join("data", "rollout_metropolis", "prokbert_otu_pca2_cache.npz")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build a cached 2D PCA projection of the full ProkBERT OTU embedding space "
            "(keys in PROKBERT_PATH['embeddings'])."
        )
    )
    p.add_argument("--out", default=OUT_NPZ)
    p.add_argument("--max-otus", type=int, default=0, help="Optional cap for debugging (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        keys = list(emb_group.keys())
        if not keys:
            raise SystemExit("No embedding keys found in PROKBERT_PATH['embeddings'].")

        if args.max_otus and args.max_otus > 0 and len(keys) > args.max_otus:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(keys), size=args.max_otus, replace=False)
            keys = [keys[i] for i in idx.tolist()]

        # Determine embedding dimension
        d = int(np.asarray(emb_group[keys[0]][()]).shape[0])
        X = np.empty((len(keys), d), dtype=np.float32)
        for i, k in enumerate(keys):
            X[i] = np.asarray(emb_group[k][()], dtype=np.float32)

    # PCA via covariance eigendecomposition (fast for d<<n).
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / float(max(1, Xc.shape[0] - 1))
    w, v = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    comps = v[:, order[:2]].T  # (2,d)
    xy = (Xc @ comps.T).astype(np.float32)  # (n,2)

    np.savez(
        args.out,
        keys=np.asarray(keys, dtype=object),
        mu=mu.astype(np.float32),
        comps=comps.astype(np.float32),
        xy=xy,
    )
    print(f"Saved: {args.out}")
    print("otus:", len(keys), "dim:", d)


if __name__ == "__main__":
    main()

