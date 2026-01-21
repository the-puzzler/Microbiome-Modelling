#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Plot a single-rollout TSV (counts + PCA trajectory).")
    p.add_argument("--dataset", choices=["gingiva", "diabimmune"], default="gingiva")
    p.add_argument("--tsv", default=None, help="Path to single-rollout TSV (defaults depend on --dataset).")
    p.add_argument("--out", default=None, help="Output PNG path (defaults depend on --dataset).")
    return p.parse_args()


def pca2_fit_transform(X):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def parse_index_list(raw):
    if not raw:
        return []
    return [int(tok) for tok in str(raw).split(";") if str(tok).strip()]


def compute_embedding_from_indices(idx_list, all_otus, model, device, emb_group, resolver=None):
    import torch

    vecs = []
    for i in idx_list:
        if not (0 <= i < len(all_otus)):
            continue
        oid = all_otus[i]
        key = resolver.get(oid, oid) if resolver else oid
        if key in emb_group:
            vecs.append(torch.tensor(emb_group[key][()], dtype=torch.float32, device=device))
    if not vecs:
        return None
    x1 = torch.stack(vecs, dim=0).unsqueeze(0)
    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        mask = torch.ones((1, h1.shape[1]), dtype=torch.bool, device=device)
        h = model.transformer(h1, src_key_padding_mask=~mask)
        vec = h.mean(dim=1).squeeze(0).cpu().numpy()
    return vec


def load_dataset_mappings(dataset):
    if dataset == "gingiva":
        from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402

        gingiva_csv = os.path.join("data", "gingivitis", "gingiva.csv")
        _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=gingiva_csv)
        micro_to_otus = collect_micro_to_otus(sra_to_micro)
    else:
        from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402

        run_rows, sra_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
        micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)

    all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}
    return all_otus, resolver


def main():
    args = parse_args()
    if args.tsv is None:
        args.tsv = (
            os.path.join("data", "gingivitis", "single_rollout_debug.tsv")
            if args.dataset == "gingiva"
            else os.path.join("data", "diabimmune", "single_rollout_debug.tsv")
        )
    if args.out is None:
        args.out = (
            os.path.join("data", "gingivitis", "single_rollout_summary.png")
            if args.dataset == "gingiva"
            else os.path.join("data", "diabimmune", "single_rollout_summary.png")
        )
    if not os.path.exists(args.tsv):
        raise SystemExit(f"Missing TSV: {args.tsv}")

    steps = []
    idx_lists = []
    n_current = []
    n_same = []
    n_added = []
    n_removed = []
    subj = None
    t_start = None

    with open(args.tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if subj is None:
                subj = row.get("subject", "")
                t_start = row.get("t_start", "")
            steps.append(int(row["step"]))
            n_current.append(int(row["n_current"]))
            n_same.append(int(row["n_same"]))
            n_added.append(int(row["n_added"]))
            n_removed.append(int(row["n_removed"]))
            idx_lists.append(parse_index_list(row.get("current_otu_indices", "")))

    if not steps:
        raise SystemExit("No rows found in TSV.")

    order = np.argsort(np.asarray(steps, dtype=int))
    steps = np.asarray(steps, dtype=int)[order]
    n_current = np.asarray(n_current, dtype=int)[order]
    n_same = np.asarray(n_same, dtype=int)[order]
    n_added = np.asarray(n_added, dtype=int)[order]
    n_removed = np.asarray(n_removed, dtype=int)[order]
    idx_lists = [idx_lists[i] for i in order.tolist()]

    all_otus, resolver = load_dataset_mappings(args.dataset)
    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)

    embs = []
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for idx_list in idx_lists:
            v = compute_embedding_from_indices(idx_list, all_otus, model, device, emb_group, resolver)
            if v is None:
                raise SystemExit("Failed to compute embedding for a step (empty OTU list?)")
            embs.append(v.astype(float))

    xy = pca2_fit_transform(np.stack(embs, axis=0))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ax0.plot(steps, n_current, label="n_current", linewidth=2.0)
    ax0.plot(steps, n_same, label="n_same", linewidth=1.5)
    ax0.plot(steps, n_added, label="n_added", linewidth=1.5)
    ax0.plot(steps, n_removed, label="n_removed", linewidth=1.5)
    ax0.set_xlabel("step")
    ax0.set_ylabel("count")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False, ncol=2)

    cmap = cm.viridis
    denom = max(1, int(steps.max() - steps.min()))
    colors = cmap((steps - steps.min()) / denom)
    ax1.plot(xy[:, 0], xy[:, 1], color="black", alpha=0.25, linewidth=1.0, zorder=1)
    ax1.scatter(xy[:, 0], xy[:, 1], c=colors, s=45, linewidths=0, zorder=2)
    ax1.scatter([xy[0, 0]], [xy[0, 1]], s=120, marker="o", facecolors="none", edgecolors="black", linewidths=1.6, zorder=3)
    ax1.scatter([xy[-1, 0]], [xy[-1, 1]], s=160, marker="*", color="black", zorder=4)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_box_aspect(1)
    ax1.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(steps.astype(float))
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("step")

    title = "Single rollout summary"
    if subj:
        title += f" ({subj} {t_start})"
    fig.suptitle(title)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
