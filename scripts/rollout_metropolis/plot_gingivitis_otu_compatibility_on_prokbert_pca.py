#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402
from scripts.rollout_metropolis.core import pick_anchor_set, score_logits_for_sets, sigmoid  # noqa: E402


GINGIVA_CSV = os.path.join("data", "gingivitis", "gingiva.csv")
PCA_CACHE = os.path.join("data", "rollout_metropolis", "prokbert_otu_pca2_cache.npz")


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay a gingivitis OTU compatibility TSV on the global ProkBERT OTU PCA map."
    )
    p.add_argument("--tsv", required=True, help="TSV produced by gingivitis_otu_compatibility_map.py")
    p.add_argument("--pca-cache", default=PCA_CACHE)
    p.add_argument("--gingiva-csv", default=GINGIVA_CSV)
    p.add_argument("--subject", required=True)
    p.add_argument("--t-start", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--p-anchor", type=float, default=0.95)
    p.add_argument("--core", choices=["start", "anchors"], default="anchors")
    p.add_argument("--out", default="")
    p.add_argument("--bg", choices=["hexbin", "points"], default="hexbin", help="How to render global background OTUs.")
    p.add_argument("--bg-alpha", type=float, default=0.25)
    p.add_argument("--bg-gridsize", type=int, default=160, help="hexbin only: grid resolution.")
    p.add_argument("--max-bg", type=int, default=0, help="Optional subsample for background points (0 = all).")
    return p.parse_args()


def time_key(t):
    s = str(t).strip()
    if s.upper() == "B":
        return -1e9
    try:
        return float(s)
    except Exception:
        return 1e9


def build_subject_time_to_otus(gingiva_csv, sra_to_micro, micro_to_otus):
    subject_time_to_otus = defaultdict(set)
    with open(gingiva_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("Run", "").strip()
            subj = row.get("subject_code", "").strip()
            tcode = row.get("time_code", "").strip()
            if not run or not subj or not tcode:
                continue
            srs = sra_to_micro.get(run)
            if not srs:
                continue
            otus = micro_to_otus.get(srs, [])
            if otus:
                subject_time_to_otus[(subj, tcode)].update(otus)
    return subject_time_to_otus


def main():
    args = parse_args()
    if not os.path.exists(args.pca_cache):
        raise SystemExit(f"Missing PCA cache: {args.pca_cache} (run build_prokbert_otu_pca_cache.py first)")

    cache = np.load(args.pca_cache, allow_pickle=True)
    keys = cache["keys"].astype(object)
    xy = cache["xy"].astype(np.float32)
    key_to_i = {str(keys[i]): i for i in range(len(keys))}

    _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=args.gingiva_csv)
    micro_to_otus = collect_micro_to_otus(sra_to_micro)
    subject_time_to_otus = build_subject_time_to_otus(args.gingiva_csv, sra_to_micro, micro_to_otus)

    start_otus = sorted(subject_time_to_otus.get((args.subject, args.t_start), set()))
    if not start_otus:
        raise SystemExit(f"No OTUs found for (subject,t_start)=({args.subject},{args.t_start}).")

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    import h5py  # local import

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        logits0 = score_logits_for_sets([sorted(start_otus)], model, device, emb_group, resolver)[0]
        anchors = set(pick_anchor_set(start_otus, logits0, p_threshold=args.p_anchor, temperature=args.temperature))
        if not anchors:
            anchors = {start_otus[0]}

    core = set(start_otus) if args.core == "start" else set(anchors)

    core_keys = []
    for o in core:
        k = resolver.get(o, o) if resolver else o
        if str(k) in key_to_i:
            core_keys.append(str(k))

    cand_keys = []
    cand_probs = []
    with open(args.tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            c = str(row.get("candidate", "")).strip()
            if not c:
                continue
            if c not in key_to_i:
                continue
            p = row.get("prob", "").strip()
            try:
                pv = float(p)
            except Exception:
                continue
            cand_keys.append(c)
            cand_probs.append(pv)

    if not cand_keys:
        raise SystemExit("No candidates from TSV were found in the PCA cache keys.")

    cand_i = np.asarray([key_to_i[k] for k in cand_keys], dtype=int)
    cand_xy = xy[cand_i]
    cand_probs = np.asarray(cand_probs, dtype=float)

    core_i = np.asarray([key_to_i[k] for k in core_keys], dtype=int) if core_keys else np.asarray([], dtype=int)
    core_xy = xy[core_i] if core_i.size else np.zeros((0, 2), dtype=np.float32)

    bg_xy = xy
    if args.max_bg and args.max_bg > 0 and bg_xy.shape[0] > args.max_bg:
        rng = np.random.default_rng(0)
        idx = rng.choice(bg_xy.shape[0], size=args.max_bg, replace=False)
        bg_xy = bg_xy[idx]

    out = args.out
    if not out:
        out = os.path.splitext(args.tsv)[0] + "_on_global_pca.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 9.0))

    if args.bg == "hexbin":
        ax.hexbin(
            bg_xy[:, 0],
            bg_xy[:, 1],
            gridsize=int(args.bg_gridsize),
            cmap="Greys",
            bins="log",
            mincnt=1,
            linewidths=0,
            alpha=float(args.bg_alpha),
            zorder=0,
        )
    else:
        ax.scatter(bg_xy[:, 0], bg_xy[:, 1], s=2, color="black", alpha=float(args.bg_alpha), linewidths=0, zorder=0)
    sc = ax.scatter(
        cand_xy[:, 0],
        cand_xy[:, 1],
        c=cand_probs,
        cmap="viridis",
        s=10,
        alpha=0.85,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
        zorder=2,
    )
    if core_xy.shape[0] > 0:
        ax.scatter(
            core_xy[:, 0],
            core_xy[:, 1],
            s=40,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            alpha=0.95,
            zorder=3,
            label=f"Core ({args.core})",
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Compatibility prob p(candidate | core)")

    ax.set_title(f"Compatibility overlay on global ProkBERT OTU PCA\n{args.subject} start={args.t_start}, core={args.core}")
    ax.set_xlabel("OTU PC1 (global)")
    ax.set_ylabel("OTU PC2 (global)")
    ax.grid(True, alpha=0.15)
    if core_xy.shape[0] > 0:
        ax.legend(frameon=True, facecolor="white", edgecolor="0.7", loc="best")

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
