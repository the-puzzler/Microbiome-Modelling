#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

import h5py
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
OUT_DIR = os.path.join("data", "rollout_metropolis", "otu_compatibility")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Gingivitis OTU compatibility map: choose a core (start or anchors), sample candidate OTUs "
            "from a pool, score their probability in the core context, and plot a 2D OTU embedding map."
        )
    )
    p.add_argument("--gingiva-csv", default=GINGIVA_CSV)
    p.add_argument("--subject", default="", help="Subject id (default: first available).")
    p.add_argument("--t-start", default="", help="Start timepoint (default: first available for subject).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--p-anchor", type=float, default=0.95)
    p.add_argument(
        "--core",
        choices=["start", "anchors"],
        default="anchors",
        help="Core set used for compatibility scoring.",
    )
    p.add_argument(
        "--pool",
        choices=["gingivitis", "prokbert"],
        default="prokbert",
        help="Candidate sampling pool. 'prokbert' samples from all embedding keys (often outside gingivitis).",
    )
    p.add_argument("--n-candidates", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256, help="How many candidate sets to score per forward pass batch.")
    p.add_argument("--out-dir", default=OUT_DIR)
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


def pca2_fit(X):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, vt[:2]


def pca2_transform(X, mu, comps):
    X = np.asarray(X, dtype=float)
    return (X - mu) @ comps.T


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=args.gingiva_csv)
    micro_to_otus = collect_micro_to_otus(sra_to_micro)
    subject_time_to_otus = build_subject_time_to_otus(args.gingiva_csv, sra_to_micro, micro_to_otus)

    starts_all = [(subj, t, sorted(otus)) for (subj, t), otus in subject_time_to_otus.items() if otus]
    if not starts_all:
        raise SystemExit("No gingivitis start points found.")

    subjects = sorted({s for (s, _t, _o) in starts_all})
    subject = args.subject.strip() or subjects[0]
    times = sorted({t for (s, t, _o) in starts_all if s == subject}, key=time_key)
    if not times:
        raise SystemExit(f"No timepoints found for subject {subject!r}.")
    t_start = args.t_start.strip() or times[0]
    start_otus = next((otus for (s, t, otus) in starts_all if s == subject and t == t_start), None)
    if not start_otus:
        raise SystemExit(f"No OTUs found for (subject,t_start)=({subject},{t_start}).")

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")
    rng = np.random.default_rng(args.seed)

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]

        # Determine anchor/core set.
        logits0 = score_logits_for_sets([sorted(start_otus)], model, device, emb_group, resolver)[0]
        anchor_set = set(pick_anchor_set(sorted(start_otus), logits0, p_threshold=args.p_anchor, temperature=args.temperature))
        if not anchor_set:
            anchor_set = {start_otus[0]}
        if args.core == "anchors":
            core = sorted(anchor_set)
        else:
            core = sorted(set(start_otus))

        # Choose candidate pool.
        if args.pool == "gingivitis":
            pool = sorted({o for otus in micro_to_otus.values() for o in otus})
        else:
            pool = list(emb_group.keys())

        # Sample candidates, excluding core members.
        core_set = set(core)
        pool = [o for o in pool if o not in core_set]
        if not pool:
            raise SystemExit("Candidate pool is empty after excluding core.")

        n = min(int(args.n_candidates), len(pool))
        cand_idx = rng.choice(len(pool), size=n, replace=False)
        candidates = [pool[i] for i in cand_idx.tolist()]

        # Score candidates in batches: score logit for candidate when present with the core.
        probs = np.full((len(candidates),), np.nan, dtype=float)
        logits = np.full((len(candidates),), np.nan, dtype=float)
        present = np.zeros((len(candidates),), dtype=bool)

        bs = max(1, int(args.batch_size))
        for i0 in range(0, len(candidates), bs):
            i1 = min(len(candidates), i0 + bs)
            sets = [core + [c] for c in candidates[i0:i1]]
            scored = score_logits_for_sets(sets, model, device, emb_group, resolver)
            for j, c in enumerate(candidates[i0:i1]):
                d = scored[j]
                if c in d:
                    l = float(d[c])
                    logits[i0 + j] = l
                    probs[i0 + j] = float(sigmoid(l / float(args.temperature)))
                    present[i0 + j] = True

        # Write TSV
        base = f"{subject}_{t_start}_core-{args.core}_pool-{args.pool}_cand{len(candidates)}_seed{args.seed}"
        out_tsv = os.path.join(args.out_dir, base + ".tsv")
        out_png = os.path.join(args.out_dir, base + ".png")

        order = np.argsort(np.nan_to_num(probs, nan=-1.0))[::-1]
        with open(out_tsv, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(
                [
                    "candidate",
                    "prob",
                    "logit",
                    "present_in_embeddings",
                    "core_kind",
                    "pool",
                    "subject",
                    "t_start",
                    "n_core",
                    "n_start",
                    "n_anchors",
                ]
            )
            for k in order.tolist():
                w.writerow(
                    [
                        str(candidates[k]),
                        "" if not np.isfinite(probs[k]) else float(probs[k]),
                        "" if not np.isfinite(logits[k]) else float(logits[k]),
                        int(present[k]),
                        args.core,
                        args.pool,
                        subject,
                        t_start,
                        len(core),
                        len(start_otus),
                        len(anchor_set),
                    ]
                )

        # Build a 2D OTU embedding map for: core + candidates with finite scores.
        # For prokbert pool, candidates are already embedding keys; for gingivitis pool, resolver maps.
        def get_vec(otu_id):
            key = resolver.get(otu_id, otu_id) if resolver else otu_id
            if key not in emb_group:
                return None
            return np.asarray(emb_group[key][()], dtype=np.float32)

        plot_otus = []
        plot_vecs = []
        plot_probs = []
        plot_kind = []

        for o in core:
            v = get_vec(o)
            if v is None:
                continue
            plot_otus.append(o)
            plot_vecs.append(v)
            plot_probs.append(1.0)
            plot_kind.append("core")

        for i, c in enumerate(candidates):
            if not np.isfinite(probs[i]):
                continue
            v = get_vec(c)
            if v is None:
                continue
            plot_otus.append(c)
            plot_vecs.append(v)
            plot_probs.append(float(probs[i]))
            plot_kind.append("candidate")

        if len(plot_vecs) < 3:
            raise SystemExit("Not enough OTU embeddings to plot.")

        X = np.stack(plot_vecs, axis=0).astype(float)
        mu, comps = pca2_fit(X)
        xy = pca2_transform(X, mu, comps)

        plt.style.use("seaborn-v0_8-white")
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 8.5))

        plot_probs_arr = np.asarray(plot_probs, dtype=float)
        is_core = np.asarray([k == "core" for k in plot_kind], dtype=bool)

        # Candidates colored by compatibility prob
        cand_mask = ~is_core
        sc = ax.scatter(
            xy[cand_mask, 0],
            xy[cand_mask, 1],
            c=plot_probs_arr[cand_mask],
            cmap="viridis",
            s=10,
            alpha=0.8,
            linewidths=0,
            vmin=0.0,
            vmax=1.0,
            label="Candidates (colored by p)",
        )
        # Core highlighted
        ax.scatter(
            xy[is_core, 0],
            xy[is_core, 1],
            s=36,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            alpha=0.9,
            label=f"Core ({args.core})",
        )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Compatibility prob p(candidate | core)")

        ax.set_title(f"OTU compatibility map ({subject}, start={t_start})\ncore={args.core}, pool={args.pool}")
        ax.set_xlabel("OTU PC1 (ProkBERT)")
        ax.set_ylabel("OTU PC2 (ProkBERT)")
        ax.grid(True, alpha=0.15)
        ax.legend(frameon=True, facecolor="white", edgecolor="0.7", loc="best")

        plt.tight_layout()
        plt.savefig(out_png, dpi=300)

    print(f"Saved: {out_tsv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

