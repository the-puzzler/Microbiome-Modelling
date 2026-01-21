#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402


ROLL_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob.tsv")
OUT_PNG = os.path.join("data", "diabimmune", "diabimmune_rollout_trajectories_counts_stability.png")
CACHE_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_trajectories_counts_stability_cache.npz")


def parse_args():
    p = argparse.ArgumentParser(description="DIABIMMUNE: plot rollout trajectories (counts + stability vs step).")
    p.add_argument("--rollout-tsv", default=ROLL_TSV)
    p.add_argument("--n-trajectories", type=int, default=10)
    p.add_argument("--n-points", type=int, default=0, help="Subsample steps per trajectory to this many points (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=OUT_PNG)
    p.add_argument("--cache", default=CACHE_NPZ)
    return p.parse_args()


def parse_index_list(raw):
    if not raw:
        return []
    return [int(tok) for tok in str(raw).split(";") if str(tok).strip()]


def list_rollout_keys(rollout_tsv):
    keys = set()
    with open(rollout_tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            subj = row.get("subject", "").strip()
            t_start = row.get("t_start", "").strip()
            if subj and t_start:
                keys.add((subj, t_start))
    return sorted(keys, key=lambda x: (x[0], float(x[1])))


def load_rows_for_keys(rollout_tsv, wanted_keys):
    wanted = set(wanted_keys)
    rows_by_key = {k: [] for k in wanted_keys}
    with open(rollout_tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            key = (row.get("subject", "").strip(), row.get("t_start", "").strip())
            if key not in wanted:
                continue
            rows_by_key[key].append(row)
    for k in list(rows_by_key.keys()):
        rows_by_key[k] = sorted(rows_by_key[k], key=lambda rr: int(rr.get("step", "0")))
    return rows_by_key


def subsample_rows(rows, n_points):
    if not rows:
        return []
    if n_points <= 0 or n_points >= len(rows):
        return rows
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, n_points)).astype(int))
    return [rows[i] for i in idx.tolist()]


def compute_mean_logit(otu_ids, resolver, model, device, emb_group):
    logits = shared_utils.score_otu_list(otu_ids, resolver=resolver, model=model, device=device, emb_group=emb_group)
    if not logits:
        return float("nan")
    return float(np.mean(list(logits.values())))


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    rng = np.random.default_rng(args.seed)

    all_keys = list_rollout_keys(args.rollout_tsv)
    if not all_keys:
        raise SystemExit(f"No rollout keys found in {args.rollout_tsv}")

    n_traj = min(args.n_trajectories, len(all_keys))
    picked_idx = rng.choice(len(all_keys), size=n_traj, replace=False)
    picked_keys = [all_keys[i] for i in picked_idx.tolist()]

    rows_by_key = load_rows_for_keys(args.rollout_tsv, picked_keys)

    # Build OTU universe consistent with rollout TSV indices.
    _run_rows, sra_to_micro, _gid_to_sample, micro_to_subject, _micro_to_sample = load_run_data()
    micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)
    all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    out_keys = []
    out_steps = []
    out_n_current = []
    out_stability = []

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for key in picked_keys:
            rows = rows_by_key.get(key, [])
            rows_sub = subsample_rows(rows, args.n_points)
            if not rows_sub:
                continue

            steps = np.asarray([int(r["step"]) for r in rows_sub], dtype=int)
            n_current = np.asarray([int(r["n_current"]) for r in rows_sub], dtype=int)

            stability = np.full((len(rows_sub),), np.nan, dtype=float)
            for i, r in enumerate(rows_sub):
                idxs = parse_index_list(r.get("current_otu_indices", ""))
                otus = [all_otus[j] for j in idxs if 0 <= j < len(all_otus)]
                stability[i] = compute_mean_logit(otus, resolver, model, device, emb_group)

            out_keys.append(key)
            out_steps.append(steps)
            out_n_current.append(n_current)
            out_stability.append(stability)

    np.savez(
        args.cache,
        keys=np.asarray(out_keys, dtype=object),
        steps=np.asarray(out_steps, dtype=object),
        n_current=np.asarray(out_n_current, dtype=object),
        stability=np.asarray(out_stability, dtype=object),
    )

    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(len(out_keys), 2, figsize=(12, max(2.2 * len(out_keys), 6.0)), sharex=False)
    if len(out_keys) == 1:
        axes = np.asarray([axes])

    for row_idx, key in enumerate(out_keys):
        subj, t_start = key
        steps = out_steps[row_idx]
        n_current = out_n_current[row_idx]
        stability = out_stability[row_idx]

        axL = axes[row_idx, 0]
        axR = axes[row_idx, 1]

        axL.plot(steps, n_current, color="black", linewidth=1.6)
        axL.set_ylabel("n_current")
        axL.set_title(f"{subj} start={t_start} (counts)")
        axL.grid(True, alpha=0.25)

        axR.plot(steps, stability, color="#d62728", linewidth=1.6)
        axR.set_ylabel("mean logit")
        axR.set_title(f"{subj} start={t_start} (stability)")
        axR.grid(True, alpha=0.25)

        if row_idx == len(out_keys) - 1:
            axL.set_xlabel("step")
            axR.set_xlabel("step")

    fig.suptitle("DIABIMMUNE rollout trajectories — OTU count and stability per step", y=0.995)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")
    print(f"Saved cache: {args.cache}")


if __name__ == "__main__":
    main()
