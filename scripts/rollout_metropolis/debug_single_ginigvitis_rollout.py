#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402
from scripts.rollout_metropolis.core import (  # noqa: E402
    build_otu_index,
    compute_embedding_from_otus,
    metropolis_steps_fixed_anchors,
    pick_anchor_set,
    score_logits_for_sets,
    sigmoid,
    write_rollout_tsv,
)
from scripts import utils as shared_utils  # noqa: E402


GINGIVA_CSV = os.path.join("data", "gingivitis", "gingiva.csv")
OUT_TSV = os.path.join("data", "gingivitis", "single_rollout_metropolis_debug.tsv")
OUT_PNG = os.path.join("data", "rollout_metropolis", "gingivitis_single_rollout_metropolis_debug.png")


def parse_args():
    p = argparse.ArgumentParser(description="Run a single long Metropolis rollout for gingivitis and plot diagnostics.")
    p.add_argument("--gingiva-csv", default=GINGIVA_CSV)
    p.add_argument("--subject", default="", help="Subject id (default: first available).")
    p.add_argument("--t-start", default="", help="Start timepoint (default: first available for subject).")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--p-anchor", type=float, default=0.95)
    p.add_argument(
        "--density-tau",
        type=float,
        default=0.95,
        help="Threshold τ used by the above-τ density objective (p(otu) >= τ).",
    )
    p.add_argument("--p-add", type=float, default=0.34)
    p.add_argument("--p-drop", type=float, default=0.33)
    p.add_argument("--p-swap", type=float, default=0.33)
    p.add_argument("--n-proposals", type=int, default=10)
    p.add_argument(
        "--target",
        choices=["anchors", "community"],
        default="anchors",
        help="What to optimize: fixed anchor set (default) or the whole current community.",
    )
    p.add_argument(
        "--accept",
        choices=["metropolis", "greedy"],
        default="metropolis",
        help="Accept rule: Metropolis (can accept worse) or greedy (only take best Δ>0 else pass).",
    )
    p.add_argument(
        "--objective",
        choices=["anchor_mean_logit", "anchor_prob_over_ncurrent", "above_tau_density"],
        default="anchor_prob_over_ncurrent",
        help="Objective used for MH accept/reject (debug only).",
    )
    p.add_argument("--out-tsv", default=OUT_TSV)
    p.add_argument("--out-png", default=OUT_PNG)
    return p.parse_args()


def time_key(t):
    s = str(t).strip()
    if s.upper() == "B":
        return -1e9
    return float(s)


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


def load_debug_series(tsv_path):
    steps, n_current, anchor_mean_logit, anchor_prob_over_ncurrent, above_tau_density = [], [], [], [], []
    with open(tsv_path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            steps.append(int(row["step"]))
            n_current.append(int(row["n_current"]))
            anchor_mean_logit.append(float(row["anchor_mean_logit"]) if row.get("anchor_mean_logit", "") != "" else np.nan)
            anchor_prob_over_ncurrent.append(
                float(row["anchor_prob_over_ncurrent"]) if row.get("anchor_prob_over_ncurrent", "") != "" else np.nan
            )
            above_tau_density.append(float(row["above_tau_density"]) if row.get("above_tau_density", "") != "" else np.nan)
    return (
        np.asarray(steps),
        np.asarray(n_current),
        np.asarray(anchor_mean_logit),
        np.asarray(anchor_prob_over_ncurrent),
        np.asarray(above_tau_density),
    )


def load_last_state(tsv_path):
    last = None
    with open(tsv_path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            last = row
    if last is None:
        return None
    return last


def load_all_rows(tsv_path):
    rows = []
    with open(tsv_path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    return rows


def parse_index_list(raw):
    if not raw:
        return []
    out = []
    for tok in str(raw).split(";"):
        tok = str(tok).strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            return []
    return out


def _extract_indices_from_row(row, key):
    """
    Robustly extract semicolon-separated integer indices from a DictReader row.
    Handles legacy TSVs where extra columns were appended without updating header
    (DictReader stores extras under the None key).
    """
    raw = row.get(key, "")
    idx = parse_index_list(raw)
    if idx:
        return idx
    extras = row.get(None, []) or []
    for extra in extras:
        idx = parse_index_list(extra)
        if idx:
            return idx
    for v in row.values():
        idx = parse_index_list(v)
        if idx:
            return idx
    return []

def objective_from_logits(
    *,
    anchor_set,
    logits,
    temperature,
    n_current,
    mode,
    density_tau,
):
    if mode == "above_tau_density":
        vals = [float(v) for v in logits.values()]
        if not vals:
            return float("-inf")
        probs = [sigmoid(v / float(temperature)) for v in vals]
        n_above = int(np.sum(np.asarray(probs) >= float(density_tau)))
        return float(n_above / float(max(1, int(n_current))))

    if anchor_set is None:
        vals = [float(v) for v in logits.values()]
    else:
        if not anchor_set:
            return float("-inf")
        vals = [float(logits[a]) for a in anchor_set if a in logits]
    if not vals:
        return float("-inf")

    if mode == "anchor_mean_logit":
        return float(np.mean(vals))

    mean_prob = float(np.mean([sigmoid(v / float(temperature)) for v in vals]))
    return float(mean_prob / float(max(1, int(n_current))))


def require_tsv_schema_for_append(path, required_fieldnames):
    with open(path, "r", newline="") as f:
        header = f.readline().rstrip("\n")
    cols = header.split("\t") if header else []
    missing = [c for c in required_fieldnames if c not in cols]
    if missing:
        miss = ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")
        raise SystemExit(
            f"Existing TSV header is missing required columns for appending: {miss}. "
            f"Delete the TSV or pick a new --out-tsv."
        )


def pca2_fit(X):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, vt[:2]


def pca2_transform(X, mu, comps):
    X = np.asarray(X, dtype=float)
    return (X - mu) @ comps.T


def pcoa2_from_distance(D):
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")
    D2 = D * D
    row_mean = D2.mean(axis=1)
    grand_mean = float(D2.mean())
    B = -0.5 * (D2 - row_mean[:, None] - row_mean[None, :] + grand_mean)
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    w_pos = w[w > 1e-12]
    if w_pos.size < 2:
        raise ValueError("Not enough positive eigenvalues for 2D PCoA")
    eigvals = w_pos[:2]
    X = v[:, :2] * np.sqrt(eigvals)[None, :]
    return X.astype(np.float32)


def jaccard_distance_matrix(sets):
    n = len(sets)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        si = sets[i]
        for j in range(i + 1, n):
            sj = sets[j]
            inter = len(si & sj)
            union = len(si) + len(sj) - inter
            d = 1.0 - (float(inter) / float(union)) if union else 0.0
            D[i, j] = d
            D[j, i] = d
    return D


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

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

    if not os.path.exists(args.out_tsv) and args.objective == "anchor_mean_logit":
        write_rollout_tsv(
            out_tsv=args.out_tsv,
            starts=[(subject, t_start, start_otus)],
            micro_to_otus=micro_to_otus,
            seed=args.seed,
            steps_per_rollout=args.steps,
            temperature=args.temperature,
            checkpoint_path=shared_utils.CHECKPOINT_PATH,
            prokbert_path=shared_utils.PROKBERT_PATH,
            rename_map_path=shared_utils.RENAME_MAP_PATH,
            prefer_resolver="B",
            p_anchor=args.p_anchor,
            p_add=args.p_add,
            p_drop=args.p_drop,
            p_swap=args.p_swap,
            n_proposals=args.n_proposals,
        )
        print(f"Saved TSV: {args.out_tsv}")
    else:
        # For the size-normalized objective, run a local debug rollout so we can change the accept/reject rule
        # without touching the main rollout pipeline.
        all_otus, otu_to_idx = build_otu_index(micro_to_otus)
        model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
        rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
        resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

        rng = np.random.default_rng(args.seed)

        step_offset = 0
        current_otus = list(start_otus)
        anchor_otus = None
        if os.path.exists(args.out_tsv):
            last = load_last_state(args.out_tsv)
            if last is None:
                raise SystemExit("Existing debug TSV is empty.")
            step_offset = int(last.get("step", "0"))
            if step_offset >= args.steps:
                print(f"TSV already has step {step_offset} >= requested steps {args.steps}; not extending.")
            else:
                cur_idxs = parse_index_list(last.get("current_otu_indices", ""))
                current_otus = [all_otus[i] for i in cur_idxs if 0 <= i < len(all_otus)]
                if not current_otus:
                    raise SystemExit("Could not reconstruct current OTU set from TSV.")
                if args.target == "anchors":
                    anchor_idxs = parse_index_list(last.get("anchor_otu_indices", ""))
                    anchor_otus = [all_otus[i] for i in anchor_idxs if 0 <= i < len(all_otus)]
                    if not anchor_otus:
                        raise SystemExit("Could not reconstruct anchor OTU set from TSV.")
                rng = np.random.default_rng(args.seed + step_offset + 1)

        remaining = max(0, int(args.steps) - int(step_offset))
        if remaining > 0:
            fieldnames = [
                "subject",
                "t_start",
                "step",
                "n_current",
                "n_same",
                "n_added",
                "n_removed",
                "n_anchors",
                "anchor_otu_indices",
                "anchor_mean_logit",
                "anchor_prob_over_ncurrent",
                "density_tau",
                "n_above_tau",
                "above_tau_density",
                "current_otu_indices",
            ]

            mode = args.objective
            if os.path.exists(args.out_tsv):
                require_tsv_schema_for_append(args.out_tsv, fieldnames)
            with open(args.out_tsv, "a" if os.path.exists(args.out_tsv) else "w", newline="") as out_f, h5py.File(
                shared_utils.PROKBERT_PATH
            ) as emb_file:
                emb_group = emb_file["embeddings"]
                writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
                if out_f.tell() == 0:
                    writer.writeheader()

                current = set(current_otus)
                anchor_set = None
                anchor_indices = []
                anchor_indices_str = ""

                if args.target == "anchors":
                    if anchor_otus is None:
                        logits0 = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
                        anchor_otus = sorted(
                            pick_anchor_set(
                                sorted(current),
                                logits0,
                                p_threshold=args.p_anchor,
                                temperature=args.temperature,
                            )
                        )
                    anchor_set = set(anchor_otus)
                    anchor_indices = sorted({otu_to_idx[o] for o in anchor_set if o in otu_to_idx})
                    anchor_indices_str = ";".join(str(i) for i in anchor_indices)

                weights = np.asarray([args.p_add, args.p_drop, args.p_swap], dtype=float)
                weights = weights / np.sum(weights)

                def sample_absent(cur_set):
                    if len(cur_set) >= len(all_otus):
                        return None
                    for _ in range(1000):
                        o = all_otus[int(rng.integers(0, len(all_otus)))]
                        if o not in cur_set:
                            return o
                    absent = [o for o in all_otus if o not in cur_set]
                    return rng.choice(np.asarray(absent, dtype=object)) if absent else None

                for local_step in tqdm(range(remaining), desc="Metropolis debug", unit="step", dynamic_ncols=True):
                    prev = set(current)
                    if args.target == "anchors":
                        candidates_drop = [o for o in current if o not in anchor_set]
                    else:
                        candidates_drop = list(current)

                    proposals = []
                    for _ in range(max(1, int(args.n_proposals))):
                        move = rng.choice(["add", "drop", "swap"], p=weights)
                        if move == "add" and len(current) >= len(all_otus):
                            move = "drop" if candidates_drop else "swap"
                        if move == "drop" and (not candidates_drop or len(current) <= 1):
                            move = "add" if len(current) < len(all_otus) else "swap"
                        if move == "swap" and (len(current) >= len(all_otus) or not candidates_drop or len(current) <= 1):
                            move = "add" if len(current) < len(all_otus) else "drop"

                        proposed = set(current)
                        if move == "add":
                            o = sample_absent(proposed)
                            if o is not None:
                                proposed.add(o)
                        elif move == "drop" and candidates_drop and len(proposed) > 1:
                            proposed.remove(rng.choice(np.asarray(candidates_drop, dtype=object)))
                        elif move == "swap" and candidates_drop and len(proposed) > 1:
                            proposed.remove(rng.choice(np.asarray(candidates_drop, dtype=object)))
                            o = sample_absent(proposed)
                            if o is not None:
                                proposed.add(o)
                        proposals.append(sorted(proposed))

                    scored = score_logits_for_sets([sorted(current)] + proposals, model, device, emb_group, resolver)
                    cur_obj = objective_from_logits(
                        anchor_set=None if args.target == "community" else anchor_set,
                        logits=scored[0],
                        temperature=args.temperature,
                        n_current=len(current),
                        mode=mode,
                        density_tau=args.density_tau,
                    )
                    prop_objs = [
                        objective_from_logits(
                            anchor_set=None if args.target == "community" else anchor_set,
                            logits=scored[i + 1],
                            temperature=args.temperature,
                            n_current=len(proposals[i]),
                            mode=mode,
                            density_tau=args.density_tau,
                        )
                        for i in range(len(proposals))
                    ]
                    best_idx = int(np.nanargmax(np.asarray(prop_objs, dtype=float)))
                    best_obj = float(prop_objs[best_idx])
                    proposed_best = set(proposals[best_idx])
                    delta = best_obj - cur_obj

                    accept = False
                    if args.accept == "greedy":
                        accept = delta > 0
                    else:
                        if delta >= 0:
                            accept = True
                        else:
                            if rng.random() < np.exp(delta / float(args.temperature)):
                                accept = True
                    if accept:
                        current = proposed_best

                    # Record diagnostics for the accepted state.
                    logits_acc = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
                    if args.target == "community":
                        anchor_mean_logit = float(np.mean([float(v) for v in logits_acc.values()])) if logits_acc else float("nan")
                    else:
                        anchor_mean_logit = float(np.mean([float(logits_acc[a]) for a in anchor_set if a in logits_acc]))

                    anchor_prob_over_n = objective_from_logits(
                        anchor_set=None if args.target == "community" else anchor_set,
                        logits=logits_acc,
                        temperature=args.temperature,
                        n_current=len(current),
                        mode="anchor_prob_over_ncurrent",
                        density_tau=args.density_tau,
                    )

                    probs_acc = np.asarray([sigmoid(float(v) / float(args.temperature)) for v in logits_acc.values()], dtype=float)
                    n_above_tau = int(np.sum(probs_acc >= float(args.density_tau))) if probs_acc.size else 0
                    above_tau_density = float(n_above_tau / float(max(1, int(len(current)))))

                    n_same = int(len(prev & current))
                    n_added = int(len(current - prev))
                    n_removed = int(len(prev - current))
                    current_indices = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]

                    writer.writerow(
                        {
                            "subject": subject,
                            "t_start": t_start,
                            "step": int(step_offset) + local_step + 1,
                            "n_current": int(len(current)),
                            "n_same": n_same,
                            "n_added": n_added,
                            "n_removed": n_removed,
                            "n_anchors": int(len(anchor_indices)) if args.target == "anchors" else 0,
                            "anchor_otu_indices": anchor_indices_str,
                            "anchor_mean_logit": anchor_mean_logit,
                            "anchor_prob_over_ncurrent": float(anchor_prob_over_n),
                            "density_tau": float(args.density_tau),
                            "n_above_tau": int(n_above_tau),
                            "above_tau_density": float(above_tau_density),
                            "current_otu_indices": ";".join(str(i) for i in current_indices),
                        }
                    )

            print(f"Saved TSV: {args.out_tsv}")

    steps, n_current, anchor_mean_logit, anchor_prob_over_ncurrent, above_tau_density = load_debug_series(args.out_tsv)

    # Histogram diagnostics: compare start vs endpoint logit distributions
    all_otus, _otu_to_idx = build_otu_index(micro_to_otus)
    start_set = set(start_otus)

    last = load_last_state(args.out_tsv)
    if last is None:
        raise SystemExit("Debug TSV is empty; cannot build endpoint histogram.")
    end_idxs = _extract_indices_from_row(last, "current_otu_indices")
    end_set = {all_otus[i] for i in end_idxs if 0 <= i < len(all_otus)}

    # Anchor set (if present in TSV)
    anchor_idxs = _extract_indices_from_row(last, "anchor_otu_indices")
    anchor_set = {all_otus[i] for i in anchor_idxs if 0 <= i < len(all_otus)}

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        logits_start = score_logits_for_sets([sorted(start_set)], model, device, emb_group, resolver)[0]
        logits_end = score_logits_for_sets([sorted(end_set)], model, device, emb_group, resolver)[0]

    start_vals = np.asarray(list(logits_start.values()), dtype=float) if logits_start else np.asarray([], dtype=float)
    end_vals = np.asarray(list(logits_end.values()), dtype=float) if logits_end else np.asarray([], dtype=float)

    start_anchor_vals = (
        np.asarray([float(logits_start[a]) for a in anchor_set if a in logits_start], dtype=float)
        if anchor_set and logits_start
        else np.asarray([], dtype=float)
    )
    end_anchor_vals = (
        np.asarray([float(logits_end[a]) for a in anchor_set if a in logits_end], dtype=float)
        if anchor_set and logits_end
        else np.asarray([], dtype=float)
    )

    # Trajectory plots (embedding PCA + OTU-set Jaccard PCoA) for this single run.
    rows_all = load_all_rows(args.out_tsv)
    rows_all = sorted(rows_all, key=lambda r: int(r.get("step", "0")) if str(r.get("step", "")).isdigit() else 0)
    traj_steps = [0] + [int(r.get("step", "0")) for r in rows_all]
    # Build index lists: step 0 uses the original start set.
    start_idx = np.asarray([i for i, o in enumerate(all_otus) if o in start_set], dtype=np.int32)
    traj_idx_lists = [start_idx] + [
        np.asarray([i for i in _extract_indices_from_row(r, "current_otu_indices") if 0 <= i < len(all_otus)], dtype=np.int32)
        for r in rows_all
    ]

    # Subsample for plotting so we don't recompute too many embeddings/distances.
    max_traj_points = 250
    if len(traj_steps) > max_traj_points:
        keep = np.unique(np.round(np.linspace(0, len(traj_steps) - 1, max_traj_points)).astype(int))
    else:
        keep = np.arange(len(traj_steps), dtype=int)
    traj_steps = [traj_steps[i] for i in keep.tolist()]
    traj_idx_lists = [traj_idx_lists[i] for i in keep.tolist()]

    traj_otus = [[all_otus[i] for i in idx.tolist() if 0 <= i < len(all_otus)] for idx in traj_idx_lists]
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        traj_embs = []
        for otus in traj_otus:
            e = compute_embedding_from_otus(
                otus,
                model=model,
                device=device,
                emb_group=emb_group,
                resolver=resolver,
                scratch_tokens=0,
                d_model=shared_utils.D_MODEL,
            )
            if e is None:
                raise SystemExit("Failed to compute embedding for a trajectory point.")
            traj_embs.append(e.astype(np.float32))
    traj_embs = np.stack(traj_embs, axis=0)
    mu_p, comps_p = pca2_fit(traj_embs)
    traj_pca_xy = pca2_transform(traj_embs, mu_p, comps_p).astype(np.float32)

    traj_sets = [set(map(int, idx.tolist())) for idx in traj_idx_lists]
    D = jaccard_distance_matrix(traj_sets)
    traj_jacc_xy = pcoa2_from_distance(D).astype(np.float32)

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(13.0, 12.0))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 1.0, 1.1], width_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, :])
    ax_hist_a = fig.add_subplot(gs[2, :])
    ax_traj_emb = fig.add_subplot(gs[3, 0])
    ax_traj_j = fig.add_subplot(gs[3, 1])

    ax.plot(steps, n_current, color="black", linewidth=2.0, label="n_current")
    ax.set_xlabel("step")
    ax.set_ylabel("n_current")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Gingivitis Metropolis single rollout ({subject}, start={t_start})")

    ax2 = ax.twinx()
    if np.isfinite(above_tau_density).any():
        ax2.plot(steps, above_tau_density, color="#d62728", linewidth=1.6, alpha=0.9, label="density(p>=τ)")
        ax2.set_ylabel("density (p >= τ)")
    else:
        ax2.plot(steps, anchor_mean_logit, color="#d62728", linewidth=1.6, alpha=0.9, label="anchor_mean_logit")
        ax2.set_ylabel("anchor mean logit")

    h0, l0 = ax.get_legend_handles_labels()
    h1, l1 = ax2.get_legend_handles_labels()
    ax.legend(h0 + h1, l0 + l1, frameon=False, loc="best")

    bins = 50
    if start_vals.size and end_vals.size:
        lo = float(min(np.min(start_vals), np.min(end_vals)))
        hi = float(max(np.max(start_vals), np.max(end_vals)))
        edges = np.linspace(lo, hi, bins + 1)
    else:
        edges = bins

    ax_hist.hist(start_vals, bins=edges, alpha=0.5, label="start", color="#1f77b4")
    ax_hist.hist(end_vals, bins=edges, alpha=0.5, label="endpoint", color="#ff7f0e")
    ax_hist.set_title("All present OTUs: logit distribution (start vs endpoint)")
    ax_hist.set_xlabel("logit")
    ax_hist.set_ylabel("count")
    ax_hist.legend(frameon=False)
    ax_hist.grid(True, alpha=0.15)

    if start_anchor_vals.size or end_anchor_vals.size:
        ax_hist_a.hist(start_anchor_vals, bins=edges, alpha=0.5, label="start anchors", color="#1f77b4")
        ax_hist_a.hist(end_anchor_vals, bins=edges, alpha=0.5, label="endpoint anchors", color="#ff7f0e")
        ax_hist_a.set_title("Anchor OTUs only: logit distribution (start vs endpoint)")
        ax_hist_a.legend(frameon=False)
    else:
        ax_hist_a.text(0.5, 0.5, "No anchor set in TSV", ha="center", va="center", transform=ax_hist_a.transAxes)
        ax_hist_a.set_title("Anchor OTUs only")
    ax_hist_a.set_xlabel("logit")
    ax_hist_a.set_ylabel("count")
    ax_hist_a.grid(True, alpha=0.15)

    # Embedding-space trajectory
    cmap = plt.cm.viridis
    cvals = np.linspace(0.0, 1.0, max(2, len(traj_steps)))
    ax_traj_emb.plot(traj_pca_xy[:, 0], traj_pca_xy[:, 1], color="0.35", alpha=0.35, linewidth=1.0, zorder=1)
    ax_traj_emb.scatter(traj_pca_xy[:, 0], traj_pca_xy[:, 1], c=cmap(cvals), s=14, alpha=0.9, linewidths=0, zorder=2)
    ax_traj_emb.scatter([traj_pca_xy[0, 0]], [traj_pca_xy[0, 1]], s=60, facecolors="none", edgecolors="black", linewidths=1.0, zorder=3)
    ax_traj_emb.scatter([traj_pca_xy[-1, 0]], [traj_pca_xy[-1, 1]], s=60, facecolors="none", edgecolors="#d62728", linewidths=1.0, zorder=3)
    ax_traj_emb.set_title("Trajectory (embedding PCA)")
    ax_traj_emb.set_xlabel("PC1")
    ax_traj_emb.set_ylabel("PC2")
    ax_traj_emb.grid(True, alpha=0.2)
    ax_traj_emb.set_aspect("equal", adjustable="datalim")

    # OTU-space trajectory
    ax_traj_j.plot(traj_jacc_xy[:, 0], traj_jacc_xy[:, 1], color="0.35", alpha=0.35, linewidth=1.0, zorder=1)
    ax_traj_j.scatter(traj_jacc_xy[:, 0], traj_jacc_xy[:, 1], c=cmap(cvals), s=14, alpha=0.9, linewidths=0, zorder=2)
    ax_traj_j.scatter([traj_jacc_xy[0, 0]], [traj_jacc_xy[0, 1]], s=60, facecolors="none", edgecolors="black", linewidths=1.0, zorder=3)
    ax_traj_j.scatter([traj_jacc_xy[-1, 0]], [traj_jacc_xy[-1, 1]], s=60, facecolors="none", edgecolors="#d62728", linewidths=1.0, zorder=3)
    ax_traj_j.set_title("Trajectory (OTU Jaccard PCoA)")
    ax_traj_j.set_xlabel("PCoA1")
    ax_traj_j.set_ylabel("PCoA2")
    ax_traj_j.grid(True, alpha=0.2)
    ax_traj_j.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Saved PNG: {args.out_png}")


if __name__ == "__main__":
    main()
