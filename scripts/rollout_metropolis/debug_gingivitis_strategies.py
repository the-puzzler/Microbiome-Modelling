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

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402
from scripts.rollout_metropolis.core import (  # noqa: E402
    build_otu_index,
    compute_embedding_from_otus,
    pick_anchor_set,
    score_logits_for_sets,
    sigmoid,
)


GINGIVA_CSV = os.path.join("data", "gingivitis", "gingiva.csv")
OUT_DIR = os.path.join("data", "rollout_metropolis", "strategy_debug")


def parse_args():
    p = argparse.ArgumentParser(description="Gingivitis: run alternative single-sample rollout strategies for debugging.")
    p.add_argument("--gingiva-csv", default=GINGIVA_CSV)
    p.add_argument("--subject", default="", help="Subject id (default: first available).")
    p.add_argument("--t-start", default="", help="Start timepoint (default: first available for subject).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--p-anchor", type=float, default=0.95, help="Anchor definition threshold on sigmoid(logit/T).")
    p.add_argument("--candidate-pool", type=int, default=200, help="Uniform candidate pool size to score per step.")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument(
        "--strategy",
        choices=[
            "oneoutonein",
            "oneoutoneinanchored",
            "likelihooddropperanchored",
            "likelihooddropper",
            "nolimitlikelihooddropper",
            "nolimitlikelihooddropperanchored",
            "mh_sizecap",
            "mh_plateau",
        ],
        default="oneoutonein",
    )
    p.add_argument("--run-all", action="store_true", help="Run all strategies (writes multiple TSV/PNG files).")
    p.add_argument("--p-add", type=float, default=0.34, help="MH only: add probability.")
    p.add_argument("--p-drop", type=float, default=0.33, help="MH only: drop probability.")
    p.add_argument("--p-swap", type=float, default=0.33, help="MH only: swap probability.")
    p.add_argument("--n-proposals", type=int, default=10, help="MH only: proposals per step (batched scoring).")
    p.add_argument("--early-stop-nochange", type=int, default=0, help="Stop if no changes for N consecutive steps (0 = off).")
    p.add_argument("--plateau-window", type=int, default=200, help="mh_plateau: window size for slope check.")
    p.add_argument("--plateau-slope-tol", type=float, default=1e-5, help="mh_plateau: stop if |slope| below tol.")
    p.add_argument("--plateau-patience", type=int, default=2, help="mh_plateau: number of consecutive windows.")
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


def mean_logit(logits):
    vals = [float(v) for v in logits.values()]
    return float(np.mean(vals)) if vals else float("-inf")


def sample_absent_with_embedding(rng, all_otus, current, emb_group, resolver, n, max_tries=10000):
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        o = all_otus[int(rng.integers(0, len(all_otus)))]
        if o in current:
            continue
        key = resolver.get(o, o) if resolver else o
        if key not in emb_group:
            continue
        out.append(o)
    if not out:
        return []
    # de-dup in case of repeats
    return list(dict.fromkeys(out))


def choose_drop_lowest_prob(current, logits, temperature):
    best = None
    best_p = 1.0
    for o in current:
        if o not in logits:
            return o
        p = float(sigmoid(float(logits[o]) / float(temperature)))
        if p < best_p:
            best_p = p
            best = o
    return best


def choose_drop_lowest_prob_non_anchor(current, logits, temperature, anchor_set):
    anchor_set = set(anchor_set or [])
    best = None
    best_p = 1.0
    for o in current:
        if o in anchor_set:
            continue
        if o not in logits:
            return o
        p = float(sigmoid(float(logits[o]) / float(temperature)))
        if p < best_p:
            best_p = p
            best = o
    return best


def strategy_oneoutonein(
    *,
    current,
    all_otus,
    model,
    device,
    emb_group,
    resolver,
    rng,
    temperature,
    candidate_pool,
):
    logits_cur = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
    drop = choose_drop_lowest_prob(current, logits_cur, temperature)
    if drop is None or len(current) <= 1:
        return current
    base = set(current)
    base.remove(drop)

    cands = sample_absent_with_embedding(rng, all_otus, base, emb_group, resolver, candidate_pool)
    if not cands:
        return base

    proposals = [sorted(base | {c}) for c in cands]
    scored = score_logits_for_sets(proposals, model, device, emb_group, resolver)
    cand_logits = []
    for i, c in enumerate(cands):
        l = scored[i].get(c, float("-inf"))
        cand_logits.append(float(l))
    best_idx = int(np.argmax(np.asarray(cand_logits, dtype=float)))
    return set(proposals[best_idx])


def strategy_oneoutonein_anchored(
    *,
    current,
    anchor_set,
    all_otus,
    model,
    device,
    emb_group,
    resolver,
    rng,
    temperature,
    candidate_pool,
):
    logits_cur = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
    drop = choose_drop_lowest_prob_non_anchor(current, logits_cur, temperature, anchor_set)
    if drop is None:
        return set(current)
    if len(current) <= max(1, len(anchor_set)):
        return set(current)
    base = set(current)
    base.remove(drop)

    cands = sample_absent_with_embedding(rng, all_otus, base, emb_group, resolver, candidate_pool)
    if not cands:
        return base

    proposals = [sorted(base | {c}) for c in cands]
    scored = score_logits_for_sets(proposals, model, device, emb_group, resolver)
    cand_logits = []
    for i, c in enumerate(cands):
        l = scored[i].get(c, float("-inf"))
        cand_logits.append(float(l))
    best_idx = int(np.argmax(np.asarray(cand_logits, dtype=float)))
    return set(proposals[best_idx])


def strategy_likelihood_dropper(
    *,
    current,
    all_otus,
    model,
    device,
    emb_group,
    resolver,
    rng,
    temperature,
    candidate_pool,
    anchor_set=None,
):
    logits_cur = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
    anchor_set = set(anchor_set or [])

    to_drop = []
    for o in current:
        if o in anchor_set:
            continue
        if o not in logits_cur:
            to_drop.append(o)
            continue
        p = float(sigmoid(float(logits_cur[o]) / float(temperature)))
        if rng.random() < (1.0 - p):
            to_drop.append(o)

    new = set(current)
    for o in to_drop:
        if len(new) <= max(1, len(anchor_set)):
            break
        new.remove(o)
    n_drop = len(current) - len(new)
    if n_drop <= 0:
        return new

    cands = sample_absent_with_embedding(rng, all_otus, new, emb_group, resolver, candidate_pool)
    if not cands:
        return new

    proposals = [sorted(new | {c}) for c in cands]
    scored = score_logits_for_sets(proposals, model, device, emb_group, resolver)

    accepted = 0
    for i, c in enumerate(cands):
        if accepted >= n_drop:
            break
        l = scored[i].get(c)
        if l is None:
            continue
        p = float(sigmoid(float(l) / float(temperature)))
        if rng.random() < p:
            new.add(c)
            accepted += 1
    return new


def strategy_nolimit_likelihood_dropper(
    *,
    current,
    all_otus,
    model,
    device,
    emb_group,
    resolver,
    rng,
    temperature,
    candidate_pool,
    anchor_set=None,
):
    logits_cur = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
    anchor_set = set(anchor_set or [])

    to_drop = []
    for o in current:
        if o in anchor_set:
            continue
        if o not in logits_cur:
            to_drop.append(o)
            continue
        p = float(sigmoid(float(logits_cur[o]) / float(temperature)))
        if rng.random() < (1.0 - p):
            to_drop.append(o)

    new = set(current)
    for o in to_drop:
        if len(new) <= max(1, len(anchor_set)):
            break
        new.remove(o)

    cands = sample_absent_with_embedding(rng, all_otus, new, emb_group, resolver, candidate_pool)
    if not cands:
        return new

    proposals = [sorted(new | {c}) for c in cands]
    scored = score_logits_for_sets(proposals, model, device, emb_group, resolver)

    for i, c in enumerate(cands):
        l = scored[i].get(c)
        if l is None:
            continue
        p = float(sigmoid(float(l) / float(temperature)))
        if rng.random() < p:
            new.add(c)
    return new


def strategy_mh(
    *,
    current,
    all_otus,
    model,
    device,
    emb_group,
    resolver,
    rng,
    temperature,
    p_add,
    p_drop,
    p_swap,
    n_proposals,
    size_cap=None,
):
    weights = np.asarray([p_add, p_drop, p_swap], dtype=float)
    weights = weights / np.sum(weights)

    def sample_absent(cur):
        if len(cur) >= len(all_otus):
            return None
        for _ in range(1000):
            o = all_otus[int(rng.integers(0, len(all_otus)))]
            if o not in cur:
                return o
        absent = [o for o in all_otus if o not in cur]
        return rng.choice(np.asarray(absent, dtype=object)) if absent else None

    cur = set(current)
    proposals = []
    for _ in range(max(1, int(n_proposals))):
        move = rng.choice(["add", "drop", "swap"], p=weights)
        if move == "add" and len(cur) >= len(all_otus):
            move = "drop"
        if move == "drop" and len(cur) <= 1:
            move = "add"

        proposed = set(cur)
        if move == "add":
            o = sample_absent(proposed)
            if o is not None:
                proposed.add(o)
        elif move == "drop" and len(proposed) > 1:
            proposed.remove(rng.choice(np.asarray(list(proposed), dtype=object)))
        elif move == "swap" and len(proposed) > 1:
            proposed.remove(rng.choice(np.asarray(list(proposed), dtype=object)))
            o = sample_absent(proposed)
            if o is not None:
                proposed.add(o)

        if size_cap is not None and len(proposed) > int(size_cap):
            proposed = set(cur)
        proposals.append(sorted(proposed))

    scored = score_logits_for_sets([sorted(cur)] + proposals, model, device, emb_group, resolver)
    j_cur = mean_logit(scored[0])
    j_props = [mean_logit(d) for d in scored[1:]]

    best_idx = int(np.nanargmax(np.asarray(j_props, dtype=float)))
    proposed_best = set(proposals[best_idx])
    j_best = float(j_props[best_idx])
    delta = j_best - j_cur

    accept = False
    if delta >= 0:
        accept = True
    else:
        if rng.random() < np.exp(delta / float(temperature)):
            accept = True
    return proposed_best if accept else cur, j_cur


def run_single_strategy(
    *,
    strategy,
    subject,
    t_start,
    start_otus,
    micro_to_otus,
    seed,
    steps,
    temperature,
    p_anchor,
    candidate_pool,
    p_add,
    p_drop,
    p_swap,
    n_proposals,
    plateau_window,
    plateau_slope_tol,
    plateau_patience,
    early_stop_nochange,
    out_tsv,
    out_png,
):
    all_otus, otu_to_idx = build_otu_index(micro_to_otus)
    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")
    rng = np.random.default_rng(seed)

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]

        current = set(start_otus)
        if not current:
            raise SystemExit("Empty start community.")

        anchor_set = set()
        if strategy in ("likelihooddropperanchored", "oneoutoneinanchored"):
            logits0 = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
            anchor_set = set(
                pick_anchor_set(sorted(current), logits0, p_threshold=p_anchor, temperature=temperature)
            )
            if not anchor_set:
                anchor_set = {sorted(current)[0]}

        fieldnames = [
            "strategy",
            "subject",
            "t_start",
            "step",
            "n_current",
            "n_same",
            "n_added",
            "n_removed",
            "mean_logit",
            "mean_prob",
            "n_anchors",
            "anchor_otu_indices",
            "current_otu_indices",
        ]
        os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
        with open(out_tsv, "w", newline="") as out_f:
            w = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()

            def record(step_idx, prev_set):
                logits = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
                mlog = mean_logit(logits)
                probs = [float(sigmoid(float(v) / float(temperature))) for v in logits.values()]
                mprob = float(np.mean(probs)) if probs else float("-inf")
                n_same = int(len(prev_set & current))
                n_added = int(len(current - prev_set))
                n_removed = int(len(prev_set - current))
                cur_idxs = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]
                anc_idxs = [otu_to_idx[o] for o in sorted(anchor_set) if o in otu_to_idx] if anchor_set else []
                w.writerow(
                    {
                        "strategy": strategy,
                        "subject": subject,
                        "t_start": t_start,
                        "step": int(step_idx),
                        "n_current": int(len(current)),
                        "n_same": n_same,
                        "n_added": n_added,
                        "n_removed": n_removed,
                        "mean_logit": float(mlog),
                        "mean_prob": float(mprob),
                        "n_anchors": int(len(anc_idxs)),
                        "anchor_otu_indices": ";".join(str(i) for i in anc_idxs),
                        "current_otu_indices": ";".join(str(i) for i in cur_idxs),
                    }
                )

            prev = set(current)
            record(0, prev)

            plateau_hits = 0
            obj_hist = []
            nochange_run = 0

            for step_idx in tqdm(range(1, int(steps) + 1), desc=f"{strategy}", unit="step", dynamic_ncols=True):
                prev = set(current)

                if strategy == "oneoutonein":
                    current = strategy_oneoutonein(
                        current=current,
                        all_otus=all_otus,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        resolver=resolver,
                        rng=rng,
                        temperature=temperature,
                        candidate_pool=candidate_pool,
                    )
                elif strategy == "oneoutoneinanchored":
                    current = strategy_oneoutonein_anchored(
                        current=current,
                        anchor_set=anchor_set,
                        all_otus=all_otus,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        resolver=resolver,
                        rng=rng,
                        temperature=temperature,
                        candidate_pool=candidate_pool,
                    )
                elif strategy == "likelihooddropperanchored":
                    current = strategy_likelihood_dropper(
                        current=current,
                        all_otus=all_otus,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        resolver=resolver,
                        rng=rng,
                        temperature=temperature,
                        candidate_pool=candidate_pool,
                        anchor_set=anchor_set,
                    )
                elif strategy == "likelihooddropper":
                    current = strategy_likelihood_dropper(
                        current=current,
                        all_otus=all_otus,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        resolver=resolver,
                        rng=rng,
                        temperature=temperature,
                        candidate_pool=candidate_pool,
                        anchor_set=None,
                    )
                elif strategy == "nolimitlikelihooddropper":
                    current = strategy_nolimit_likelihood_dropper(
                        current=current,
                        all_otus=all_otus,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        resolver=resolver,
                        rng=rng,
                        temperature=temperature,
                        candidate_pool=candidate_pool,
                        anchor_set=None,
                    )
                elif strategy == "nolimitlikelihooddropperanchored":
                    current = strategy_nolimit_likelihood_dropper(
                        current=current,
                        all_otus=all_otus,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        resolver=resolver,
                        rng=rng,
                        temperature=temperature,
                        candidate_pool=candidate_pool,
                        anchor_set=anchor_set,
                    )
                elif strategy == "mh_sizecap":
                    current, obj = strategy_mh(
                        current=current,
                        all_otus=all_otus,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        resolver=resolver,
                        rng=rng,
                        temperature=temperature,
                        p_add=p_add,
                        p_drop=p_drop,
                        p_swap=p_swap,
                        n_proposals=n_proposals,
                        size_cap=len(start_otus),
                    )
                    obj_hist.append(float(obj))
                elif strategy == "mh_plateau":
                    current, obj = strategy_mh(
                        current=current,
                        all_otus=all_otus,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        resolver=resolver,
                        rng=rng,
                        temperature=temperature,
                        p_add=p_add,
                        p_drop=p_drop,
                        p_swap=p_swap,
                        n_proposals=n_proposals,
                        size_cap=None,
                    )
                    obj_hist.append(float(obj))
                    if len(obj_hist) >= plateau_window:
                        y = np.asarray(obj_hist[-plateau_window:], dtype=float)
                        x = np.arange(len(y), dtype=float)
                        if np.all(np.isfinite(y)) and np.std(y) > 0:
                            slope = float(np.polyfit(x, y, deg=1)[0])
                            if abs(slope) < float(plateau_slope_tol):
                                plateau_hits += 1
                            else:
                                plateau_hits = 0
                        if plateau_hits >= int(plateau_patience):
                            record(step_idx, prev)
                            break
                else:
                    raise SystemExit(f"Unknown strategy: {strategy}")

                record(step_idx, prev)
                if current == prev:
                    nochange_run += 1
                else:
                    nochange_run = 0
                if early_stop_nochange and nochange_run >= int(early_stop_nochange):
                    break

        # Plot: n_current + mean_prob, and trajectories (embedding PCA + Jaccard PCoA) for the sampled steps.
        rows = []
        with open(out_tsv, "r", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                rows.append(row)
        steps_arr = np.asarray([int(r["step"]) for r in rows], dtype=int)
        n_current_arr = np.asarray([int(r["n_current"]) for r in rows], dtype=int)
        mean_prob_arr = np.asarray([float(r["mean_prob"]) for r in rows], dtype=float)

        idx_lists = [
            np.asarray([int(tok) for tok in str(r["current_otu_indices"]).split(";") if str(tok).strip()], dtype=np.int32) for r in rows
        ]

        # Convergence diagnostics in OTU-set space
        end_set = set(map(int, idx_lists[-1].tolist()))
        j_to_final = []
        for idx in idx_lists:
            s = set(map(int, idx.tolist()))
            inter = len(s & end_set)
            union = len(s) + len(end_set) - inter
            j = (float(inter) / float(union)) if union else 1.0
            j_to_final.append(j)
        j_to_final = np.asarray(j_to_final, dtype=float)

        # Subsample for trajectory plots
        max_pts = 250
        if len(idx_lists) > max_pts:
            keep = np.unique(np.round(np.linspace(0, len(idx_lists) - 1, max_pts)).astype(int))
            steps_tr = steps_arr[keep]
            idx_lists_tr = [idx_lists[i] for i in keep.tolist()]
        else:
            steps_tr = steps_arr
            idx_lists_tr = idx_lists

        otus_tr = [[all_otus[i] for i in idx.tolist() if 0 <= i < len(all_otus)] for idx in idx_lists_tr]
        traj_embs = []
        for otus in otus_tr:
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
        traj_xy = pca2_transform(traj_embs, mu_p, comps_p).astype(np.float32)

        traj_sets = [set(map(int, idx.tolist())) for idx in idx_lists_tr]
        D = jaccard_distance_matrix(traj_sets)
        traj_j_xy = pcoa2_from_distance(D)

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(13.0, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.1, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 0])

    ax0.plot(steps_arr, n_current_arr, color="black", linewidth=2.0, label="n_current")
    ax0.set_xlabel("step")
    ax0.set_ylabel("n_current")
    ax0.grid(True, alpha=0.25)
    ax0.set_title(f"{strategy} ({subject}, start={t_start})")
    ax0b = ax0.twinx()
    ax0b.plot(steps_arr, mean_prob_arr, color="#d62728", linewidth=1.6, alpha=0.85, label="mean_prob")
    ax0b.plot(steps_arr, j_to_final, color="#1f77b4", linewidth=1.4, alpha=0.85, label="Jaccard to final")
    ax0b.axhline(0.99, color="#1f77b4", linewidth=1.0, alpha=0.25, linestyle="--")
    ax0b.set_ylabel("mean prob")
    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax0b.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, frameon=False, loc="best")

    # Trajectory plots (color by step progression)
    cvals = np.linspace(0.0, 1.0, max(2, len(steps_tr)))
    cmap = plt.cm.viridis

    ax1.plot(traj_xy[:, 0], traj_xy[:, 1], color="0.35", alpha=0.35, linewidth=1.0, zorder=1)
    ax1.scatter(traj_xy[:, 0], traj_xy[:, 1], c=cmap(cvals), s=14, alpha=0.9, linewidths=0, zorder=2)
    ax1.scatter([traj_xy[0, 0]], [traj_xy[0, 1]], s=60, facecolors="none", edgecolors="black", linewidths=1.0, zorder=3)
    ax1.scatter([traj_xy[-1, 0]], [traj_xy[-1, 1]], s=60, facecolors="none", edgecolors="#d62728", linewidths=1.0, zorder=3)
    ax1.set_title("Trajectory (embedding PCA)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect("equal", adjustable="datalim")

    ax2.plot(traj_j_xy[:, 0], traj_j_xy[:, 1], color="0.35", alpha=0.35, linewidth=1.0, zorder=1)
    ax2.scatter(traj_j_xy[:, 0], traj_j_xy[:, 1], c=cmap(cvals), s=14, alpha=0.9, linewidths=0, zorder=2)
    ax2.scatter([traj_j_xy[0, 0]], [traj_j_xy[0, 1]], s=60, facecolors="none", edgecolors="black", linewidths=1.0, zorder=3)
    ax2.scatter([traj_j_xy[-1, 0]], [traj_j_xy[-1, 1]], s=60, facecolors="none", edgecolors="#d62728", linewidths=1.0, zorder=3)
    ax2.set_title("Trajectory (OTU Jaccard PCoA)")
    ax2.set_xlabel("PCoA1")
    ax2.set_ylabel("PCoA2")
    ax2.grid(True, alpha=0.2)
    ax2.set_aspect("equal", adjustable="datalim")

    # Histogram: start vs end logits distribution
    # Re-open embedding file here; the earlier emb_group comes from a closed h5py file.
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file2:
        emb_group2 = emb_file2["embeddings"]
        logits_start = score_logits_for_sets([sorted(set(start_otus))], model, device, emb_group2, resolver)[0]
        end_otus = [all_otus[i] for i in idx_lists[-1].tolist() if 0 <= i < len(all_otus)]
        logits_end = score_logits_for_sets([sorted(set(end_otus))], model, device, emb_group2, resolver)[0]
    start_vals = np.asarray(list(logits_start.values()), dtype=float) if logits_start else np.asarray([], dtype=float)
    end_vals = np.asarray(list(logits_end.values()), dtype=float) if logits_end else np.asarray([], dtype=float)

    bins = 50
    if start_vals.size and end_vals.size:
        lo = float(min(np.min(start_vals), np.min(end_vals)))
        hi = float(max(np.max(start_vals), np.max(end_vals)))
        edges = np.linspace(lo, hi, bins + 1)
    else:
        edges = bins

    if start_vals.size or end_vals.size:
        ax3.hist(start_vals, bins=edges, alpha=0.5, label="start", color="#1f77b4")
        ax3.hist(end_vals, bins=edges, alpha=0.5, label="end", color="#ff7f0e")
        ax3.legend(frameon=False)
    else:
        ax3.text(0.5, 0.5, "No logits available (check embedding keys/resolver)", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("Logit distribution (start vs end)")
    ax3.set_xlabel("logit")
    ax3.set_ylabel("count")
    ax3.grid(True, alpha=0.15)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


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

    strategies = (
        [
            "oneoutonein",
            "oneoutoneinanchored",
            "likelihooddropperanchored",
            "likelihooddropper",
            "nolimitlikelihooddropper",
            "nolimitlikelihooddropperanchored",
            "mh_sizecap",
            "mh_plateau",
        ]
        if args.run_all
        else [args.strategy]
    )

    for strat in strategies:
        base = f"{subject}_{t_start}_{strat}_seed{args.seed}_steps{args.steps}"
        out_tsv = os.path.join(args.out_dir, base + ".tsv")
        out_png = os.path.join(args.out_dir, base + ".png")
        run_single_strategy(
            strategy=strat,
            subject=subject,
            t_start=t_start,
            start_otus=start_otus,
            micro_to_otus=micro_to_otus,
            seed=args.seed,
            steps=args.steps,
            temperature=args.temperature,
            p_anchor=args.p_anchor,
            candidate_pool=args.candidate_pool,
            p_add=args.p_add,
            p_drop=args.p_drop,
            p_swap=args.p_swap,
            n_proposals=args.n_proposals,
            plateau_window=args.plateau_window,
            plateau_slope_tol=args.plateau_slope_tol,
            plateau_patience=args.plateau_patience,
            early_stop_nochange=args.early_stop_nochange,
            out_tsv=out_tsv,
            out_png=out_png,
        )
        print(f"Saved: {out_tsv}")
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
