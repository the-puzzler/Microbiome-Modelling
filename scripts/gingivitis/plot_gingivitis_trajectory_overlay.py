#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402


ROLL_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob.tsv")
ENDPOINTS_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_endpoints_cache.npz")
REAL_NPZ = os.path.join("data", "gingivitis", "visionary_rollout_direction_cache.npz")
GINGIVA_CSV = os.path.join("data", "gingivitis", "gingiva.csv")

OUT_PNG = os.path.join("data", "gingivitis", "gingivitis_rollout_trajectory_overlay.png")
CACHE_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_trajectory_overlay_cache.npz")

JACCARD_LANDMARKS = 600


def parse_args():
    p = argparse.ArgumentParser(description="Gingivitis: counts + PC1/PC2 vector field + example trajectories.")
    p.add_argument("--out", default=OUT_PNG)
    p.add_argument("--cache", default=CACHE_NPZ)
    p.add_argument("--subject", default="", help="Subject id (default: first available).")
    p.add_argument("--t-start", default="", help="Start timepoint for rollout (default: first timepoint of subject).")
    p.add_argument("--subsample-frac", type=float, default=0.15, help="Fraction of rollout steps to embed/plot.")
    return p.parse_args()


def pca2_fit(X):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, vt[:2]


def pca2_transform(X, mu, comps):
    X = np.asarray(X, dtype=float)
    return (X - mu) @ comps.T


def time_key(t):
    s = str(t).strip()
    if s.upper() == "B":
        return (-1e9, s)
    return (float(s), s)


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


def _minhash_signatures(index_lists, *, k=64, seed=0):
    rng = np.random.default_rng(seed)
    max_idx = 0
    for arr in index_lists:
        if arr.size:
            max_idx = max(max_idx, int(np.max(arr)))
    p = 2147483647  # 2^31-1, prime
    if max_idx >= p:
        raise ValueError("OTU index too large for chosen prime.")
    a = rng.integers(1, p - 1, size=k, dtype=np.int64)
    b = rng.integers(0, p - 1, size=k, dtype=np.int64)

    sig = np.empty((len(index_lists), k), dtype=np.int64)
    for i, idx in enumerate(index_lists):
        if idx.size == 0:
            sig[i, :] = p
            continue
        vals = (a[:, None] * idx[None, :] + b[:, None]) % p
        sig[i, :] = np.min(vals, axis=1)
    return sig.astype(np.int32, copy=False)


def _minhash_jaccard_distance_matrix(sig_a, sig_b=None, *, block=128):
    sig_a = np.asarray(sig_a)
    if sig_b is None:
        sig_b = sig_a
    else:
        sig_b = np.asarray(sig_b)

    na, k = sig_a.shape
    nb = sig_b.shape[0]
    out = np.empty((na, nb), dtype=np.float32)
    for i0 in range(0, na, block):
        i1 = min(na, i0 + block)
        eq = sig_a[i0:i1, None, :] == sig_b[None, :, :]
        matches = eq.sum(axis=2, dtype=np.int32)
        out[i0:i1, :] = 1.0 - (matches.astype(np.float32) / float(k))
    return out


def _jaccard_distance_sorted_unique(a, b):
    """
    Jaccard distance = 1 - |A∩B|/|A∪B| for sorted unique int arrays.
    """
    na = int(a.size)
    nb = int(b.size)
    if na == 0 and nb == 0:
        return 0.0

    i = 0
    j = 0
    inter = 0
    while i < na and j < nb:
        va = int(a[i])
        vb = int(b[j])
        if va == vb:
            inter += 1
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    union = na + nb - inter
    return 1.0 - (float(inter) / float(union)) if union else 0.0


def _jaccard_distance_matrix(index_lists_a, index_lists_b=None):
    """
    Exact Jaccard distance matrix for lists of index arrays.

    - Each list element can be any 1D int array; it is normalized to sorted unique.
    - If index_lists_b is None, returns a square (n,n) matrix.
    """
    a_lists = [np.unique(np.asarray(x, dtype=np.int32)) for x in index_lists_a]
    if index_lists_b is None:
        n = len(a_lists)
        out = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            out[i, i] = 0.0
            for j in range(i + 1, n):
                d = _jaccard_distance_sorted_unique(a_lists[i], a_lists[j])
                out[i, j] = d
                out[j, i] = d
        return out

    b_lists = [np.unique(np.asarray(x, dtype=np.int32)) for x in index_lists_b]
    na = len(a_lists)
    nb = len(b_lists)
    out = np.empty((na, nb), dtype=np.float32)
    for i in range(na):
        ai = a_lists[i]
        for j in range(nb):
            out[i, j] = _jaccard_distance_sorted_unique(ai, b_lists[j])
    return out


def _pcoa_2d_from_distance(D):
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
        raise ValueError("Not enough positive eigenvalues for 2D PCoA.")
    eigvals = w_pos[:2]
    V2 = v[:, :2]
    X = V2 * np.sqrt(eigvals)[None, :]
    return X.astype(np.float32), eigvals.astype(np.float64), row_mean, grand_mean


def _pcoa_project_out_of_sample(d_to_train, X_train, eigvals, train_row_mean, train_grand_mean):
    d = np.asarray(d_to_train, dtype=np.float64)
    d2 = d * d
    d2_mean = float(d2.mean())
    b = -0.5 * (d2 - train_row_mean - d2_mean + float(train_grand_mean))
    return np.asarray([(X_train[:, k].astype(np.float64) @ b) / float(eigvals[k]) for k in range(2)], dtype=np.float32)


def _pcoa_project_many(d_to_train, X_train, eigvals, train_row_mean, train_grand_mean):
    """
    Vectorized out-of-sample projection for many points.
    - d_to_train: (N, M) distances to the M training (landmark) points.
    - X_train: (M, 2) landmark coordinates from _pcoa_2d_from_distance.
    """
    D = np.asarray(d_to_train, dtype=np.float64)
    D2 = D * D
    d2_mean = D2.mean(axis=1, keepdims=True)  # (N,1)
    B = -0.5 * (D2 - np.asarray(train_row_mean, dtype=np.float64)[None, :] - d2_mean + float(train_grand_mean))
    X = (B @ np.asarray(X_train, dtype=np.float64)) / np.asarray(eigvals, dtype=np.float64)[None, :]
    return X.astype(np.float32)



def _compute_pc1pc2_arrow_field(pca_real_xy, real_keys, pca_end_xy, subj_arr, t1_arr):
    real_key_to_xy = {real_keys[i]: pca_real_xy[i] for i in range(len(real_keys))}
    by_parent = {}
    for i in range(len(pca_end_xy)):
        by_parent.setdefault((str(subj_arr[i]), str(t1_arr[i])), []).append(pca_end_xy[i])
    arrow_px, arrow_py, arrow_dx, arrow_dy = [], [], [], []
    for (subj, tt), pts in by_parent.items():
        k = (subj, tt)
        if k not in real_key_to_xy:
            continue
        parent_xy = real_key_to_xy[k]
        mean_xy = np.mean(np.stack(pts, axis=0), axis=0)
        arrow_px.append(parent_xy[0])
        arrow_py.append(parent_xy[1])
        arrow_dx.append(mean_xy[0] - parent_xy[0])
        arrow_dy.append(mean_xy[1] - parent_xy[1])
    return (
        np.asarray(arrow_px, dtype=float),
        np.asarray(arrow_py, dtype=float),
        np.asarray(arrow_dx, dtype=float),
        np.asarray(arrow_dy, dtype=float),
    )


def _compute_jaccard_arrow_field(jacc_real_xy, real_keys, jacc_end_xy, subj_arr, t1_arr):
    jacc_real_key_to_xy = {real_keys[i]: jacc_real_xy[i] for i in range(len(real_keys))}
    jacc_arrow_px, jacc_arrow_py, jacc_arrow_dx, jacc_arrow_dy = [], [], [], []
    for i in range(len(jacc_end_xy)):
        k = (str(subj_arr[i]), str(t1_arr[i]))
        if k not in jacc_real_key_to_xy:
            continue
        parent_xy = jacc_real_key_to_xy[k]
        end_xy = jacc_end_xy[i]
        jacc_arrow_px.append(parent_xy[0])
        jacc_arrow_py.append(parent_xy[1])
        jacc_arrow_dx.append(float(end_xy[0]) - float(parent_xy[0]))
        jacc_arrow_dy.append(float(end_xy[1]) - float(parent_xy[1]))
    return (
        np.asarray(jacc_arrow_px, dtype=float),
        np.asarray(jacc_arrow_py, dtype=float),
        np.asarray(jacc_arrow_dx, dtype=float),
        np.asarray(jacc_arrow_dy, dtype=float),
    )


def load_rollout_rows(subject, t_start):
    rows = []
    with open(ROLL_TSV, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row.get("subject", "").strip() != subject:
                continue
            if row.get("t_start", "").strip() != t_start:
                continue
            rows.append(row)
    rows_sorted = sorted(rows, key=lambda rr: int(rr.get("step", "0")))
    return rows_sorted


def _best_row_for_endpoint(rows_sorted, *, window=10):
    """
    Select the endpoint row for a truncated "optimum search":
    stop after `window` consecutive steps without a new best anchor_mean_logit,
    and return the best-scoring row seen up to that stopping point.
    """
    if not rows_sorted:
        return None
    if "anchor_mean_logit" not in rows_sorted[0]:
        return rows_sorted[-1]

    best = float("-inf")
    best_row = rows_sorted[0]
    since_best = 0
    for row in rows_sorted:
        try:
            v = float(row.get("anchor_mean_logit", "nan"))
        except Exception:
            v = float("nan")
        if np.isfinite(v) and v > best:
            best = float(v)
            best_row = row
            since_best = 0
        else:
            since_best += 1
            if since_best >= int(window):
                break
    return best_row


def _truncate_rows_nobest(rows_sorted, *, window=10):
    """
    Return a truncated prefix of `rows_sorted` ending when `window` consecutive
    steps fail to set a new best anchor_mean_logit.
    """
    if not rows_sorted:
        return rows_sorted
    if "anchor_mean_logit" not in rows_sorted[0]:
        return rows_sorted

    best = float("-inf")
    since_best = 0
    out = []
    for row in rows_sorted:
        out.append(row)
        try:
            v = float(row.get("anchor_mean_logit", "nan"))
        except Exception:
            v = float("nan")
        if np.isfinite(v) and v > best:
            best = float(v)
            since_best = 0
        else:
            since_best += 1
            if since_best >= int(window):
                break
    return out


def _cutoff_after_nobest(steps, series, *, window=10):
    """
    Return an inclusive cutoff index for plotting such that we stop after `window`
    consecutive steps without a new best value in `series`.
    """
    if series is None:
        return None
    y = np.asarray(series, dtype=float)
    if y.size == 0 or y.size != len(steps):
        return None

    best = float("-inf")
    since_best = 0
    for i in range(len(y)):
        v = float(y[i])
        if np.isfinite(v) and v > best:
            best = v
            since_best = 0
        else:
            since_best += 1
            if since_best >= int(window):
                return i
    return None


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    jacc_real_xy = None
    jacc_end_xy = None
    jacc_real_traj_xy = None
    jacc_rollout_xy = None
    jacc_arrow_px = None
    jacc_arrow_py = None
    jacc_arrow_dx = None
    jacc_arrow_dy = None

    use_cache = os.path.exists(args.cache)
    cache = None
    if use_cache:
        cache = np.load(args.cache, allow_pickle=True)
        traj_mode = str(cache["traj_mode"].item()) if "traj_mode" in cache.files else ""
        if traj_mode != "from_start":
            use_cache = False

    if use_cache:
        subject = str(cache["subject"].item())
        t_start = str(cache["t_start"].item())
        steps = cache["steps"].astype(int)
        n_current = cache["n_current"].astype(int)
        anchor_mean_logit = cache["anchor_mean_logit"].astype(float) if "anchor_mean_logit" in cache.files else None
        real_traj_xy = cache["real_traj_xy"].astype(float)
        rollout_xy = cache["rollout_xy"].astype(float)
        rollout_steps = cache["rollout_steps"].astype(int)
        pca_real_xy = cache["pca_real_xy"].astype(float)
        pca_end_xy = cache["pca_end_xy"].astype(float)
        pca_real_times = cache["pca_real_times"].astype(str)
        pca_t1_labels = cache["pca_t1_labels"].astype(str)
        arrow_px = arrow_py = arrow_dx = arrow_dy = None

        if "jacc_real_xy" in cache.files:
            jacc_real_xy = cache["jacc_real_xy"].astype(float)
            jacc_end_xy = cache["jacc_end_xy"].astype(float)
            jacc_real_traj_xy = cache["jacc_real_traj_xy"].astype(float)
            jacc_rollout_xy = cache["jacc_rollout_xy"].astype(float)
            jacc_metric = str(cache["jacc_metric"].item()) if "jacc_metric" in cache.files else ""
            jacc_basis = str(cache["jacc_basis"].item()) if "jacc_basis" in cache.files else ""
            jacc_arrow_px = jacc_arrow_py = jacc_arrow_dx = jacc_arrow_dy = None
            if jacc_metric != "exact" or jacc_basis != "real+endpoints":
                # Force a recompute for the Jaccard panel to update arrow semantics.
                jacc_real_xy = None
    else:
        endpoints_cache = np.load(ENDPOINTS_NPZ, allow_pickle=True)
        endpoints = endpoints_cache["endpoints"].astype(float)
        t1_labels = endpoints_cache["t1_labels"].astype(str)
        subj_arr = endpoints_cache["subject"].astype(str)
        t1_arr = endpoints_cache["t1"].astype(str)

        real_cache = np.load(REAL_NPZ, allow_pickle=True)
        keys = real_cache["keys"]
        emb = real_cache["emb"].astype(float)
        subj_time_to_emb = {(str(keys[i][0]), str(keys[i][1])): emb[i] for i in range(len(keys))}

        subjects = sorted({s for (s, _t) in subj_time_to_emb.keys()})
        subject = args.subject.strip() or subjects[0]

        times_for_subject_all = sorted({t for (s, t) in subj_time_to_emb.keys() if s == subject}, key=time_key)
        if not times_for_subject_all:
            raise SystemExit(f"No timepoints found for subject={subject}.")
        t_start = args.t_start.strip() or times_for_subject_all[0]
        if t_start not in set(times_for_subject_all):
            t_start = times_for_subject_all[0]
        # Only show real trajectory from the chosen start onwards (avoid "two directions" from an interior point).
        try:
            start_idx = times_for_subject_all.index(t_start)
        except ValueError:
            start_idx = 0
        times_for_subject = times_for_subject_all[start_idx:]

        real_times = np.asarray([t for (_s, t) in subj_time_to_emb.keys()], dtype=str)
        real_embs = np.stack([v for v in subj_time_to_emb.values()], axis=0).astype(float)

        # Fit PCA including endpoints so the 2D basis "sees" them.
        mu, comps = pca2_fit(np.vstack([real_embs, endpoints]).astype(float))
        pca_real_xy = pca2_transform(real_embs, mu, comps)
        pca_end_xy = pca2_transform(endpoints, mu, comps)

        real_keys = list(subj_time_to_emb.keys())
        real_key_to_xy = {real_keys[i]: pca_real_xy[i] for i in range(len(real_keys))}
        real_traj_xy = np.stack([real_key_to_xy[(subject, t)] for t in times_for_subject], axis=0).astype(float)

        arrow_px, arrow_py, arrow_dx, arrow_dy = _compute_pc1pc2_arrow_field(
            pca_real_xy, real_keys, pca_end_xy, subj_arr, t1_arr
        )

        rows = load_rollout_rows(subject, t_start)
        if not rows:
            raise SystemExit(f"No rollout rows found for (subject,t_start)=({subject},{t_start}).")

        rows = _truncate_rows_nobest(rows, window=10)
        steps = np.asarray([int(r["step"]) for r in rows], dtype=int)
        n_current = np.asarray([int(r["n_current"]) for r in rows], dtype=int)
        anchor_mean_logit = (
            np.asarray([float(r["anchor_mean_logit"]) for r in rows], dtype=float) if "anchor_mean_logit" in rows[0] else None
        )

        n_steps = len(rows)
        n_keep = int(np.ceil(args.subsample_frac * n_steps))
        n_keep = max(100, n_keep)
        n_keep = max(2, min(n_steps, n_keep))
        idxs = np.unique(np.round(np.linspace(0, n_steps - 1, n_keep)).astype(int))
        idxs = np.unique(np.concatenate([idxs, np.asarray([0, n_steps - 1], dtype=int)]))
        rows_sub = [rows[i] for i in idxs.tolist()]
        rollout_steps = np.asarray([int(r["step"]) for r in rows_sub], dtype=int)

        # Build all_otus mapping consistent with rollout generation.
        _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=GINGIVA_CSV)
        micro_to_otus = collect_micro_to_otus(sra_to_micro)
        all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
        otu_to_idx = {otu: i for i, otu in enumerate(all_otus)}

        model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
        rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
        resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

        embs_sub = []
        with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
            emb_group = emb_file["embeddings"]
            for r in rows_sub:
                idx_list = parse_index_list(r.get("current_otu_indices", ""))
                e = compute_embedding_from_indices(idx_list, all_otus, model, device, emb_group, resolver)
                if e is None:
                    raise SystemExit("Failed to compute embedding for a rollout step.")
                embs_sub.append(e.astype(float))

        rollout_xy = pca2_transform(np.stack(embs_sub, axis=0), mu, comps)

        # Include the start point (real t_start) as step 0 in the plotted rollout trajectory.
        start_xy = real_key_to_xy[(subject, t_start)]
        # Avoid duplicating step 0 (rows_sub often already includes step==0).
        if rollout_steps.size == 0 or int(rollout_steps[0]) != 0:
            rollout_xy = np.vstack([start_xy[None, :], rollout_xy])
            rollout_steps = np.concatenate([np.asarray([0], dtype=int), rollout_steps], axis=0)

        # Jaccard PCoA space: fit on real points; project endpoints and trajectories.
        subject_time_to_otus = defaultdict(set)
        with open(GINGIVA_CSV) as f:
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

        real_idx_lists = []
        for (subj, tt) in real_keys:
            otus = subject_time_to_otus.get((subj, tt), set())
            real_idx_lists.append(np.asarray([otu_to_idx[o] for o in otus if o in otu_to_idx], dtype=np.int32))

        last_row_by_key = {}
        with open(ROLL_TSV, "r", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                subj = row.get("subject", "").strip()
                tt = row.get("t_start", "").strip()
                if not subj or not tt:
                    continue
                try:
                    step = int(row.get("step", "0"))
                except Exception:
                    continue
                k = (subj, tt)
                last_row_by_key.setdefault(k, []).append(row)

        end_idx_lists = []
        for i in range(len(endpoints)):
            k = (str(subj_arr[i]), str(t1_arr[i]))
            rows_k = last_row_by_key.get(k, [])
            rows_k_sorted = sorted(
                rows_k,
                key=lambda rr: int(rr.get("step", "0")) if str(rr.get("step", "")).isdigit() else -1,
            )
            endpoint_row = _best_row_for_endpoint(rows_k_sorted, window=10) or {}
            idx_list = parse_index_list(endpoint_row.get("current_otu_indices", ""))
            end_idx_lists.append(np.asarray([j for j in idx_list if 0 <= j < len(all_otus)], dtype=np.int32))

        # Real trajectory index lists
        real_traj_idx_lists = []
        for t in times_for_subject:
            otus = subject_time_to_otus.get((subject, t), set())
            real_traj_idx_lists.append(np.asarray([otu_to_idx[o] for o in otus if o in otu_to_idx], dtype=np.int32))

        # Rollout trajectory index lists aligned to plotted steps (start + subsampled)
        rollout_idx_lists = []
        # times_for_subject is sliced to start at t_start, so the start is index 0.
        rollout_idx_lists.append(real_traj_idx_lists[0])
        row_by_step = {int(r["step"]): r for r in rows}
        for st in rollout_steps[1:]:
            rr = row_by_step.get(int(st))
            idx_list = parse_index_list(rr.get("current_otu_indices", "")) if rr else []
            rollout_idx_lists.append(np.asarray([j for j in idx_list if 0 <= j < len(all_otus)], dtype=np.int32))

        # Jaccard PCoA basis uses landmarks sampled from BOTH real points and rollout endpoints.
        all_idx = list(real_idx_lists) + list(end_idx_lists)
        n_real = len(real_idx_lists)
        n_all = len(all_idx)
        L = min(int(JACCARD_LANDMARKS), max(2, n_all))
        rng = np.random.default_rng(0)
        landmark_idx = rng.choice(n_all, size=L, replace=False) if n_all > L else np.arange(n_all, dtype=int)
        idx_L = [all_idx[int(i)] for i in landmark_idx.tolist()]
        D_LL = _jaccard_distance_matrix(idx_L, idx_L)
        X_L, eigvals, row_mean, grand_mean = _pcoa_2d_from_distance(D_LL)

        D_all_L = _jaccard_distance_matrix(all_idx, idx_L)
        XY_all = _pcoa_project_many(D_all_L, X_L, eigvals, row_mean, grand_mean)
        jacc_real_xy = XY_all[:n_real].astype(np.float32)
        jacc_end_xy = XY_all[n_real:].astype(np.float32)

        D_rtL = _jaccard_distance_matrix(real_traj_idx_lists, idx_L)
        jacc_real_traj_xy = _pcoa_project_many(D_rtL, X_L, eigvals, row_mean, grand_mean).astype(np.float32)

        D_rollL = _jaccard_distance_matrix(rollout_idx_lists, idx_L)
        jacc_rollout_xy = _pcoa_project_many(D_rollL, X_L, eigvals, row_mean, grand_mean).astype(np.float32)

        jacc_arrow_px, jacc_arrow_py, jacc_arrow_dx, jacc_arrow_dy = _compute_jaccard_arrow_field(
            jacc_real_xy, real_keys, jacc_end_xy, subj_arr, t1_arr
        )

        np.savez(
            args.cache,
            subject=np.asarray(subject, dtype=object),
            t_start=np.asarray(t_start, dtype=object),
            steps=steps,
            n_current=n_current,
            anchor_mean_logit=anchor_mean_logit if anchor_mean_logit is not None else np.asarray([], dtype=float),
            traj_mode=np.asarray("from_start", dtype=object),
            real_traj_xy=real_traj_xy,
            rollout_xy=rollout_xy,
            rollout_steps=rollout_steps,
            pca_real_xy=pca_real_xy,
            pca_end_xy=pca_end_xy,
            pca_real_times=real_times,
            pca_t1_labels=t1_labels,
            arrow_px=arrow_px,
            arrow_py=arrow_py,
            arrow_dx=arrow_dx,
            arrow_dy=arrow_dy,
            jacc_real_xy=jacc_real_xy if jacc_real_xy is not None else np.asarray([], dtype=float),
            jacc_end_xy=jacc_end_xy if jacc_end_xy is not None else np.asarray([], dtype=float),
            jacc_real_traj_xy=jacc_real_traj_xy if jacc_real_traj_xy is not None else np.asarray([], dtype=float),
            jacc_rollout_xy=jacc_rollout_xy if jacc_rollout_xy is not None else np.asarray([], dtype=float),
            jacc_metric=np.asarray("exact", dtype=object),
            jacc_basis=np.asarray("real+endpoints", dtype=object),
        )

        pca_real_times = real_times
        pca_t1_labels = t1_labels

    if jacc_real_xy is None:
        # Build Jaccard PCoA panel data even if the cache was created before this subplot existed.
        endpoints_cache = np.load(ENDPOINTS_NPZ, allow_pickle=True)
        subj_arr = endpoints_cache["subject"].astype(str)
        t1_arr = endpoints_cache["t1"].astype(str)

        real_cache = np.load(REAL_NPZ, allow_pickle=True)
        keys = real_cache["keys"]
        real_keys = [(str(keys[i][0]), str(keys[i][1])) for i in range(len(keys))]

        _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=GINGIVA_CSV)
        micro_to_otus = collect_micro_to_otus(sra_to_micro)
        all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
        otu_to_idx = {otu: i for i, otu in enumerate(all_otus)}

        subject_time_to_otus = defaultdict(set)
        with open(GINGIVA_CSV) as f:
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

        real_idx_lists = []
        for (subj, tt) in real_keys:
            otus = subject_time_to_otus.get((subj, tt), set())
            real_idx_lists.append(np.asarray([otu_to_idx[o] for o in otus if o in otu_to_idx], dtype=np.int32))

        last_row_by_key = {}
        with open(ROLL_TSV, "r", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                subj = row.get("subject", "").strip()
                tt = row.get("t_start", "").strip()
                if not subj or not tt:
                    continue
                try:
                    step = int(row.get("step", "0"))
                except Exception:
                    continue
                k = (subj, tt)
                last_row_by_key.setdefault(k, []).append(row)

        end_idx_lists = []
        for i in range(len(subj_arr)):
            k = (str(subj_arr[i]), str(t1_arr[i]))
            rows_k = last_row_by_key.get(k, [])
            rows_k_sorted = sorted(
                rows_k,
                key=lambda rr: int(rr.get("step", "0")) if str(rr.get("step", "")).isdigit() else -1,
            )
            endpoint_row = _best_row_for_endpoint(rows_k_sorted, window=10) or {}
            idx_list = parse_index_list(endpoint_row.get("current_otu_indices", ""))
            end_idx_lists.append(np.asarray([j for j in idx_list if 0 <= j < len(all_otus)], dtype=np.int32))

        times_for_subject = sorted({t for (s, t) in real_keys if s == subject}, key=time_key)
        real_traj_idx_lists = []
        for t in times_for_subject:
            otus = subject_time_to_otus.get((subject, t), set())
            real_traj_idx_lists.append(np.asarray([otu_to_idx[o] for o in otus if o in otu_to_idx], dtype=np.int32))

        rows = load_rollout_rows(subject, t_start)
        row_by_step = {int(r["step"]): r for r in rows}
        rollout_idx_lists = []
        start_otus = subject_time_to_otus.get((subject, t_start), set())
        rollout_idx_lists.append(np.asarray([otu_to_idx[o] for o in start_otus if o in otu_to_idx], dtype=np.int32))
        for st in rollout_steps[1:]:
            rr = row_by_step.get(int(st))
            idx_list = parse_index_list(rr.get("current_otu_indices", "")) if rr else []
            rollout_idx_lists.append(np.asarray([j for j in idx_list if 0 <= j < len(all_otus)], dtype=np.int32))

        # Jaccard PCoA basis uses landmarks sampled from BOTH real points and rollout endpoints.
        all_idx = list(real_idx_lists) + list(end_idx_lists)
        n_real = len(real_idx_lists)
        n_all = len(all_idx)
        L = min(int(JACCARD_LANDMARKS), max(2, n_all))
        rng = np.random.default_rng(0)
        landmark_idx = rng.choice(n_all, size=L, replace=False) if n_all > L else np.arange(n_all, dtype=int)
        idx_L = [all_idx[int(i)] for i in landmark_idx.tolist()]
        D_LL = _jaccard_distance_matrix(idx_L, idx_L)
        X_L, eigvals, row_mean, grand_mean = _pcoa_2d_from_distance(D_LL)

        D_all_L = _jaccard_distance_matrix(all_idx, idx_L)
        XY_all = _pcoa_project_many(D_all_L, X_L, eigvals, row_mean, grand_mean)
        jacc_real_xy = XY_all[:n_real].astype(np.float32)
        jacc_end_xy = XY_all[n_real:].astype(np.float32)

        D_rtL = _jaccard_distance_matrix(real_traj_idx_lists, idx_L)
        jacc_real_traj_xy = _pcoa_project_many(D_rtL, X_L, eigvals, row_mean, grand_mean).astype(np.float32)

        D_rollL = _jaccard_distance_matrix(rollout_idx_lists, idx_L)
        jacc_rollout_xy = _pcoa_project_many(D_rollL, X_L, eigvals, row_mean, grand_mean).astype(np.float32)

        jacc_arrow_px, jacc_arrow_py, jacc_arrow_dx, jacc_arrow_dy = _compute_jaccard_arrow_field(
            jacc_real_xy, real_keys, jacc_end_xy, subj_arr, t1_arr
        )

        # Update cache in-place so future runs don't recompute Jaccard.
        np.savez(
            args.cache,
            subject=np.asarray(subject, dtype=object),
            t_start=np.asarray(t_start, dtype=object),
            steps=steps,
            n_current=n_current,
            anchor_mean_logit=anchor_mean_logit if anchor_mean_logit is not None else np.asarray([], dtype=float),
            traj_mode=np.asarray("from_start", dtype=object),
            real_traj_xy=real_traj_xy,
            rollout_xy=rollout_xy,
            rollout_steps=rollout_steps,
            pca_real_xy=pca_real_xy,
            pca_end_xy=pca_end_xy,
            pca_real_times=pca_real_times,
            pca_t1_labels=pca_t1_labels,
            arrow_px=arrow_px,
            arrow_py=arrow_py,
            arrow_dx=arrow_dx,
            arrow_dy=arrow_dy,
            jacc_real_xy=jacc_real_xy,
            jacc_end_xy=jacc_end_xy,
            jacc_real_traj_xy=jacc_real_traj_xy,
            jacc_rollout_xy=jacc_rollout_xy,
            jacc_metric=np.asarray("exact", dtype=object),
            jacc_basis=np.asarray("real+endpoints", dtype=object),
        )

    # Compute arrow fields from cached 2D geometry (fast) so visual tweaks don't require cache deletion.
    endpoints_cache = np.load(ENDPOINTS_NPZ, allow_pickle=True)
    subj_arr = endpoints_cache["subject"].astype(str)
    t1_arr = endpoints_cache["t1"].astype(str)

    real_cache = np.load(REAL_NPZ, allow_pickle=True)
    keys = real_cache["keys"]
    real_keys = [(str(keys[i][0]), str(keys[i][1])) for i in range(len(keys))]

    arrow_px, arrow_py, arrow_dx, arrow_dy = _compute_pc1pc2_arrow_field(pca_real_xy, real_keys, pca_end_xy, subj_arr, t1_arr)
    jacc_arrow_px, jacc_arrow_py, jacc_arrow_dx, jacc_arrow_dy = _compute_jaccard_arrow_field(
        jacc_real_xy, real_keys, jacc_end_xy, subj_arr, t1_arr
    )

    # (Cosine panel removed.)

    # Plot
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(12.0, 14.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.65, 1.65])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    cutoff = _cutoff_after_nobest(steps, anchor_mean_logit, window=10) if anchor_mean_logit is not None else None
    if cutoff is not None:
        steps0 = steps[: cutoff + 1]
        n_current0 = n_current[: cutoff + 1]
        anchor_mean0 = anchor_mean_logit[: cutoff + 1] if anchor_mean_logit is not None else None
    else:
        steps0 = steps
        n_current0 = n_current
        anchor_mean0 = anchor_mean_logit

    ax0.plot(steps0, n_current0, label="n_current", linewidth=2.0, color="black")
    ax0.set_xlabel("step")
    ax0.set_ylabel("n_current")
    ax0.set_title(f"Rollout counts ({subject}, start={t_start})")
    ax0.grid(True, alpha=0.25)

    if anchor_mean0 is not None and len(anchor_mean0) == len(steps0):
        ax0b = ax0.twinx()
        ax0b.plot(steps0, anchor_mean0, label="anchor_mean_logit", linewidth=1.6, color="#d62728", alpha=0.9)
        ax0b.set_ylabel("anchor mean logit")
        h0, l0 = ax0.get_legend_handles_labels()
        h1, l1 = ax0b.get_legend_handles_labels()
        ax0.legend(h0 + h1, l0 + l1, frameon=False, ncol=2, loc="lower left")
    else:
        ax0.legend(frameon=False, loc="lower left")

    # Background + mean-arrow field
    timepoints = sorted(set(pca_real_times.tolist()) | set(pca_t1_labels.tolist()), key=time_key)
    colors = cm.viridis(np.linspace(0.0, 1.0, max(2, len(timepoints))))
    color_by_time = {t: colors[i] for i, t in enumerate(timepoints)}

    ax1.scatter(
        pca_end_xy[:, 0],
        pca_end_xy[:, 1],
        s=10,
        marker=".",
        color="#666666",
        alpha=0.35,
        linewidths=0,
        label="Rollout endpoints",
        zorder=0,
    )
    for t in timepoints:
        mask_r = pca_real_times == t
        if np.any(mask_r):
            ax1.scatter(
                pca_real_xy[mask_r, 0],
                pca_real_xy[mask_r, 1],
                s=18,
                marker="o",
                color=color_by_time[t],
                alpha=0.75,
                linewidths=0,
                zorder=1,
            )

    if arrow_px.size:
        ax1.quiver(
            arrow_px,
            arrow_py,
            arrow_dx,
            arrow_dy,
            angles="xy",
            scale_units="xy",
            pivot="tail",
            scale=8.0,
            color="black",
            width=0.0016,
            headwidth=3,
            headlength=4,
            headaxislength=3,
            alpha=0.14,
            zorder=2,
        )

    # Real subject trajectory
    (real_line,) = ax1.plot(
        real_traj_xy[:, 0],
        real_traj_xy[:, 1],
        color="black",
        alpha=0.70,
        linewidth=0.5,
        zorder=4,
    )
    real_line.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])
    ax1.scatter(
        real_traj_xy[:, 0],
        real_traj_xy[:, 1],
        s=42,
        facecolors="none",
        edgecolors="black",
        alpha=0.9,
        linewidths=0.7,
        zorder=5,
        label="Real trajectory",
    )

    # Simulated rollout trajectory
    rollout_start_xy = rollout_xy[0]
    ax1.scatter(
        [rollout_start_xy[0]],
        [rollout_start_xy[1]],
        s=70,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=1.4,
        zorder=7,
        label="Rollout start",
    )
    (rollout_line,) = ax1.plot(
        rollout_xy[:, 0],
        rollout_xy[:, 1],
        color="#d62728",
        alpha=0.55,
        linewidth=0.5,
        zorder=5,
    )
    rollout_line.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])
    ax1.scatter(
        rollout_xy[:, 0],
        rollout_xy[:, 1],
        s=10,
        marker=".",
        color="#d62728",
        alpha=0.75,
        linewidths=0,
        zorder=6,
        label="Rollout trajectory",
    )

    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title("PC1/PC2 vector field + example trajectories")

    class _HandlerArrow(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            p = FancyArrowPatch(
                (xdescent, ydescent + 0.5 * height),
                (xdescent + width, ydescent + 0.5 * height),
                arrowstyle="-|>",
                mutation_scale=fontsize * 1.1,
                linewidth=1.0,
                color="black",
                alpha=0.5,
            )
            p.set_transform(trans)
            return [p]

    arrow_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", label="Direction to endpoint")
    handles, labels = ax1.get_legend_handles_labels()
    handles = [arrow_handle] + handles
    traj_legend = ax1.legend(
        handles=handles,
        frameon=True,
        loc="lower right",
        handler_map={FancyArrowPatch: _HandlerArrow()},
    )
    traj_legend.get_frame().set_edgecolor("0.7")
    traj_legend.get_frame().set_linewidth(0.8)
    traj_legend.get_frame().set_facecolor("white")
    traj_legend.get_frame().set_alpha(0.9)
    ax1.add_artist(traj_legend)

    time_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_by_time[t],
            markeredgecolor=color_by_time[t],
            markeredgewidth=0.0,
            markersize=6,
            label=str(t),
        )
        for t in timepoints
    ]
    time_legend = ax1.legend(
        handles=time_handles,
        title="Timepoint",
        frameon=True,
        loc="upper left",
        markerscale=1.5,
        fontsize=8,
    )
    time_legend.get_frame().set_edgecolor("0.7")
    time_legend.get_frame().set_linewidth(0.8)
    time_legend.get_frame().set_facecolor("white")
    time_legend.get_frame().set_alpha(0.9)

    x_all = np.concatenate([pca_real_xy[:, 0], pca_end_xy[:, 0]])
    y_all = np.concatenate([pca_real_xy[:, 1], pca_end_xy[:, 1]])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    x_pad = 0.05 * (x_max - x_min + 1e-12)
    y_pad = 0.05 * (y_max - y_min + 1e-12)
    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    # OTU-space Jaccard PCoA panel (same styling; different geometry)
    ax2.scatter(
        jacc_end_xy[:, 0],
        jacc_end_xy[:, 1],
        s=10,
        marker=".",
        color="#666666",
        alpha=0.35,
        linewidths=0,
        label="Rollout endpoints",
        zorder=0,
    )
    mean_end_xy = np.mean(np.asarray(jacc_end_xy, dtype=float), axis=0)
    if np.all(np.isfinite(mean_end_xy)):
        ax2.scatter(
            [float(mean_end_xy[0])],
            [float(mean_end_xy[1])],
            s=70,
            marker=".",
            color="#d62728",
            linewidths=0,
            zorder=3,
            label="Mean endpoint",
        )

    for t in timepoints:
        mask_r = pca_real_times == t
        if np.any(mask_r):
            ax2.scatter(
                jacc_real_xy[mask_r, 0],
                jacc_real_xy[mask_r, 1],
                s=18,
                marker="o",
                color=color_by_time[t],
                alpha=0.75,
                linewidths=0,
                zorder=1,
            )

    if jacc_arrow_px.size:
        ax2.quiver(
            jacc_arrow_px,
            jacc_arrow_py,
            jacc_arrow_dx,
            jacc_arrow_dy,
            angles="xy",
            scale_units="xy",
            pivot="tail",
            scale=1.4,
            color="black",
            width=0.0016,
            headwidth=3,
            headlength=4,
            headaxislength=3,
            alpha=0.14,
            zorder=2,
        )

    ax2.set_xlabel("PCoA1 (Jaccard)")
    ax2.set_ylabel("PCoA2 (Jaccard)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("OTU Jaccard PCoA + example trajectories")

    # Legend for the Jaccard panel.
    arrow_handle2 = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", label="Direction to endpoint")

    handles2, labels2 = ax2.get_legend_handles_labels()
    drop_labels = {"Real trajectory", "Rollout start", "Rollout trajectory"}
    kept = [(h, l) for (h, l) in zip(handles2, labels2) if l not in drop_labels]
    handles2 = [arrow_handle2] + [h for (h, _l) in kept]
    labels2 = ["Direction to endpoint"] + [l for (_h, l) in kept]
    leg2 = ax2.legend(
        handles=handles2,
        labels=labels2,
        frameon=True,
        loc="lower right",
        handler_map={FancyArrowPatch: _HandlerArrow()},
    )
    leg2.get_frame().set_edgecolor("0.7")
    leg2.get_frame().set_linewidth(0.8)
    leg2.get_frame().set_facecolor("white")
    leg2.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")
    print(f"Saved cache: {args.cache}")


if __name__ == "__main__":
    main()
