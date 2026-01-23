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
from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402


ROLL_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob.tsv")
ENDPOINTS_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoints_cache.npz")
REAL_NPZ = os.path.join("data", "diabimmune", "diabimmune_real_embeddings_cache.npz")
SAMPLES_CSV = os.path.join("data", "diabimmune", "samples.csv")

OUT_PNG = os.path.join("data", "diabimmune", "diabimmune_rollout_trajectory_overlay.png")
CACHE_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_trajectory_overlay_cache.npz")

JACCARD_K = 64
JACCARD_LANDMARKS = 400


def parse_args():
    p = argparse.ArgumentParser(description="Diabimmune: counts + PC1/PC2 vector field + example trajectories.")
    p.add_argument("--out", default=OUT_PNG)
    p.add_argument("--cache", default=CACHE_NPZ)
    p.add_argument("--subject", default="", help="Subject id (default: first available with >=2 timepoints).")
    p.add_argument("--t-start", default="", help="Start age key (days, binned) for rollout (default: first for subject).")
    p.add_argument("--subsample-frac", type=float, default=0.05, help="Fraction of rollout steps to embed/plot.")
    p.add_argument("--samples-csv", default=SAMPLES_CSV)
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
    try:
        return float(str(t).strip())
    except Exception:
        return 1e18


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


def _minhash_jaccard_distance_matrix(sig_a, sig_b, *, block=128):
    sig_a = np.asarray(sig_a)
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
    return sorted(rows, key=lambda rr: int(rr.get("step", "0")))


def _diabimmune_age_bin_label(age_days):
    edges = [27.0, 216.2, 405.3, 594.5, 783.7, 972.8]
    if age_days < edges[0]:
        return f"<{edges[0]:g}"
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= age_days < hi:
            return f"{lo:g}-{hi:g}"
    return f"{edges[-1]:g}+"


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

    if os.path.exists(args.cache):
        cache = np.load(args.cache, allow_pickle=True)
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
        arrow_px = cache["arrow_px"].astype(float)
        arrow_py = cache["arrow_py"].astype(float)
        arrow_dx = cache["arrow_dx"].astype(float)
        arrow_dy = cache["arrow_dy"].astype(float)

        if "jacc_real_xy" in cache.files:
            jacc_real_xy = cache["jacc_real_xy"].astype(float)
            jacc_end_xy = cache["jacc_end_xy"].astype(float)
            jacc_real_traj_xy = cache["jacc_real_traj_xy"].astype(float)
            jacc_rollout_xy = cache["jacc_rollout_xy"].astype(float)
            jacc_arrow_px = cache["jacc_arrow_px"].astype(float)
            jacc_arrow_py = cache["jacc_arrow_py"].astype(float)
            jacc_arrow_dx = cache["jacc_arrow_dx"].astype(float)
            jacc_arrow_dy = cache["jacc_arrow_dy"].astype(float)
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
        if not subjects:
            raise SystemExit("No diabimmune subjects found in real cache.")

        def subj_has_multi(s):
            return len({t for (ss, t) in subj_time_to_emb.keys() if ss == s}) >= 2

        subject = args.subject.strip() or next((s for s in subjects if subj_has_multi(s)), subjects[0])
        times_for_subject = sorted({t for (s, t) in subj_time_to_emb.keys() if s == subject}, key=time_key)
        t_start = args.t_start.strip() or times_for_subject[0]

        real_times = np.asarray([t for (_s, t) in subj_time_to_emb.keys()], dtype=str)
        real_embs = np.stack([v for v in subj_time_to_emb.values()], axis=0).astype(float)

        mu, comps = pca2_fit(real_embs)
        pca_real_xy = pca2_transform(real_embs, mu, comps)
        pca_end_xy = pca2_transform(endpoints, mu, comps)

        real_keys = list(subj_time_to_emb.keys())
        real_key_to_xy = {real_keys[i]: pca_real_xy[i] for i in range(len(real_keys))}
        real_traj_xy = np.stack([real_key_to_xy[(subject, t)] for t in times_for_subject], axis=0).astype(float)

        by_parent = {}
        for i in range(len(endpoints)):
            by_parent.setdefault((subj_arr[i], t1_arr[i]), []).append(pca_end_xy[i])
        arrow_px, arrow_py, arrow_dx, arrow_dy = [], [], [], []
        for (subj, tt), pts in by_parent.items():
            if (subj, tt) not in real_key_to_xy:
                continue
            parent_xy = real_key_to_xy[(subj, tt)]
            mean_xy = np.mean(np.stack(pts, axis=0), axis=0)
            arrow_px.append(parent_xy[0])
            arrow_py.append(parent_xy[1])
            arrow_dx.append(mean_xy[0] - parent_xy[0])
            arrow_dy.append(mean_xy[1] - parent_xy[1])
        arrow_px = np.asarray(arrow_px, dtype=float)
        arrow_py = np.asarray(arrow_py, dtype=float)
        arrow_dx = np.asarray(arrow_dx, dtype=float)
        arrow_dy = np.asarray(arrow_dy, dtype=float)

        rows = load_rollout_rows(subject, t_start)
        if not rows:
            raise SystemExit(f"No rollout rows found for (subject,t_start)=({subject},{t_start}).")

        steps = np.asarray([int(r["step"]) for r in rows], dtype=int)
        n_current = np.asarray([int(r["n_current"]) for r in rows], dtype=int)
        anchor_mean_logit = (
            np.asarray([float(r["anchor_mean_logit"]) for r in rows], dtype=float) if "anchor_mean_logit" in rows[0] else None
        )

        n_steps = len(rows)
        n_keep = max(2, int(np.ceil(args.subsample_frac * n_steps)))
        idxs = np.unique(np.round(np.linspace(0, n_steps - 1, n_keep)).astype(int))
        idxs = np.unique(np.concatenate([idxs, np.asarray([0, n_steps - 1], dtype=int)]))
        rows_sub = [rows[i] for i in idxs.tolist()]
        rollout_steps = np.asarray([int(r["step"]) for r in rows_sub], dtype=int)

        _run_rows, sra_to_micro, _gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
        micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)
        all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
        otu_to_idx = {otu: i for i, otu in enumerate(all_otus)}

        model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
        rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
        resolver = (
            shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}
        )

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

        start_xy = real_key_to_xy[(subject, t_start)]
        rollout_xy = np.vstack([start_xy[None, :], rollout_xy])
        rollout_steps = np.concatenate([np.asarray([0], dtype=int), rollout_steps], axis=0)

        np.savez(
            args.cache,
            subject=np.asarray(subject, dtype=object),
            t_start=np.asarray(t_start, dtype=object),
            steps=steps,
            n_current=n_current,
            anchor_mean_logit=anchor_mean_logit if anchor_mean_logit is not None else np.asarray([], dtype=float),
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
        )

        pca_real_times = real_times
        pca_t1_labels = t1_labels

    if jacc_real_xy is None:
        endpoints_cache = np.load(ENDPOINTS_NPZ, allow_pickle=True)
        subj_arr = endpoints_cache["subject"].astype(str)
        t1_arr = endpoints_cache["t1"].astype(str)

        real_cache = np.load(REAL_NPZ, allow_pickle=True)
        keys = real_cache["keys"]
        real_keys = [(str(keys[i][0]), str(keys[i][1])) for i in range(len(keys))]

        _run_rows, sra_to_micro, _gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
        micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)
        all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
        otu_to_idx = {otu: i for i, otu in enumerate(all_otus)}

        samples_table = {}
        with open(args.samples_csv) as f:
            header = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if header is None:
                    header = parts
                    header[0] = header[0].lstrip("\ufeff")
                    continue
                row = dict(zip(header, parts))
                sid = row.get("sampleID", "").strip()
                if sid:
                    samples_table[sid] = row

        def safe_float(x):
            try:
                return float(x)
            except Exception:
                return None

        def age_bin_days(age_value):
            return str(int(round(float(age_value))))

        subject_age_to_otus = defaultdict(set)
        subject_age_to_age = {}
        for srs, info in micro_to_sample.items():
            subj = str(info.get("subject", "")).strip()
            sample_id = str(info.get("sample", "")).strip()
            if not subj or not sample_id:
                continue
            age = safe_float(samples_table.get(sample_id, {}).get("age_at_collection", ""))
            if age is None:
                continue
            age_key = age_bin_days(age)
            otus = micro_to_otus.get(srs, [])
            if otus:
                subject_age_to_otus[(subj, age_key)].update(otus)
                subject_age_to_age[(subj, age_key)] = float(age)

        real_idx_all = []
        real_labels = []
        for (subj, tt) in real_keys:
            otus = subject_age_to_otus.get((subj, tt), set())
            idx = np.asarray([otu_to_idx[o] for o in otus if o in otu_to_idx], dtype=np.int32)
            real_idx_all.append(idx)
            age_val = subject_age_to_age.get((subj, tt))
            real_labels.append(_diabimmune_age_bin_label(float(age_val)) if age_val is not None else "unknown")

        sig_all = _minhash_signatures(real_idx_all, k=JACCARD_K, seed=0)
        n = sig_all.shape[0]
        L = min(JACCARD_LANDMARKS, n)
        rng = np.random.default_rng(0)
        landmark_idx = rng.choice(n, size=L, replace=False) if n > L else np.arange(n, dtype=int)
        sig_L = sig_all[landmark_idx]
        D_LL = _minhash_jaccard_distance_matrix(sig_L, sig_L, block=128)
        X_L, eigvals, row_mean, grand_mean = _pcoa_2d_from_distance(D_LL)

        D_RL = _minhash_jaccard_distance_matrix(sig_all, sig_L, block=128)
        jacc_real_xy = np.stack(
            [_pcoa_project_out_of_sample(D_RL[i], X_L, eigvals, row_mean, grand_mean) for i in range(n)],
            axis=0,
        ).astype(np.float32)

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
                prev = last_row_by_key.get(k)
                if prev is None or step >= int(prev.get("step", "0")):
                    last_row_by_key[k] = row

        end_idx_lists = []
        for i in range(len(subj_arr)):
            k = (subj_arr[i], t1_arr[i])
            rr = last_row_by_key.get(k, {})
            idx_list = parse_index_list(rr.get("current_otu_indices", ""))
            end_idx_lists.append(np.asarray([j for j in idx_list if 0 <= j < len(all_otus)], dtype=np.int32))

        sig_end = _minhash_signatures(end_idx_lists, k=JACCARD_K, seed=1)
        D_endL = _minhash_jaccard_distance_matrix(sig_end, sig_L, block=128)
        jacc_end_xy = np.stack(
            [_pcoa_project_out_of_sample(D_endL[i], X_L, eigvals, row_mean, grand_mean) for i in range(D_endL.shape[0])],
            axis=0,
        ).astype(np.float32)

        times_for_subject = sorted({t for (s, t) in real_keys if s == subject}, key=time_key)
        real_traj_idx = []
        for t in times_for_subject:
            otus = subject_age_to_otus.get((subject, t), set())
            real_traj_idx.append(np.asarray([otu_to_idx[o] for o in otus if o in otu_to_idx], dtype=np.int32))
        sig_rt = _minhash_signatures(real_traj_idx, k=JACCARD_K, seed=2)
        D_rtL = _minhash_jaccard_distance_matrix(sig_rt, sig_L, block=128)
        jacc_real_traj_xy = np.stack(
            [_pcoa_project_out_of_sample(D_rtL[i], X_L, eigvals, row_mean, grand_mean) for i in range(D_rtL.shape[0])],
            axis=0,
        ).astype(np.float32)

        rows = load_rollout_rows(subject, t_start)
        row_by_step = {int(r["step"]): r for r in rows}
        rollout_idx = []
        start_otus = subject_age_to_otus.get((subject, t_start), set())
        rollout_idx.append(np.asarray([otu_to_idx[o] for o in start_otus if o in otu_to_idx], dtype=np.int32))
        for st in rollout_steps[1:]:
            rr = row_by_step.get(int(st))
            idx_list = parse_index_list(rr.get("current_otu_indices", "")) if rr else []
            rollout_idx.append(np.asarray([j for j in idx_list if 0 <= j < len(all_otus)], dtype=np.int32))

        sig_roll = _minhash_signatures(rollout_idx, k=JACCARD_K, seed=3)
        D_rollL = _minhash_jaccard_distance_matrix(sig_roll, sig_L, block=128)
        jacc_rollout_xy = np.stack(
            [_pcoa_project_out_of_sample(D_rollL[i], X_L, eigvals, row_mean, grand_mean) for i in range(D_rollL.shape[0])],
            axis=0,
        ).astype(np.float32)

        key_to_xy = {real_keys[i]: jacc_real_xy[i] for i in range(len(real_keys))}
        end_by_parent = {}
        for i in range(len(subj_arr)):
            end_by_parent.setdefault((subj_arr[i], t1_arr[i]), []).append(jacc_end_xy[i])
        jacc_arrow_px, jacc_arrow_py, jacc_arrow_dx, jacc_arrow_dy = [], [], [], []
        for (subj, tt), pts in end_by_parent.items():
            if (subj, tt) not in key_to_xy:
                continue
            parent_xy = key_to_xy[(subj, tt)]
            mean_xy = np.mean(np.stack(pts, axis=0), axis=0)
            jacc_arrow_px.append(parent_xy[0])
            jacc_arrow_py.append(parent_xy[1])
            jacc_arrow_dx.append(mean_xy[0] - parent_xy[0])
            jacc_arrow_dy.append(mean_xy[1] - parent_xy[1])
        jacc_arrow_px = np.asarray(jacc_arrow_px, dtype=float)
        jacc_arrow_py = np.asarray(jacc_arrow_py, dtype=float)
        jacc_arrow_dx = np.asarray(jacc_arrow_dx, dtype=float)
        jacc_arrow_dy = np.asarray(jacc_arrow_dy, dtype=float)

        # Update cache so future runs don't recompute Jaccard.
        np.savez(
            args.cache,
            subject=np.asarray(subject, dtype=object),
            t_start=np.asarray(t_start, dtype=object),
            steps=steps,
            n_current=n_current,
            anchor_mean_logit=anchor_mean_logit if anchor_mean_logit is not None else np.asarray([], dtype=float),
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
            jacc_arrow_px=jacc_arrow_px,
            jacc_arrow_py=jacc_arrow_py,
            jacc_arrow_dx=jacc_arrow_dx,
            jacc_arrow_dy=jacc_arrow_dy,
        )

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(12.0, 14.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.65, 1.65])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    ax0.plot(steps, n_current, label="n_current", linewidth=2.0, color="black")
    ax0.set_xlabel("step")
    ax0.set_ylabel("n_current")
    ax0.set_title(f"Rollout counts ({subject}, start={t_start})")
    ax0.grid(True, alpha=0.25)

    if anchor_mean_logit is not None and len(anchor_mean_logit) == len(steps):
        ax0b = ax0.twinx()
        ax0b.plot(steps, anchor_mean_logit, label="anchor_mean_logit", linewidth=1.6, color="#d62728", alpha=0.9)
        ax0b.set_ylabel("anchor mean logit")
        h0, l0 = ax0.get_legend_handles_labels()
        h1, l1 = ax0b.get_legend_handles_labels()
        ax0.legend(h0 + h1, l0 + l1, frameon=False, ncol=2)
    else:
        ax0.legend(frameon=False)

    # Color map by age bin (use same scheme as other diabimmune plots: plasma over sorted bins)
    bins = sorted(set(pca_real_times.tolist()) | set(pca_t1_labels.tolist()), key=time_key)
    cmap3 = plt.cm.plasma
    color_by_bin = {b: cmap3(i / float(len(bins) - 1)) if len(bins) > 1 else cmap3(0.5) for i, b in enumerate(bins)}

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
    for b in bins:
        mask_r = pca_real_times == b
        if np.any(mask_r):
            ax1.scatter(
                pca_real_xy[mask_r, 0],
                pca_real_xy[mask_r, 1],
                s=18,
                marker="o",
                color=color_by_bin[b],
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

    (real_line,) = ax1.plot(real_traj_xy[:, 0], real_traj_xy[:, 1], color="black", alpha=0.70, linewidth=0.5, zorder=4)
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
    (rollout_line,) = ax1.plot(rollout_xy[:, 0], rollout_xy[:, 1], color="#d62728", alpha=0.55, linewidth=0.5, zorder=5)
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
    handles, _labels = ax1.get_legend_handles_labels()
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

    bin_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_by_bin[b],
            markeredgecolor=color_by_bin[b],
            markeredgewidth=0.0,
            markersize=6,
            label=str(b),
        )
        for b in bins
    ]
    bin_legend = ax1.legend(handles=bin_handles, title="Age bin (days)", frameon=True, loc="upper left", markerscale=1.5, fontsize=8)
    bin_legend.get_frame().set_edgecolor("0.7")
    bin_legend.get_frame().set_linewidth(0.8)
    bin_legend.get_frame().set_facecolor("white")
    bin_legend.get_frame().set_alpha(0.9)

    x_all = np.concatenate([pca_real_xy[:, 0], pca_end_xy[:, 0]])
    y_all = np.concatenate([pca_real_xy[:, 1], pca_end_xy[:, 1]])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    x_pad = 0.05 * (x_max - x_min + 1e-12)
    y_pad = 0.05 * (y_max - y_min + 1e-12)
    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    # Jaccard PCoA panel
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
    for b in bins:
        mask_r = pca_real_times == b
        if np.any(mask_r):
            ax2.scatter(
                jacc_real_xy[mask_r, 0],
                jacc_real_xy[mask_r, 1],
                s=18,
                marker="o",
                color=color_by_bin[b],
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
            scale=8.0,
            color="black",
            width=0.0016,
            headwidth=3,
            headlength=4,
            headaxislength=3,
            alpha=0.14,
            zorder=2,
        )

    (real_line2,) = ax2.plot(
        jacc_real_traj_xy[:, 0],
        jacc_real_traj_xy[:, 1],
        color="black",
        alpha=0.70,
        linewidth=0.5,
        zorder=4,
    )
    real_line2.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])
    ax2.scatter(
        jacc_real_traj_xy[:, 0],
        jacc_real_traj_xy[:, 1],
        s=42,
        facecolors="none",
        edgecolors="black",
        alpha=0.9,
        linewidths=0.7,
        zorder=5,
        label="Real trajectory",
    )

    rollout_start_xy2 = jacc_rollout_xy[0]
    ax2.scatter(
        [rollout_start_xy2[0]],
        [rollout_start_xy2[1]],
        s=70,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=1.4,
        zorder=7,
        label="Rollout start",
    )
    (rollout_line2,) = ax2.plot(
        jacc_rollout_xy[:, 0],
        jacc_rollout_xy[:, 1],
        color="#d62728",
        alpha=0.55,
        linewidth=0.5,
        zorder=5,
    )
    rollout_line2.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])
    ax2.scatter(
        jacc_rollout_xy[:, 0],
        jacc_rollout_xy[:, 1],
        s=10,
        marker=".",
        color="#d62728",
        alpha=0.75,
        linewidths=0,
        zorder=6,
        label="Rollout trajectory",
    )
    ax2.set_xlabel("PCoA1 (Jaccard)")
    ax2.set_ylabel("PCoA2 (Jaccard)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("OTU Jaccard PCoA + example trajectories")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")
    print(f"Saved cache: {args.cache}")


if __name__ == "__main__":
    main()
