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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

JACCARD_LANDMARKS = 600
WGS_RUN_TABLE = os.path.join("data", "diabimmune", "SraRunTable_wgs.csv")
EXTRA_RUN_TABLE = os.path.join("data", "diabimmune", "SraRunTable_extra.csv")

# Plot customization knobs (can be overridden by wrapper scripts).
# - FONT_SIZE: set e.g. 14 to increase font sizes globally.
# - HIDE_XY_TICK_LABELS: hides numeric tick labels on main panels.
FONT_SIZE = None
HIDE_XY_TICK_LABELS = False
AGE_BIN_LEGEND_FONT_SIZE = 8
REAL_TRAJ_LINEWIDTH = 0.5
ROLLOUT_TRAJ_LINEWIDTH = 0.5
TRAJ_STROKE_LINEWIDTH = 2.0
REAL_TRAJ_COLOR = "#0097a7"
ROLLOUT_TRAJ_COLOR = "#00cfe8"
PCA_REAL_POINTS_ALPHA = 0.75
PCA_ARROW_MERGE_BINS = 0
PCA_ARROW_HEADWIDTH = 3
PCA_ARROW_HEADLENGTH = 4
PCA_ARROW_HEADAXISLENGTH = 3

# Jaccard (bottom-left) panel style controls
JACCARD_REAL_USE_AGE_COLORS = True
JACCARD_REAL_POINT_COLOR = "#9a9a9a"
JACCARD_REAL_POINT_ALPHA = 0.75
JACCARD_ENDPOINT_COLOR = "#666666"
JACCARD_ENDPOINT_ALPHA = 0.35
JACCARD_ARROW_COLOR = "black"
JACCARD_ARROW_ALPHA = 0.14
JACCARD_ARROW_WIDTH = 0.0016
JACCARD_ARROW_SCALE = 1.1
JACCARD_ARROW_MERGE_BINS = 0
JACCARD_ARROW_HEADWIDTH = 3
JACCARD_ARROW_HEADLENGTH = 4
JACCARD_ARROW_HEADAXISLENGTH = 3

# Instrument (bottom-right) panel controls
SHOW_INSTRUMENT_PANEL_ENDPOINTS = True


def _merge_arrows_on_grid(px, py, dx, dy, n_bins):
    if n_bins is None or int(n_bins) <= 0:
        return px, py, dx, dy
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    if px.size == 0:
        return px, py, dx, dy

    n_bins = int(n_bins)
    x0, x1 = float(np.min(px)), float(np.max(px))
    y0, y1 = float(np.min(py)), float(np.max(py))
    if not np.isfinite([x0, x1, y0, y1]).all() or x1 <= x0 or y1 <= y0:
        return px, py, dx, dy

    xi = np.floor((px - x0) / (x1 - x0 + 1e-12) * n_bins).astype(int)
    yi = np.floor((py - y0) / (y1 - y0 + 1e-12) * n_bins).astype(int)
    xi = np.clip(xi, 0, n_bins - 1)
    yi = np.clip(yi, 0, n_bins - 1)

    buckets = {}
    for i in range(px.size):
        buckets.setdefault((int(xi[i]), int(yi[i])), []).append(i)

    mpx, mpy, mdx, mdy = [], [], [], []
    for ids in buckets.values():
        idx = np.asarray(ids, dtype=int)
        mpx.append(float(np.mean(px[idx])))
        mpy.append(float(np.mean(py[idx])))
        mdx.append(float(np.mean(dx[idx])))
        mdy.append(float(np.mean(dy[idx])))

    return (
        np.asarray(mpx, dtype=float),
        np.asarray(mpy, dtype=float),
        np.asarray(mdx, dtype=float),
        np.asarray(mdy, dtype=float),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Diabimmune: counts + PC1/PC2 vector field + example trajectories.")
    p.add_argument("--out", default=OUT_PNG)
    p.add_argument("--cache", default=CACHE_NPZ)
    p.add_argument("--subject", default="", help="Subject id (default: first available with >=2 timepoints).")
    p.add_argument("--t-start", default="", help="Start age key (days, binned) for rollout (default: first for subject).")
    p.add_argument("--subsample-frac", type=float, default=0.1, help="Fraction of rollout steps to embed/plot.")
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


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


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


def _jaccard_distance_sorted_unique(a, b):
    """
    Exact Jaccard distance = 1 - |A∩B|/|A∪B| for sorted unique int arrays.
    """
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
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


def _jaccard_distance_matrix(index_lists_a, index_lists_b):
    """
    Exact Jaccard distance matrix between two lists of index arrays.
    Arrays are normalized to sorted unique.
    """
    a_lists = [np.unique(np.asarray(x, dtype=np.int32)) for x in index_lists_a]
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


def pick_example_with_movement(tsv_path, valid_keys=None):
    """
    Pick (subject, t_start) with "decent movement" from the rollout TSV.

    Heuristic: maximize anchor_mean_logit range, then number of changed steps.
    If valid_keys is provided, only consider keys in that set.
    """
    best_key = None
    best_tuple = None
    stats = {}

    with open(tsv_path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            subj = row.get("subject", "").strip()
            t0 = row.get("t_start", "").strip()
            if not subj or not t0:
                continue
            key = (subj, t0)
            if valid_keys is not None and key not in valid_keys:
                continue

            st = stats.get(key)
            if st is None:
                st = {"min": float("inf"), "max": float("-inf"), "changed": 0, "rows": 0}
                stats[key] = st
            st["rows"] += 1

            try:
                v = float(row.get("anchor_mean_logit", "nan"))
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                st["min"] = min(st["min"], v)
                st["max"] = max(st["max"], v)

            try:
                n_added = int(row.get("n_added", "0"))
                n_removed = int(row.get("n_removed", "0"))
            except Exception:
                n_added = 0
                n_removed = 0
            if n_added != 0 or n_removed != 0:
                st["changed"] += 1

    for key, st in stats.items():
        if st["rows"] < 3 or not np.isfinite(st["min"]) or not np.isfinite(st["max"]):
            continue
        rng = float(st["max"] - st["min"])
        score = (rng, int(st["changed"]), int(st["rows"]))
        if best_tuple is None or score > best_tuple:
            best_tuple = score
            best_key = key

    return best_key


def _diabimmune_age_bin_label(age_days):
    edges = [27.0, 216.2, 405.3, 594.5, 783.7, 972.8]
    edges_i = [int(round(x)) for x in edges]
    if age_days < edges[0]:
        return f"<{edges_i[0]}"
    for i in range(len(edges_i) - 1):
        lo, hi = edges_i[i], edges_i[i + 1]
        if lo <= age_days < hi:
            return f"{lo}-{hi}"
    return f"{edges_i[-1]}+"


def _age_key_to_bin_label(age_key):
    v = safe_float(age_key)
    if v is None:
        return "unknown"
    return _diabimmune_age_bin_label(float(v))


def _bin_sort_key(label):
    s = str(label)
    if s.startswith("<"):
        try:
            return (0, float(s[1:]))
        except Exception:
            return (0, 0.0)
    if s.endswith("+"):
        try:
            return (2, float(s[:-1]))
        except Exception:
            return (2, 1e18)
    if "-" in s:
        try:
            lo = float(s.split("-", 1)[0])
            return (1, lo)
        except Exception:
            return (1, 1e18)
    if s == "unknown":
        return (3, 1e18)
    try:
        return (1, float(s))
    except Exception:
        return (3, 1e18)


def _load_sra_run_table_rows(paths):
    out = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = csv.DictReader(f)
            if r.fieldnames:
                r.fieldnames[0] = r.fieldnames[0].lstrip("\ufeff")
            for row in r:
                rid = str(row.get("Run", "")).strip()
                if rid:
                    out[rid] = row
    return out


def _assay_category(raw):
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if "amplicon" in s:
        return "amplicon"
    if "wgs" in s:
        return "WGS"
    return "Other"


def _assay_by_subject_age(samples_csv):
    """
    Return dict[(subjectID, age_key)] -> assay category in {WGS, amplicon, Other, mixed, ''}.
    """
    # sampleID -> age_key
    sample_to_age = {}
    with open(samples_csv) as f:
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
            sid = str(row.get("sampleID", "")).strip()
            age = row.get("age_at_collection", "")
            if sid and age != "":
                try:
                    sample_to_age[sid] = str(int(round(float(age))))
                except Exception:
                    continue

    _run_rows, sra_to_micro, _gid_to_sample, _micro_to_subject, micro_to_sample = load_run_data()
    # SRS -> list[Run]
    srs_to_runs = {}
    for rid, srs in sra_to_micro.items():
        srs_to_runs.setdefault(str(srs), []).append(str(rid))

    run_meta = _load_sra_run_table_rows([WGS_RUN_TABLE, EXTRA_RUN_TABLE])
    agg = {}
    for srs, info in micro_to_sample.items():
        subj = str(info.get("subject", "")).strip()
        sample_id = str(info.get("sample", "")).strip()
        if not subj or not sample_id:
            continue
        age_key = sample_to_age.get(sample_id, "")
        if not age_key:
            continue
        cats = []
        for rid in srs_to_runs.get(str(srs), []):
            row = run_meta.get(str(rid), {})
            cat = _assay_category(row.get("Assay Type", ""))
            if cat:
                cats.append(cat)
        if not cats:
            continue
        k = (subj, age_key)
        agg.setdefault(k, set()).update(cats)

    out = {}
    for k, cats in agg.items():
        cats = sorted(cats)
        if len(cats) == 1:
            out[k] = cats[0]
        else:
            out[k] = "mixed"
    return out


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

    # If a specific subject/start is requested, bypass cache so the request is honored.
    force_target = bool(str(args.subject).strip() or str(args.t_start).strip())
    use_cache = os.path.exists(args.cache) and (not force_target)
    cache = None
    if use_cache:
        cache = np.load(args.cache, allow_pickle=True)
        traj_mode = str(cache["traj_mode"].item()) if "traj_mode" in cache.files else ""
        pca_basis = str(cache["pca_basis"].item()) if "pca_basis" in cache.files else ""
        if traj_mode != "from_start" or pca_basis != "real":
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
            jacc_metric = str(cache["jacc_metric"].item()) if "jacc_metric" in cache.files else ""
            jacc_basis = str(cache["jacc_basis"].item()) if "jacc_basis" in cache.files else ""
            if jacc_metric != "exact" or jacc_basis != "real":
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
        if not subjects:
            raise SystemExit("No diabimmune subjects found in real cache.")

        def subj_has_multi(s):
            return len({t for (ss, t) in subj_time_to_emb.keys() if ss == s}) >= 2

        if args.subject.strip() and args.t_start.strip():
            subject = args.subject.strip()
            t_start = args.t_start.strip()
        else:
            valid_keys = set(subj_time_to_emb.keys())
            picked = pick_example_with_movement(ROLL_TSV, valid_keys=valid_keys)
            if picked is not None:
                subject, t_start = picked
            else:
                subject = args.subject.strip() or next((s for s in subjects if subj_has_multi(s)), subjects[0])
                t_start = args.t_start.strip() or ""

        times_for_subject_all = sorted({t for (s, t) in subj_time_to_emb.keys() if s == subject}, key=time_key)
        if not times_for_subject_all:
            raise SystemExit(f"No timepoints found for subject={subject}.")
        if not t_start:
            t_start = times_for_subject_all[0]
        if t_start not in set(times_for_subject_all):
            t_start = times_for_subject_all[0]

        # Plot the "real trajectory" only from the chosen start onwards.
        # Otherwise the start point is in the middle of a polyline and looks like two paths.
        try:
            start_idx = times_for_subject_all.index(t_start)
        except ValueError:
            start_idx = 0
        times_for_subject = times_for_subject_all[start_idx:]

        real_times = np.asarray([t for (_s, t) in subj_time_to_emb.keys()], dtype=str)
        real_embs = np.stack([v for v in subj_time_to_emb.values()], axis=0).astype(float)

        # Fit PCA on REAL samples only; endpoints are projected into that basis.
        mu, comps = pca2_fit(real_embs.astype(float))
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
        # Keep more points for the plotted rollout trajectory so it looks sequential/smooth.
        n_keep = int(np.ceil(args.subsample_frac * n_steps))
        n_keep = max(50, n_keep)
        n_keep = max(2, min(n_steps, n_keep))
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
        # Avoid duplicating step 0 (rows_sub often already includes step==0).
        if rollout_steps.size == 0 or int(rollout_steps[0]) != 0:
            rollout_xy = np.vstack([start_xy[None, :], rollout_xy])
            rollout_steps = np.concatenate([np.asarray([0], dtype=int), rollout_steps], axis=0)

        np.savez(
            args.cache,
            subject=np.asarray(subject, dtype=object),
            t_start=np.asarray(t_start, dtype=object),
            steps=steps,
            n_current=n_current,
            anchor_mean_logit=anchor_mean_logit if anchor_mean_logit is not None else np.asarray([], dtype=float),
            traj_mode=np.asarray("from_start", dtype=object),
            pca_basis=np.asarray("real", dtype=object),
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
                k = (str(subj), str(tt))
                prev = last_row_by_key.get(k)
                if prev is None or step >= int(prev.get("step", "0")):
                    last_row_by_key[k] = row

        end_idx_lists = []
        for i in range(len(subj_arr)):
            k = (str(subj_arr[i]), str(t1_arr[i]))
            rr = last_row_by_key.get(k, {})
            idx_list = parse_index_list(rr.get("current_otu_indices", ""))
            end_idx_lists.append(np.asarray([j for j in idx_list if 0 <= j < len(all_otus)], dtype=np.int32))

        # Build a Jaccard PCoA basis using landmarks sampled from REAL points only;
        # endpoints are projected into that basis.
        real_n = len(real_idx_all)
        L = min(int(JACCARD_LANDMARKS), max(2, real_n))
        rng = np.random.default_rng(0)
        landmark_idx = rng.choice(real_n, size=L, replace=False) if real_n > L else np.arange(real_n, dtype=int)
        idx_L = [real_idx_all[int(i)] for i in landmark_idx.tolist()]
        D_LL = _jaccard_distance_matrix(idx_L, idx_L)
        X_L, eigvals, row_mean, grand_mean = _pcoa_2d_from_distance(D_LL)

        D_real_L = _jaccard_distance_matrix(real_idx_all, idx_L)
        jacc_real_xy = _pcoa_project_many(D_real_L, X_L, eigvals, row_mean, grand_mean).astype(np.float32)
        D_end_L = _jaccard_distance_matrix(end_idx_lists, idx_L)
        jacc_end_xy = _pcoa_project_many(D_end_L, X_L, eigvals, row_mean, grand_mean).astype(np.float32)

        times_for_subject_all = sorted({t for (s, t) in real_keys if s == subject}, key=time_key)
        try:
            start_idx = times_for_subject_all.index(t_start)
        except ValueError:
            start_idx = 0
        times_for_subject = times_for_subject_all[start_idx:]
        real_traj_idx = []
        for t in times_for_subject:
            otus = subject_age_to_otus.get((subject, t), set())
            real_traj_idx.append(np.asarray([otu_to_idx[o] for o in otus if o in otu_to_idx], dtype=np.int32))
        D_rtL = _jaccard_distance_matrix(real_traj_idx, idx_L)
        jacc_real_traj_xy = _pcoa_project_many(D_rtL, X_L, eigvals, row_mean, grand_mean).astype(np.float32)

        rows = load_rollout_rows(subject, t_start)
        row_by_step = {int(r["step"]): r for r in rows}
        rollout_idx = []
        start_otus = subject_age_to_otus.get((subject, t_start), set())
        rollout_idx.append(np.asarray([otu_to_idx[o] for o in start_otus if o in otu_to_idx], dtype=np.int32))
        for st in rollout_steps[1:]:
            rr = row_by_step.get(int(st))
            idx_list = parse_index_list(rr.get("current_otu_indices", "")) if rr else []
            rollout_idx.append(np.asarray([j for j in idx_list if 0 <= j < len(all_otus)], dtype=np.int32))

        D_rollL = _jaccard_distance_matrix(rollout_idx, idx_L)
        jacc_rollout_xy = _pcoa_project_many(D_rollL, X_L, eigvals, row_mean, grand_mean).astype(np.float32)

        key_to_xy = {real_keys[i]: jacc_real_xy[i] for i in range(len(real_keys))}
        end_by_parent = {}
        for i in range(len(subj_arr)):
            end_by_parent.setdefault((str(subj_arr[i]), str(t1_arr[i])), []).append(jacc_end_xy[i])
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
            traj_mode=np.asarray("from_start", dtype=object),
            pca_basis=np.asarray("real", dtype=object),
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
            jacc_metric=np.asarray("exact", dtype=object),
            jacc_basis=np.asarray("real", dtype=object),
        )

    plt.style.use("seaborn-v0_8-white")
    if FONT_SIZE is not None:
        plt.rcParams.update(
            {
                "font.size": FONT_SIZE,
                "axes.titlesize": FONT_SIZE + 2,
                "axes.labelsize": FONT_SIZE,
                "legend.fontsize": max(1, FONT_SIZE - 1),
                "xtick.labelsize": max(1, FONT_SIZE - 2),
                "ytick.labelsize": max(1, FONT_SIZE - 2),
            }
        )
    fig = plt.figure(figsize=(14.0, 10.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])
    ax1 = fig.add_subplot(gs[0, :])  # PCA panel (full width)
    ax2 = fig.add_subplot(gs[1, 0])  # Jaccard
    ax3 = fig.add_subplot(gs[1, 1])  # Cohort in Jaccard coords

    # Color map by age bin (use same scheme as other diabimmune plots).
    pca_real_bins = np.asarray([_age_key_to_bin_label(t) for t in pca_real_times.tolist()], dtype=str)
    pca_t1_bins = np.asarray([_age_key_to_bin_label(t) for t in pca_t1_labels.tolist()], dtype=str)
    bins = sorted(set(pca_real_bins.tolist()) | set(pca_t1_bins.tolist()), key=_bin_sort_key)
    cmap3 = plt.cm.plasma
    color_by_bin = {b: cmap3(i / float(len(bins) - 1)) if len(bins) > 1 else cmap3(0.5) for i, b in enumerate(bins)}

    ax1.scatter(
        pca_end_xy[:, 0],
        pca_end_xy[:, 1],
        s=10,
        marker=".",
        color=JACCARD_ENDPOINT_COLOR,
        alpha=JACCARD_ENDPOINT_ALPHA,
        linewidths=0,
        label="Rollout endpoints",
        zorder=0,
    )
    for b in bins:
        mask_r = pca_real_bins == b
        if np.any(mask_r):
            ax1.scatter(
                pca_real_xy[mask_r, 0],
                pca_real_xy[mask_r, 1],
                s=18,
                marker="o",
                color=color_by_bin[b],
                alpha=PCA_REAL_POINTS_ALPHA,
                linewidths=0,
                zorder=1,
            )

    pca_px_plot, pca_py_plot, pca_dx_plot, pca_dy_plot = _merge_arrows_on_grid(
        arrow_px,
        arrow_py,
        arrow_dx,
        arrow_dy,
        PCA_ARROW_MERGE_BINS,
    )

    if pca_px_plot.size:
        ax1.quiver(
            pca_px_plot,
            pca_py_plot,
            pca_dx_plot,
            pca_dy_plot,
            angles="xy",
            scale_units="xy",
            pivot="tail",
            scale=8,
            color="black",
            width=0.0016,
            headwidth=PCA_ARROW_HEADWIDTH,
            headlength=PCA_ARROW_HEADLENGTH,
            headaxislength=PCA_ARROW_HEADAXISLENGTH,
            alpha=0.14,
            zorder=2,
        )

    (real_line,) = ax1.plot(
        real_traj_xy[:, 0],
        real_traj_xy[:, 1],
        color=REAL_TRAJ_COLOR,
        alpha=0.70,
        linewidth=REAL_TRAJ_LINEWIDTH,
        zorder=4,
    )
    real_line.set_path_effects([pe.Stroke(linewidth=TRAJ_STROKE_LINEWIDTH, foreground="white"), pe.Normal()])
    ax1.scatter(
        real_traj_xy[:, 0],
        real_traj_xy[:, 1],
        s=42,
        facecolors="none",
        edgecolors=REAL_TRAJ_COLOR,
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
        edgecolors=ROLLOUT_TRAJ_COLOR,
        linewidths=1.4,
        zorder=7,
        label="Rollout trajectory",
    )
    (rollout_line,) = ax1.plot(
        rollout_xy[:, 0],
        rollout_xy[:, 1],
        color=ROLLOUT_TRAJ_COLOR,
        alpha=0.55,
        linewidth=ROLLOUT_TRAJ_LINEWIDTH,
        zorder=5,
    )
    rollout_line.set_path_effects([pe.Stroke(linewidth=TRAJ_STROKE_LINEWIDTH, foreground="white"), pe.Normal()])
    ax1.scatter(
        rollout_xy[:, 0],
        rollout_xy[:, 1],
        s=10,
        marker=".",
        color=ROLLOUT_TRAJ_COLOR,
        alpha=0.95,
        linewidths=0,
        zorder=6,
        label="_nolegend_",
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title("Stability space PCA")

    # Inset: anchor_mean_logit over steps (no n_current line).
    if anchor_mean_logit is not None and len(anchor_mean_logit) == len(steps):
        ax_in = inset_axes(ax1, width="35%", height="28%", loc="upper right", borderpad=0.9)
        ax_in.plot(steps, anchor_mean_logit, color=ROLLOUT_TRAJ_COLOR, linewidth=1.4, alpha=0.9)
        ax_in.scatter(steps, anchor_mean_logit, s=10, color=ROLLOUT_TRAJ_COLOR, alpha=0.95, linewidths=0)
        ax_in.set_xlabel("rollout step", fontsize=7, labelpad=1)
        ax_in.set_ylabel("anchor mean logit", fontsize=7, labelpad=1)
        ax_in.tick_params(axis="both", labelsize=7, length=2)
        ax_in.grid(True, alpha=0.2)
        ax_in.set_facecolor("white")

    class _HandlerArrowTop(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            p = FancyArrowPatch(
                (xdescent, ydescent + 0.5 * height),
                (xdescent + width, ydescent + 0.5 * height),
                arrowstyle="-|>",
                mutation_scale=fontsize * 1.5,
                linewidth=1.0,
                color="#5f5f5f",
                alpha=0.35,
            )
            p.set_transform(trans)
            return [p]

    class _HandlerArrowBottom(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            p = FancyArrowPatch(
                (xdescent, ydescent + 0.5 * height),
                (xdescent + width, ydescent + 0.5 * height),
                arrowstyle="-|>",
                mutation_scale=fontsize * 1.5,
                linewidth=1.0,
                color=JACCARD_ARROW_COLOR,
                alpha=max(0.6, float(JACCARD_ARROW_ALPHA)),
            )
            p.set_transform(trans)
            return [p]

    arrow_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", label="Direction to endpoint")
    real_start_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="#111111",
        markeredgewidth=1.3,
        markersize=8.0,
        label="Real start sample",
    )
    handles, labels = ax1.get_legend_handles_labels()
    top_handles = []
    top_labels = []
    for h, l in zip(handles, labels):
        if l == "Rollout endpoints":
            top_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=JACCARD_ENDPOINT_COLOR,
                    markeredgecolor=JACCARD_ENDPOINT_COLOR,
                    markeredgewidth=0.0,
                    alpha=JACCARD_ENDPOINT_ALPHA,
                    markersize=5.5,
                    label=l,
                )
            )
        elif l == "Real trajectory":
            top_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    color=REAL_TRAJ_COLOR,
                    alpha=0.85,
                    linewidth=REAL_TRAJ_LINEWIDTH,
                    label=l,
                )
            )
        elif l == "Rollout trajectory":
            top_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    color=ROLLOUT_TRAJ_COLOR,
                    alpha=0.85,
                    linewidth=ROLLOUT_TRAJ_LINEWIDTH,
                    label=l,
                )
            )
        else:
            top_handles.append(h)
        top_labels.append(l)

    handles = [arrow_handle, real_start_handle] + top_handles
    labels = ["Direction to endpoint", "Real start sample"] + top_labels
    traj_legend = ax1.legend(
        handles=handles,
        labels=labels,
        frameon=True,
        loc="lower right",
        handler_map={FancyArrowPatch: _HandlerArrowTop()},
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
            alpha=PCA_REAL_POINTS_ALPHA,
            label=str(b),
        )
        for b in bins
    ]
    bin_legend = ax1.legend(
        handles=bin_handles,
        title="Age bin (days)",
        frameon=True,
        loc="upper left",
        markerscale=1.5,
        fontsize=AGE_BIN_LEGEND_FONT_SIZE,
    )
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
        color=JACCARD_ENDPOINT_COLOR,
        alpha=JACCARD_ENDPOINT_ALPHA,
        linewidths=0,
        label="Rollout endpoints",
        zorder=0,
    )
    if JACCARD_REAL_USE_AGE_COLORS:
        for b in bins:
            mask_r = pca_real_bins == b
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
    else:
        ax2.scatter(
            jacc_real_xy[:, 0],
            jacc_real_xy[:, 1],
            s=18,
            marker="o",
            color=JACCARD_REAL_POINT_COLOR,
            alpha=JACCARD_REAL_POINT_ALPHA,
            linewidths=0,
            zorder=1,
            label="Real samples",
        )

    jacc_px_plot, jacc_py_plot, jacc_dx_plot, jacc_dy_plot = _merge_arrows_on_grid(
        jacc_arrow_px,
        jacc_arrow_py,
        jacc_arrow_dx,
        jacc_arrow_dy,
        JACCARD_ARROW_MERGE_BINS,
    )

    if jacc_px_plot.size:
        ax2.quiver(
            jacc_px_plot,
            jacc_py_plot,
            jacc_dx_plot,
            jacc_dy_plot,
            angles="xy",
            scale_units="xy",
            pivot="tail",
            scale=JACCARD_ARROW_SCALE,
            color=JACCARD_ARROW_COLOR,
            width=JACCARD_ARROW_WIDTH,
            headwidth=JACCARD_ARROW_HEADWIDTH,
            headlength=JACCARD_ARROW_HEADLENGTH,
            headaxislength=JACCARD_ARROW_HEADAXISLENGTH,
            alpha=JACCARD_ARROW_ALPHA,
            zorder=2,
        )

    ax2.set_xlabel("PCoA1 (Jaccard)")
    ax2.set_ylabel("PCoA2 (Jaccard)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("OTU Jaccard PCoA")

    # Legend for Jaccard panel (match gingivitis style).
    arrow_handle2 = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", label="Direction to endpoint")
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2_mod = []
    labels2_mod = []
    for h, l in zip(handles2, labels2):
        if l == "Rollout endpoints":
            handles2_mod.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=JACCARD_ENDPOINT_COLOR,
                    markeredgecolor=JACCARD_ENDPOINT_COLOR,
                    markeredgewidth=0.0,
                    alpha=JACCARD_ENDPOINT_ALPHA,
                    markersize=5.5,
                    label=l,
                )
            )
        elif l == "Real samples":
            handles2_mod.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=JACCARD_REAL_POINT_COLOR,
                    markeredgecolor=JACCARD_REAL_POINT_COLOR,
                    markeredgewidth=0.0,
                    alpha=JACCARD_REAL_POINT_ALPHA,
                    markersize=7.5,
                    label=l,
                )
            )
        else:
            handles2_mod.append(h)
        labels2_mod.append(l)

    handles2 = handles2_mod
    labels2 = labels2_mod
    drop_labels = {"Real trajectory", "Rollout start", "Rollout trajectory"}
    kept = [(h, l) for (h, l) in zip(handles2, labels2) if l not in drop_labels]
    handles2 = [arrow_handle2] + [h for (h, _l) in kept]
    labels2 = ["Direction to endpoint"] + [l for (_h, l) in kept]
    leg2 = ax2.legend(
        handles=handles2,
        labels=labels2,
        frameon=True,
        loc="lower right",
        handler_map={FancyArrowPatch: _HandlerArrowBottom()},
    )
    leg2.get_frame().set_edgecolor("0.7")
    leg2.get_frame().set_linewidth(0.8)
    leg2.get_frame().set_facecolor("white")
    leg2.get_frame().set_alpha(0.9)

    # Assay-type panel in the same Jaccard coordinates.
    inst_by_key = _assay_by_subject_age(args.samples_csv)
    real_cache = np.load(REAL_NPZ, allow_pickle=True)
    keys_raw = real_cache["keys"]
    real_keys = [(str(keys_raw[i][0]), str(keys_raw[i][1])) for i in range(len(keys_raw))]
    inst_labels = np.asarray([inst_by_key.get(k, "") for k in real_keys], dtype=str)

    if SHOW_INSTRUMENT_PANEL_ENDPOINTS:
        ax3.scatter(
            jacc_end_xy[:, 0],
            jacc_end_xy[:, 1],
            s=10,
            marker=".",
            color="#666666",
            alpha=0.25,
            linewidths=0,
            zorder=0,
            label="Rollout endpoints",
        )

    order = ["amplicon", "WGS", "mixed", "Other"]
    uniq = [c for c in order if c in set(inst_labels.tolist())]
    cmap = plt.cm.Set2
    color_by = {c: cmap(i % cmap.N) for i, c in enumerate(uniq)}
    for c in uniq:
        m = inst_labels == c
        if np.any(m):
            ax3.scatter(
                jacc_real_xy[m, 0],
                jacc_real_xy[m, 1],
                s=18,
                marker="o",
                color=color_by[c],
                alpha=0.85,
                linewidths=0,
                zorder=1,
                label=c,
            )

    # Unknown assay type
    m_unk = inst_labels == ""
    if np.any(m_unk):
        ax3.scatter(
            jacc_real_xy[m_unk, 0],
            jacc_real_xy[m_unk, 1],
            s=12,
            marker="o",
            color="#bbbbbb",
            alpha=0.55,
            linewidths=0,
            zorder=1,
            label="unknown",
        )

    ax3.set_xlabel("PCoA1 (Jaccard)")
    ax3.set_ylabel("PCoA2 (Jaccard)")
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.set_title("Assay type in Jaccard space")
    ax3.legend(frameon=True, loc="best", markerscale=1.2, fontsize=8)

    if HIDE_XY_TICK_LABELS:
        for ax in (ax1, ax2, ax3):
            ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=350)
    print(f"Saved: {args.out}")
    print(f"Saved cache: {args.cache}")


if __name__ == "__main__":
    main()
