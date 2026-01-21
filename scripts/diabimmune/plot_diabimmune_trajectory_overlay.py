#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402


ROLL_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob.tsv")
ENDPOINTS_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoints_cache.npz")
REAL_NPZ = os.path.join("data", "diabimmune", "diabimmune_real_embeddings_cache.npz")

OUT_PNG = os.path.join("data", "diabimmune", "diabimmune_rollout_trajectory_overlay.png")
CACHE_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_trajectory_overlay_cache.npz")


def parse_args():
    p = argparse.ArgumentParser(description="DIABIMMUNE: counts + PC1/PC2 vector field + example trajectories.")
    p.add_argument("--out", default=OUT_PNG)
    p.add_argument("--cache", default=CACHE_NPZ)
    p.add_argument("--subject", default="", help="Subject id (default: first available with a rollout).")
    p.add_argument("--t-start", default="", help="Start day for rollout (default: first available for subject).")
    p.add_argument("--subsample-frac", type=float, default=0.05, help="Fraction of rollout steps to embed/plot.")
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
    return float(str(t).strip())


def age_bin_label(age):
    a = float(age)
    if a < 216.2:
        return "27-216.2"
    if a < 405.3:
        return "216.2-405.3"
    if a < 594.5:
        return "405.3-594.5"
    if a < 783.7:
        return "594.5-783.7"
    if a < 972.8:
        return "783.7-972.8"
    return "972.8-"


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


def load_rollout_pairs():
    pairs = []
    with open(ROLL_TSV, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            subj = row.get("subject", "").strip()
            t_start = row.get("t_start", "").strip()
            if subj and t_start:
                pairs.append((subj, t_start))
    return sorted(set(pairs), key=lambda x: (x[0], time_key(x[1])))


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


def style_legend_frame(legend):
    legend.get_frame().set_edgecolor("0.7")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    if os.path.exists(args.cache):
        cache = np.load(args.cache, allow_pickle=True)
        subject = str(cache["subject"].item())
        t_start = str(cache["t_start"].item())
        steps = cache["steps"].astype(int)
        n_current = cache["n_current"].astype(int)
        n_same = cache["n_same"].astype(int)
        n_added = cache["n_added"].astype(int)
        n_removed = cache["n_removed"].astype(int)
        real_traj_xy = cache["real_traj_xy"].astype(float)
        rollout_xy = cache["rollout_xy"].astype(float)
        rollout_steps = cache["rollout_steps"].astype(int)
        pca_real_xy = cache["pca_real_xy"].astype(float)
        pca_end_xy = cache["pca_end_xy"].astype(float)
        pca_real_bins = cache["pca_real_bins"].astype(str)
        pca_t1_bins = cache["pca_t1_bins"].astype(str)
        arrow_px = cache["arrow_px"].astype(float)
        arrow_py = cache["arrow_py"].astype(float)
        arrow_dx = cache["arrow_dx"].astype(float)
        arrow_dy = cache["arrow_dy"].astype(float)
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

        rollout_pairs = load_rollout_pairs()
        if not rollout_pairs:
            raise SystemExit(f"No rollout pairs found in {ROLL_TSV}")

        subjects_with_rollout = sorted({s for (s, _t) in rollout_pairs})
        subject = args.subject.strip() or subjects_with_rollout[0]
        if subject not in subjects_with_rollout:
            raise SystemExit(f"Subject {subject!r} not found in rollout TSV.")

        starts_for_subject = [t for (s, t) in rollout_pairs if s == subject]
        t_start = args.t_start.strip() or starts_for_subject[0]
        if t_start not in starts_for_subject:
            raise SystemExit(f"t_start {t_start!r} not found for subject {subject!r} in rollout TSV.")

        times_for_subject = sorted({t for (s, t) in subj_time_to_emb.keys() if s == subject}, key=time_key)
        if not times_for_subject:
            raise SystemExit(f"No real embeddings found for subject {subject!r}.")

        real_days = np.asarray([t for (_s, t) in subj_time_to_emb.keys()], dtype=str)
        real_bins = np.asarray([age_bin_label(t) for t in real_days.tolist()], dtype=str)
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
        n_same = np.asarray([int(r["n_same"]) for r in rows], dtype=int)
        n_added = np.asarray([int(r["n_added"]) for r in rows], dtype=int)
        n_removed = np.asarray([int(r["n_removed"]) for r in rows], dtype=int)

        n_steps = len(rows)
        n_keep = max(2, int(np.ceil(args.subsample_frac * n_steps)))
        idxs = np.unique(np.round(np.linspace(0, n_steps - 1, n_keep)).astype(int))
        idxs = np.unique(np.concatenate([idxs, np.asarray([0, n_steps - 1], dtype=int)]))
        rows_sub = [rows[i] for i in idxs.tolist()]
        rollout_steps = np.asarray([int(r["step"]) for r in rows_sub], dtype=int)

        _, sra_to_micro, _gid_to_sample, micro_to_subject, _micro_to_sample = load_run_data()
        micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)
        all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})

        model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
        rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
        resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

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

        t1_bins = np.asarray([age_bin_label(t) for t in t1_labels.tolist()], dtype=str)

        np.savez(
            args.cache,
            subject=np.asarray(subject, dtype=object),
            t_start=np.asarray(t_start, dtype=object),
            steps=steps,
            n_current=n_current,
            n_same=n_same,
            n_added=n_added,
            n_removed=n_removed,
            real_traj_xy=real_traj_xy,
            rollout_xy=rollout_xy,
            rollout_steps=rollout_steps,
            pca_real_xy=pca_real_xy,
            pca_end_xy=pca_end_xy,
            pca_real_bins=real_bins,
            pca_t1_bins=t1_bins,
            arrow_px=arrow_px,
            arrow_py=arrow_py,
            arrow_dx=arrow_dx,
            arrow_dy=arrow_dy,
        )

        pca_real_bins = real_bins
        pca_t1_bins = t1_bins

    # Plot
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(12.0, 10.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.65])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    ax0.plot(steps, n_current, label="n_current", linewidth=2.0)
    ax0.plot(steps, n_same, label="n_same", linewidth=1.4)
    ax0.plot(steps, n_added, label="n_added", linewidth=1.4)
    ax0.plot(steps, n_removed, label="n_removed", linewidth=1.4)
    ax0.set_xlabel("step")
    ax0.set_ylabel("count")
    ax0.set_title(f"Rollout counts ({subject}, start_day={t_start})")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False, ncol=2)

    bin_order = ["27-216.2", "216.2-405.3", "405.3-594.5", "594.5-783.7", "783.7-972.8", "972.8-"]
    timepoints = [b for b in bin_order if (b in set(pca_real_bins.tolist()) or b in set(pca_t1_bins.tolist()))]
    uniq_bins = timepoints
    cmap3 = plt.cm.plasma
    color_by_time = {
        b: cmap3(i / float(len(uniq_bins) - 1)) if len(uniq_bins) > 1 else cmap3(0.5) for i, b in enumerate(uniq_bins)
    }

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
        mask_r = pca_real_bins == t
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
    style_legend_frame(traj_legend)
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
        title="Age bin (days)",
        frameon=True,
        loc="upper left",
        markerscale=1.5,
        fontsize=8,
    )
    style_legend_frame(time_legend)

    x_all = np.concatenate([pca_real_xy[:, 0], pca_end_xy[:, 0]])
    y_all = np.concatenate([pca_real_xy[:, 1], pca_end_xy[:, 1]])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    x_pad = 0.05 * (x_max - x_min + 1e-12)
    y_pad = 0.05 * (y_max - y_min + 1e-12)
    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")
    print(f"Saved cache: {args.cache}")


if __name__ == "__main__":
    main()
