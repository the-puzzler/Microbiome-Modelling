#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


ENDPOINTS_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoints_cache.npz")
REAL_NPZ = os.path.join("data", "diabimmune", "diabimmune_real_embeddings_cache.npz")
OUT_PNG = os.path.join("data", "diabimmune", "diabimmune_real_vs_rollouts_pc1_pc2.png")


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


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    endpoints_cache = np.load(ENDPOINTS_NPZ, allow_pickle=True)
    endpoints = endpoints_cache["endpoints"].astype(float)
    t1_labels = endpoints_cache["t1_labels"].astype(str)
    subj_arr = endpoints_cache["subject"].astype(str)
    t1_arr = endpoints_cache["t1"].astype(str)

    real_cache = np.load(REAL_NPZ, allow_pickle=True)
    keys = real_cache["keys"]
    emb = real_cache["emb"].astype(float)

    subj_time_to_emb = {(str(keys[i][0]), str(keys[i][1])): emb[i] for i in range(len(keys))}

    real_keys = list(subj_time_to_emb.keys())
    real_times = np.asarray([age_bin_label(t) for (_s, t) in real_keys], dtype=str)
    real_embs = np.stack([subj_time_to_emb[k] for k in real_keys], axis=0).astype(float)

    mu, comps = pca2_fit(real_embs)
    real_xy = pca2_transform(real_embs, mu, comps)
    end_xy = pca2_transform(endpoints, mu, comps)
    real_key_to_xy = {real_keys[i]: real_xy[i] for i in range(len(real_keys))}

    t1_binned = np.asarray([age_bin_label(t) for t in t1_labels.tolist()], dtype=str)
    bin_order = ["27-216.2", "216.2-405.3", "405.3-594.5", "594.5-783.7", "783.7-972.8", "972.8-"]
    timepoints = [b for b in bin_order if (b in set(real_times.tolist()) or b in set(t1_binned.tolist()))]
    colors = cm.viridis(np.linspace(0.0, 1.0, max(2, len(timepoints))))
    color_by_time = {t: colors[i] for i, t in enumerate(timepoints)}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 8.5))

    # Average arrow per real point (subject,t_start): parent -> mean rollout endpoint.
    by_parent = {}
    for i in range(len(endpoints)):
        by_parent.setdefault((subj_arr[i], t1_arr[i]), []).append(end_xy[i])

    px, py, dx, dy = [], [], [], []
    for (subj, t_start), pts in by_parent.items():
        parent_xy = real_key_to_xy[(subj, t_start)]
        mean_xy = np.mean(np.stack(pts, axis=0), axis=0)
        px.append(parent_xy[0])
        py.append(parent_xy[1])
        dx.append(mean_xy[0] - parent_xy[0])
        dy.append(mean_xy[1] - parent_xy[1])

    if len(px) > 0:
        ax.quiver(
            np.asarray(px),
            np.asarray(py),
            np.asarray(dx),
            np.asarray(dy),
            angles="xy",
            scale_units="xy",
            pivot="tail",
            scale=8.0,
            color="black",
            width=0.0016,
            headwidth=3,
            headlength=4,
            headaxislength=3,
            alpha=0.18,
            zorder=2,
        )

    ax.scatter(
        end_xy[:, 0],
        end_xy[:, 1],
        s=10,
        marker=".",
        color="black",
        alpha=0.35,
        linewidths=0,
        label="Rollout endpoints",
    )

    for t in timepoints:
        mask_r = real_times == t
        if np.any(mask_r):
            ax.scatter(
                real_xy[mask_r, 0],
                real_xy[mask_r, 1],
                s=22,
                marker="o",
                color=color_by_time[t],
                alpha=0.9,
                linewidths=0,
            )

    type_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="k", markersize=7, label="Real"),
        Line2D([0], [0], marker=".", color="k", markersize=10, linestyle="None", label="Rollout endpoints"),
    ]
    type_legend = ax.legend(handles=type_handles, frameon=False, loc="lower right")
    ax.add_artist(type_legend)

    time_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_by_time[t], markersize=7, label=str(t))
        for t in timepoints
    ]
    ax.legend(handles=time_handles, title="Age bin (days)", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_title("DIABIMMUNE: real embeddings + rollout endpoints (PC1/PC2)")

    # Subplot: connect each parent (subject,t_start) to its rollout endpoints.
    parent_xy_by_endpoint = np.stack([real_key_to_xy[(subj_arr[i], t1_arr[i])] for i in range(len(end_xy))], axis=0)
    segs = np.stack([parent_xy_by_endpoint, end_xy], axis=1)
    ax2.add_collection(LineCollection(segs, colors=[(0, 0, 0, 0.06)], linewidths=0.7, zorder=1))
    ax2.scatter(end_xy[:, 0], end_xy[:, 1], s=8, marker=".", color="black", alpha=0.18, linewidths=0, zorder=2)
    for t in timepoints:
        mask_r = real_times == t
        if np.any(mask_r):
            ax2.scatter(
                real_xy[mask_r, 0],
                real_xy[mask_r, 1],
                s=18,
                marker="o",
                color=color_by_time[t],
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_box_aspect(1)
    ax2.set_title("Parents connected to rollouts")

    x_all = np.concatenate([real_xy[:, 0], end_xy[:, 0]])
    y_all = np.concatenate([real_xy[:, 1], end_xy[:, 1]])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    r = 0.5 * max(x_max - x_min, y_max - y_min)
    xc = 0.5 * (x_min + x_max)
    yc = 0.5 * (y_min + y_max)
    pad = 0.04 * (2.0 * r)
    for a in (ax, ax2):
        a.set_xlim(xc - r - pad, xc + r + pad)
        a.set_ylim(yc - r - pad, yc + r + pad)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
