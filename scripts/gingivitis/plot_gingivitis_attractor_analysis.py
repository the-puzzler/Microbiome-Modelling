#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


ENDPOINTS_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_endpoints_cache.npz")
DIRECTION_NPZ = os.path.join("data", "gingivitis", "visionary_rollout_direction_cache.npz")
OUT_PNG = os.path.join("data", "gingivitis", "gingivitis_real_vs_rollouts_pc1_pc2.png")


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

def smooth_vector_field(px, py, vx, vy, xs, ys, sigma):
    X, Y = np.meshgrid(xs, ys)
    Vx = np.zeros_like(X, dtype=float)
    Vy = np.zeros_like(Y, dtype=float)
    W = np.zeros_like(X, dtype=float)

    sigma2 = float(sigma) ** 2
    for i in range(len(px)):
        dx = X - px[i]
        dy = Y - py[i]
        w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma2))
        Vx += w * vx[i]
        Vy += w * vy[i]
        W += w

    Vx = Vx / (W + 1e-12)
    Vy = Vy / (W + 1e-12)
    return Vx, Vy, W


def bilinear_sample(grid, xs, ys, x, y):
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))

    x0, x1 = float(xs[i]), float(xs[i + 1])
    y0, y1 = float(ys[j]), float(ys[j + 1])
    tx = 0.0 if x1 == x0 else (float(x) - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (float(y) - y0) / (y1 - y0)

    g00 = float(grid[j, i])
    g10 = float(grid[j, i + 1])
    g01 = float(grid[j + 1, i])
    g11 = float(grid[j + 1, i + 1])
    return (1 - tx) * (1 - ty) * g00 + tx * (1 - ty) * g10 + (1 - tx) * ty * g01 + tx * ty * g11


def trace_line(x0, y0, Vx, Vy, xs, ys, step, n_steps=120, min_speed=1e-6):
    x, y = float(x0), float(y0)
    pts = [(x, y)]
    x_min, x_max = float(xs[0]), float(xs[-1])
    y_min, y_max = float(ys[0]), float(ys[-1])

    for _ in range(int(n_steps)):
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            break
        vx = bilinear_sample(Vx, xs, ys, x, y)
        vy = bilinear_sample(Vy, xs, ys, x, y)
        sp = float(np.hypot(vx, vy))
        if not np.isfinite(sp) or sp < min_speed:
            break
        x += step * vx / sp
        y += step * vy / sp
        pts.append((x, y))
    return np.asarray(pts, dtype=float)


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

    direction_cache = np.load(DIRECTION_NPZ, allow_pickle=True)
    keys = direction_cache["keys"]
    emb = direction_cache["emb"].astype(float)

    subj_time_to_emb = {}
    for i in range(len(keys)):
        subj, t = keys[i]
        subj_time_to_emb[(str(subj), str(t))] = emb[i]

    real_times = []
    real_keys = []
    real_embs = []
    for (subj, t), v in subj_time_to_emb.items():
        real_keys.append((subj, str(t)))
        real_times.append(str(t))
        real_embs.append(v)
    real_embs = np.stack(real_embs, axis=0).astype(float)
    real_times = np.asarray(real_times, dtype=str)

    mu, comps = pca2_fit(real_embs)
    real_xy = pca2_transform(real_embs, mu, comps)
    end_xy = pca2_transform(endpoints, mu, comps)
    real_key_to_xy = {real_keys[i]: real_xy[i] for i in range(len(real_keys))}

    timepoints = sorted(set(real_times.tolist()) | set(t1_labels.tolist()), key=time_key)
    colors = cm.viridis(np.linspace(0.0, 1.0, max(2, len(timepoints))))
    color_by_time = {t: colors[i] for i, t in enumerate(timepoints)}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 8.5))

    # Average arrow per real point (subject,timepoint): parent -> mean rollout endpoint.
    px, py, dx, dy = [], [], [], []
    subj_arr = endpoints_cache["subject"].astype(str)
    t1_arr = endpoints_cache["t1"].astype(str)

    by_parent = {}
    for i in range(len(endpoints)):
        k = (subj_arr[i], t1_arr[i])
        by_parent.setdefault(k, []).append(end_xy[i])

    for (subj, t1), pts in by_parent.items():
        parent_xy = real_key_to_xy[(subj, t1)]
        mean_xy = np.mean(np.stack(pts, axis=0), axis=0)
        px.append(parent_xy[0])
        py.append(parent_xy[1])
        dx.append(mean_xy[0] - parent_xy[0])
        dy.append(mean_xy[1] - parent_xy[1])

    if len(px) > 0:
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)

        # Smoothed flow-line overlay (disabled for now; kept for later experiments).
        # x_all = np.concatenate([real_xy[:, 0], end_xy[:, 0]])
        # y_all = np.concatenate([real_xy[:, 1], end_xy[:, 1]])
        # x_pad = 0.05 * float(np.ptp(x_all))
        # y_pad = 0.05 * float(np.ptp(y_all))
        # xs = np.linspace(float(np.min(x_all) - x_pad), float(np.max(x_all) + x_pad), 55)
        # ys = np.linspace(float(np.min(y_all) - y_pad), float(np.max(y_all) + y_pad), 55)
        # sigma = 0.18 * max(float(np.ptp(xs)), float(np.ptp(ys)))
        # Vx, Vy, _w = smooth_vector_field(px, py, dx, dy, xs, ys, sigma=sigma)
        #
        # rng = np.random.default_rng(0)
        # x_span = float(xs[-1] - xs[0])
        # y_span = float(ys[-1] - ys[0])
        # min_sep = 0.22
        # starts = []
        # for _ in range(2000):
        #     if len(starts) >= 10:
        #         break
        #     sx = float(rng.uniform(xs[0], xs[-1]))
        #     sy = float(rng.uniform(ys[0], ys[-1]))
        #     if starts:
        #         d = np.asarray(starts, dtype=float) - np.asarray([sx, sy], dtype=float)
        #         d[:, 0] /= max(x_span, 1e-12)
        #         d[:, 1] /= max(y_span, 1e-12)
        #         if float(np.min(np.hypot(d[:, 0], d[:, 1]))) < min_sep:
        #             continue
        #     starts.append([sx, sy])
        #
        # step_len = 0.55 * min(float(xs[1] - xs[0]), float(ys[1] - ys[0]))
        # for sx, sy in starts:
        #     pts = trace_line(sx, sy, Vx, Vy, xs, ys, step=step_len, n_steps=70)
        #     if pts.shape[0] < 6:
        #         continue
        #     ax.plot(pts[:, 0], pts[:, 1], color="black", alpha=0.08, linewidth=1.2, zorder=1)
        #     ax.annotate(
        #         "",
        #         xy=(float(pts[-1, 0]), float(pts[-1, 1])),
        #         xytext=(float(pts[-2, 0]), float(pts[-2, 1])),
        #         arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1, alpha=0.18),
        #         zorder=2,
        #     )

        ax.quiver(
            px,
            py,
            dx,
            dy,
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
    ax.legend(handles=time_handles, title="Timepoint", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Real embeddings + rollout endpoints (PC1/PC2)")

    # Subplot: connect each parent (subject,t1) to each of its rollout endpoints.
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
    ax2.set_title("Parents connected to rollouts")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
