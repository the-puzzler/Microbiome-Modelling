#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


REAL_NPZ = os.path.join("data", "gingivitis", "visionary_rollout_direction_cache_metropolis.npz")
ENDPOINTS_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_endpoints_cache_metropolis.npz")
OUT_PNG = os.path.join("data", "rollout_metropolis", "gingivitis_pc1_shift_arrows.png")


def parse_args():
    p = argparse.ArgumentParser(description="Gingivitis Metropolis: show per-start PC1 shift (start -> endpoint) as arrows.")
    p.add_argument("--real-npz", default=REAL_NPZ)
    p.add_argument("--endpoints-npz", default=ENDPOINTS_NPZ)
    p.add_argument("--out", default=OUT_PNG)
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
    s = normalize_time(t)
    if s.upper() == "B":
        return (-1e9, s)
    try:
        return (float(s), s)
    except Exception:
        return (1e9, s)


def time_value(t):
    s = normalize_time(t)
    if s.upper() == "B":
        return float("-inf")
    try:
        return float(s)
    except Exception:
        return float("inf")


def normalize_time(t):
    s = str(t).strip()
    if not s:
        return s
    if s.upper() == "B":
        return "B"
    try:
        v = float(s)
        if np.isfinite(v) and abs(v - int(v)) < 1e-9:
            return str(int(v))
        return str(v)
    except Exception:
        return s


def is_valid_time_label(t):
    s = normalize_time(t)
    if not s:
        return False
    if s.upper() == "B":
        return True
    try:
        v = float(s)
        return bool(np.isfinite(v))
    except Exception:
        return False


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    real_cache = np.load(args.real_npz, allow_pickle=True)
    real_keys_raw = real_cache["keys"]
    real_emb = real_cache["emb"].astype(float)
    mu, comps = pca2_fit(real_emb)
    real_xy_full = pca2_transform(real_emb, mu, comps)

    real_keys = []
    real_xy_rows = []
    for i in range(len(real_keys_raw)):
        subj = str(real_keys_raw[i][0])
        t = normalize_time(real_keys_raw[i][1])
        if not is_valid_time_label(t):
            continue
        real_keys.append((subj, t))
        real_xy_rows.append(real_xy_full[i])
    real_xy = np.asarray(real_xy_rows, dtype=float)

    endpoints_cache = np.load(args.endpoints_npz, allow_pickle=True)
    endpoints = endpoints_cache["endpoints"].astype(float)
    endpoints_subject = endpoints_cache["subject"].astype(str)
    endpoints_t1 = endpoints_cache["t1"].astype(str)
    end_xy_full = pca2_transform(endpoints, mu, comps)

    endpoint_keys = []
    end_xy_rows = []
    for i in range(len(endpoints_subject)):
        subj = str(endpoints_subject[i])
        t = normalize_time(endpoints_t1[i])
        if not is_valid_time_label(t):
            continue
        endpoint_keys.append((subj, t))
        end_xy_rows.append(end_xy_full[i])
    end_xy = np.asarray(end_xy_rows, dtype=float)

    start_pc1 = {real_keys[i]: float(real_xy[i, 0]) for i in range(len(real_keys))}
    end_pc1 = {endpoint_keys[i]: float(end_xy[i, 0]) for i in range(len(endpoint_keys))}

    # Next-real timepoint per subject (for "real → next real" arrow).
    # Use numeric time ordering on normalized time labels.
    subj_to_times = {}
    for s, t in start_pc1.keys():
        if not is_valid_time_label(t):
            continue
        subj_to_times.setdefault(s, set()).add(t)
    next_real = {}
    for s, times_set in subj_to_times.items():
        ordered_times = [t for t in sorted(times_set, key=time_value) if is_valid_time_label(t)]
        for i in range(len(ordered_times) - 1):
            next_real[(s, ordered_times[i])] = ordered_times[i + 1]

    common_keys = sorted(set(start_pc1.keys()) & set(end_pc1.keys()), key=lambda k: (k[0], time_key(k[1])))
    if not common_keys:
        raise SystemExit("No overlapping (subject,t_start) keys between real embeddings and endpoints.")

    subjects = sorted({s for (s, _t) in common_keys})
    timepoints = sorted({t for (_s, t) in common_keys}, key=time_key)

    colors = cm.viridis(np.linspace(0.0, 1.0, max(2, len(timepoints))))
    color_by_time = {t: colors[i] for i, t in enumerate(timepoints)}

    # Build a single ordered list: timepoint blocks, each with the same subject order.
    ordered = []
    for t in timepoints:
        for s in subjects:
            k = (s, t)
            if k in start_pc1 and k in end_pc1:
                ordered.append(k)

    y = np.arange(len(ordered), dtype=float)
    x0 = np.asarray([start_pc1[k] for k in ordered], dtype=float)
    x1 = np.asarray([end_pc1[k] for k in ordered], dtype=float)
    x_next = np.asarray(
        [start_pc1.get((k[0], next_real[k]), np.nan) if k in next_real else np.nan for k in ordered],
        dtype=float,
    )
    has_next = np.isfinite(x_next)

    plt.style.use("seaborn-v0_8-white")
    fig_h = max(6.0, 0.10 * len(ordered))
    fig, (ax, ax_avg) = plt.subplots(
        1,
        2,
        figsize=(16.0, fig_h),
        gridspec_kw={"width_ratios": [3.0, 1.6]},
    )

    for i, k in enumerate(ordered):
        t = k[1]
        c = color_by_time[t]
        # Real → next real (grey), if available.
        if has_next[i]:
            a2 = FancyArrowPatch(
                (x0[i], y[i]),
                (x_next[i], y[i]),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color="0.35",
                alpha=0.75,
                zorder=1,
            )
            ax.add_patch(a2)
        # Real → rollout endpoint (colored by timepoint).
        a = FancyArrowPatch(
            (x0[i], y[i]),
            (x1[i], y[i]),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=c,
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(a)

    avg_by_time = []
    blocks = []

    # Timepoint separators + block geometry
    start_idx = 0
    for t in timepoints:
        idx_in_block = [i for i, k in enumerate(ordered) if k[1] == t]
        n_in_block = len(idx_in_block)
        if n_in_block == 0:
            continue
        y0 = start_idx - 0.5
        y1 = start_idx + n_in_block - 0.5
        y_mid = start_idx + 0.5 * (n_in_block - 1)
        blocks.append((t, y0, y1, y_mid))

        # Light background bands to align blocks across subplots.
        band_alpha = 0.06 if (len(blocks) % 2 == 1) else 0.0
        if band_alpha > 0:
            ax.axhspan(y0, y1, color="0.0", alpha=band_alpha, zorder=0)

        if start_idx > 0:
            ax.axhline(start_idx - 0.5, color="0.9", linewidth=1.0, zorder=0)

        # Collect average shifts per timepoint block for the summary subplot.
        idx_next = [i for i in idx_in_block if has_next[i]]
        mean_dx_real = float(np.mean((x_next - x0)[idx_next])) if idx_next else float("nan")
        mean_dx_end = float(np.mean((x1 - x0)[idx_in_block]))
        avg_by_time.append((t, mean_dx_real, mean_dx_end))

        start_idx += n_in_block

    ax.set_yticks([])
    ax.set_xlabel("PC1 (start → endpoint)")
    ax.set_title("Gingivitis Metropolis rollouts: direction and magnitude of PC1 shift per start")
    ax.invert_yaxis()

    x_all = np.concatenate([x0, x1, x_next[np.isfinite(x_next)]])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    x_pad = 0.05 * (x_max - x_min + 1e-12)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    legend_handles = [
        FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="0.35", alpha=0.75),
        FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=cm.viridis(0.8), alpha=0.85),
    ]
    ax.legend(legend_handles, ["Real → next real", "Real → rollout endpoint"], frameon=False, loc="lower right")

    # Put timepoint labels on the outside-left of the main panel, aligned to block midpoints.
    ax_tp = ax.twinx()
    ax_tp.set_ylim(ax.get_ylim())
    y_mids = [y_mid for (_t, _y0, _y1, y_mid) in blocks]
    ax_tp.set_yticks(y_mids)
    ax_tp.set_yticklabels([str(t) for (t, _y0, _y1, _y_mid) in blocks])
    ax_tp.tick_params(axis="y", which="both", length=0, pad=8, labelcolor="black")
    ax_tp.yaxis.set_ticks_position("left")
    ax_tp.yaxis.set_label_position("left")
    ax_tp.spines["left"].set_position(("outward", 18))
    ax_tp.spines["left"].set_visible(False)
    ax_tp.spines["right"].set_visible(False)
    for tick_label in ax_tp.get_yticklabels():
        tick_label.set_fontsize(10)

    # Summary subplot: average arrows per timepoint block
    ax_avg.set_title("Mean shift")

    # Match timepoint bands between subplots and place arrows at the band midpoint.
    def _avg_for_time(t, which):
        for tt, a, b in avg_by_time:
            if tt == t:
                return a if which == "real" else b
        return float("nan")

    for i, (t, y0, y1, y_mid) in enumerate(blocks):
        c = color_by_time[t]
        mean_dx_real = _avg_for_time(t, "real")
        mean_dx_end = _avg_for_time(t, "end")

        band_alpha = 0.06 if (i % 2 == 0) else 0.0
        if band_alpha > 0:
            ax_avg.axhspan(y0, y1, color="0.0", alpha=band_alpha, zorder=0)
        if i > 0:
            ax_avg.axhline(y0 + 0.0, color="0.9", linewidth=1.0, zorder=0)

        # Start arrows at x=0 for readability; add a dot at the endpoint so near-zero shifts are visible.
        if np.isfinite(mean_dx_real):
            ax_avg.add_patch(
                FancyArrowPatch(
                    (0.0, y_mid),
                    (mean_dx_real, y_mid),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=2.0,
                    color="0.35",
                    alpha=0.9,
                    zorder=2,
                )
            )
            ax_avg.scatter([mean_dx_real], [y_mid], s=18, color="0.35", alpha=0.9, zorder=3)

        ax_avg.add_patch(
            FancyArrowPatch(
                (0.0, y_mid),
                (mean_dx_end, y_mid),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=2.0,
                color=c,
                alpha=0.9,
                zorder=2,
            )
        )
        ax_avg.scatter([mean_dx_end], [y_mid], s=18, color=c, alpha=0.9, zorder=3)

    ax_avg.set_yticks(y_mids)
    ax_avg.set_yticklabels([])
    ax_avg.set_xlabel("ΔPC1")
    ax_avg.invert_yaxis()
    ax_avg.axvline(0.0, color="0.8", linewidth=1.0)
    dx_all = [d for (_t, d, _e) in avg_by_time if np.isfinite(d)] + [e for (_t, _d, e) in avg_by_time]
    if dx_all:
        dx_min, dx_max = float(np.min(dx_all)), float(np.max(dx_all))
        pad = 0.08 * (dx_max - dx_min + 1e-12)
        ax_avg.set_xlim(dx_min - pad, dx_max + pad)
    ax_avg.set_ylim(ax.get_ylim())

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
