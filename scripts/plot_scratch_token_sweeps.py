#!/usr/bin/env python3
import csv
import os

import matplotlib.pyplot as plt


DATA_PATH = os.path.join("data", "scratch_token_sweeps.tsv")
OUT_PATH = os.path.join("data", "scratch_token_sweeps.png")

MARKERS = {
    "no-text": "o",
    "text-trained": "s",
}

COLORS = {
    "zero": "#1f77b4",
    "real_text_plus_zero": "#2ca02c",
    "random": "#d62728",
}

LABELS = {
    "zero": "zero tokens",
    "real_text_plus_zero": "real text + zero tokens",
    "random": "random tokens",
}


def load_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["tokens"] = int(row["tokens"])
            row["auc"] = float(row["auc"])
            rows.append(row)
    return rows


def plot_panel(ax, rows, title, xlabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AUC")
    for condition in sorted({r["condition"] for r in rows}):
        for checkpoint in sorted({r["checkpoint"] for r in rows}):
            series = [
                r for r in rows
                if r["condition"] == condition and r["checkpoint"] == checkpoint
            ]
            if not series:
                continue
            series = sorted(series, key=lambda r: r["tokens"])
            xs = [r["tokens"] for r in series]
            ys = [r["auc"] for r in series]
            ax.plot(
                xs,
                ys,
                marker=MARKERS.get(checkpoint, "o"),
                color=COLORS.get(condition, "black"),
                label=f"{LABELS.get(condition, condition)} | {checkpoint}",
            )
    ax.grid(True, alpha=0.2)


def main():
    rows = load_rows(DATA_PATH)
    top_rows = [r for r in rows if r["condition"] != "random"]
    bottom_rows = [r for r in rows if r["condition"] == "random"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    plot_panel(axes[0], top_rows, "Zero tokens (with/without real text)", "# zero tokens added")
    plot_panel(axes[1], bottom_rows, "Random tokens", "# random tokens added")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.87),
            fontsize=9,
        )
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
