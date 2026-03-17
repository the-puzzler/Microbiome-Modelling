#!/usr/bin/env python3
"""Compute abundance-based evenness and plot stability vs evenness.

Evenness metric:
  Pielou's J = H / ln(S)
where:
  H = Shannon entropy over OTU relative abundances
  S = observed OTU richness in the sample

Inputs:
  - data/ood_text_pred/average_otu_logits_notextabl.tsv
  - data/microbeatlas/samples-otus.97.mapped

Outputs:
  - data/ood_text_pred/stability_vs_evenness.tsv
  - data/ood_text_pred/stability_vs_evenness_colored_by_count.png
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import LogNorm


STAB_RAW = Path('data/ood_text_pred/average_otu_logits_notextabl.tsv')
MAPPED = Path('data/microbeatlas/samples-otus.97.mapped')

OUT_TSV = Path('data/ood_text_pred/stability_vs_evenness.tsv')
OUT_PNG_SIZE_COLOR = Path('data/ood_text_pred/stability_vs_evenness_colored_by_count.png')
OUT_PNG_SIZE_COLOR_SIGMOID = Path('data/ood_text_pred/stability_vs_evenness_colored_by_count_sigmoid.png')


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_stability(path):
    s2y = {}
    with path.open() as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            srs = (row.get('sampleid') or '').strip()
            if not srs:
                continue
            try:
                s2y[srs] = float(row['average_logit'])
            except Exception:
                continue
    return s2y


def finalize_sample(otu_to_n):
    if not otu_to_n:
        return None
    richness_s = len(otu_to_n)
    total_n = float(sum(otu_to_n.values()))
    if total_n <= 0:
        return None
    if richness_s == 1:
        return 0.0, float(richness_s)
    sum_n_lnn = float(sum(v * math.log(v) for v in otu_to_n.values() if v > 0))
    h = math.log(total_n) - (sum_n_lnn / total_n)
    denom = math.log(richness_s)
    if denom <= 0:
        return None
    j = h / denom
    if not np.isfinite(j):
        return None
    # Keep in [0, 1] for numeric stability.
    j = max(0.0, min(1.0, j))
    return j, float(richness_s)


def compute_evenness_for_needed_samples(mapped_path, needed):
    s2otu_to_n = {}
    current_srs = None

    with mapped_path.open('r', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue

            if line.startswith('>'):
                header = line[1:].split()[0]
                parts = header.split('.')
                srs = parts[-1] if parts else header
                current_srs = srs if srs in needed else None
                if current_srs and current_srs not in s2otu_to_n:
                    s2otu_to_n[current_srs] = {}
                continue

            if not current_srs:
                continue

            fields = line.split('\t')
            if len(fields) < 2:
                continue
            try:
                n = float(fields[1])
            except Exception:
                continue
            if n <= 0:
                continue
            first = fields[0]
            toks = [t for t in first.split(';') if t]
            otu = next((t for t in toks if t.startswith('97_')), toks[-1] if toks else None)
            if not otu:
                continue
            d = s2otu_to_n[current_srs]
            d[otu] = d.get(otu, 0.0) + n

    s2stats = {}
    for srs, otu_to_n in s2otu_to_n.items():
        out = finalize_sample(otu_to_n)
        if out is not None:
            s2stats[srs] = out

    return s2stats


def main():
    if not STAB_RAW.exists():
        raise FileNotFoundError(f'Missing stability file: {STAB_RAW}')
    if not MAPPED.exists():
        raise FileNotFoundError(f'Missing mapped file: {MAPPED}')

    s2stab = load_stability(STAB_RAW)
    s2stats = compute_evenness_for_needed_samples(MAPPED, set(s2stab.keys()))

    rows = []
    for srs, stab in s2stab.items():
        stats = s2stats.get(srs)
        if stats is None:
            continue
        ev, richness = stats
        rows.append((srs, stab, ev, richness))

    if not rows:
        raise RuntimeError('No matched samples with valid evenness and stability.')

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['sampleid', 'stability_score', 'evenness_pielou', 'count'])
        for srs, stab, ev, richness in rows:
            w.writerow([srs, f'{stab:.6f}', f'{ev:.6f}', f'{richness:.0f}'])

    x = np.asarray([r[1] for r in rows], dtype=float)  # stability
    y = np.asarray([r[2] for r in rows], dtype=float)  # evenness
    richness = np.asarray([r[3] for r in rows], dtype=float)  # OTU richness

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    # Full-range color plot with a single continuous color scale (no count cap).
    fig2, ax2 = plt.subplots(figsize=(7.2, 5.6))
    sc = ax2.scatter(
        x,
        y,
        c=richness,
        cmap='cividis',
        norm=LogNorm(vmin=max(1.0, float(np.min(richness))), vmax=float(np.max(richness))),
        s=7,
        alpha=0.25,
        edgecolors='none',
    )
    ax2.grid(True, alpha=0.25)
    ax2.set_xlabel('Stability score (average OTU logit)')
    ax2.set_ylabel("Evenness (Pielou's J)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title('Stability vs Evenness')
    cb = fig2.colorbar(sc, ax=ax2, pad=0.01)
    cb.set_label('Count (log scale)')
    plt.tight_layout()
    plt.savefig(OUT_PNG_SIZE_COLOR, dpi=350)

    # Sigmoid(stability) variant
    x_sig = sigmoid(x)
    fig3, ax3 = plt.subplots(figsize=(7.2, 5.6))
    sc3 = ax3.scatter(
        x_sig,
        y,
        c=richness,
        cmap='cividis',
        norm=LogNorm(vmin=max(1.0, float(np.min(richness))), vmax=float(np.max(richness))),
        s=7,
        alpha=0.25,
        edgecolors='none',
    )
    ax3.grid(True, alpha=0.25)
    ax3.set_xlabel('Sigmoid(stability score)')
    ax3.set_ylabel("Evenness (Pielou's J)")
    ax3.set_ylim(0, 1.02)
    ax3.set_xlim(0, 1)
    ax3.set_title(f'All Samples: Sigmoid(stability) vs Evenness, colored by count (n={len(rows)})')
    cb3 = fig3.colorbar(sc3, ax=ax3, pad=0.01)
    cb3.set_label('Count (log scale)')
    plt.tight_layout()
    plt.savefig(OUT_PNG_SIZE_COLOR_SIGMOID, dpi=350)

    print(f'Saved TSV: {OUT_TSV} | rows={len(rows)}')
    print(f'Saved figure: {OUT_PNG_SIZE_COLOR}')
    print(f'Saved figure: {OUT_PNG_SIZE_COLOR_SIGMOID}')
    print(f'Pearson r={pr:.6f} p={pp:.6g}')
    print(f'Spearman rho={sr:.6f} p={sp:.6g}')


if __name__ == '__main__':
    main()
