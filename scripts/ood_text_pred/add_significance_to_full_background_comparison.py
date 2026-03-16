#!/usr/bin/env python3
"""Add per-term significance columns to full-background comparison TSV.

Input:
  data/ood_text_pred/top_bottom30_corr_size_enrichment_vs_full_background.tsv

Outputs (overwrites input TSV with added columns and updates summary):
  - p/q for delta_rho vs background
  - p/q for delta_mean_size vs background
"""

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm


IN_TSV = Path('data/ood_text_pred/top_bottom30_corr_size_enrichment_vs_full_background.tsv')
OUT_SUMMARY = Path('data/ood_text_pred/top_bottom30_corr_size_enrichment_vs_full_background_summary.txt')

STAB_RAW = Path('data/ood_text_pred/average_otu_logits_notextabl.tsv')
MAPPED = Path('data/microbeatlas/samples-otus.97.mapped')


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def load_full_background_sizes():
    # sample universe from raw stability file
    needed = set()
    with STAB_RAW.open() as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            srs = (row.get('sampleid') or '').strip()
            if srs:
                needed.add(srs)

    s2otu = {}
    current = None
    with MAPPED.open('r', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                header = line[1:].split()[0]
                parts = header.split('.')
                srs = parts[-1] if parts else header
                current = srs if srs in needed else None
                if current and current not in s2otu:
                    s2otu[current] = set()
                continue
            if not current:
                continue
            first = line.split()[0]
            toks = [t for t in first.split(';') if t]
            otu = next((t for t in toks if t.startswith('97_')), toks[-1] if toks else None)
            if otu:
                s2otu[current].add(otu)

    sizes = np.asarray([len(v) for v in s2otu.values() if len(v) > 0], dtype=float)
    if sizes.size == 0:
        raise RuntimeError('No background sizes recovered from mapped file.')
    return sizes


def main():
    if not IN_TSV.exists():
        raise FileNotFoundError(f'Missing input TSV: {IN_TSV}')
    if not STAB_RAW.exists():
        raise FileNotFoundError(f'Missing raw stability file: {STAB_RAW}')
    if not MAPPED.exists():
        raise FileNotFoundError(f'Missing mapped file: {MAPPED}')

    rows = []
    with IN_TSV.open() as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            rows.append(row)
    if not rows:
        raise RuntimeError('Input TSV has no rows.')

    # Background SD for mean-size delta significance.
    bg_sizes = load_full_background_sizes()
    bg_sd = float(np.std(bg_sizes, ddof=1))
    bg_n = int(bg_sizes.size)

    p_rho = []
    p_size = []

    # Compute per-term p-values (two-sided)
    for row in rows:
        n = int(float(row['n_points']))
        rho_t = float(row['rho_term'])
        rho_bg = float(row['rho_background'])
        d_size = float(row['delta_mean_size_vs_background'])

        # Rho delta significance via Fisher z against background rho.
        # z = (atanh(r_t)-atanh(r_bg)) * sqrt(n-3)
        r_t = max(min(rho_t, 0.999999), -0.999999)
        r_b = max(min(rho_bg, 0.999999), -0.999999)
        if n > 3:
            z_r = (math.atanh(r_t) - math.atanh(r_b)) * math.sqrt(max(n - 3, 1))
            p_r = 2.0 * norm.sf(abs(z_r))
        else:
            p_r = 1.0

        # Mean-size delta significance against background mean using SE(mean).
        if n > 1 and bg_sd > 0:
            se = bg_sd / math.sqrt(n)
            z_s = d_size / se
            p_s = 2.0 * norm.sf(abs(z_s))
        else:
            p_s = 1.0

        p_rho.append(p_r)
        p_size.append(p_s)

    q_rho = bh_fdr(p_rho)
    q_size = bh_fdr(p_size)

    # Rewrite TSV with added columns
    fieldnames = list(rows[0].keys()) + [
        'p_delta_rho',
        'q_delta_rho_bh',
        'p_delta_mean_size',
        'q_delta_mean_size_bh',
    ]
    with IN_TSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        for i, row in enumerate(rows):
            row = dict(row)
            row['p_delta_rho'] = f'{float(p_rho[i]):.6g}'
            row['q_delta_rho_bh'] = f'{float(q_rho[i]):.6g}'
            row['p_delta_mean_size'] = f'{float(p_size[i]):.6g}'
            row['q_delta_mean_size_bh'] = f'{float(q_size[i]):.6g}'
            w.writerow(row)

    # Update summary (append significance overview)
    n_sig_rho = int(np.sum(q_rho < 0.05))
    n_sig_size = int(np.sum(q_size < 0.05))
    with OUT_SUMMARY.open('a') as f:
        f.write('\n--- Added Significance Columns ---\n')
        f.write('Tests are two-sided analytic approximations per term.\n')
        f.write('delta_rho: Fisher-z against background rho with n_points term sample size.\n')
        f.write('delta_mean_size: z-test using background size SD and term n_points.\n')
        f.write(f'Background size n={bg_n}, sd={bg_sd:.6f}\n')
        f.write(f'Significant terms at q<0.05: delta_rho={n_sig_rho}, delta_mean_size={n_sig_size}\n')

    print('Updated TSV with significance columns:', IN_TSV)
    print('Appended summary:', OUT_SUMMARY)


if __name__ == '__main__':
    main()

