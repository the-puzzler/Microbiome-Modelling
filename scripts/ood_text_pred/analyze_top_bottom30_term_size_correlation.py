#!/usr/bin/env python3
"""Analyze term-level stability-vs-size correlations for top/bottom 30 terms.

Uses term lists from:
  data/ood_text_pred/term_enrichment_wilcoxon_top_bottom_notextabl.txt

Computes per-term Pearson/Spearman correlation between:
  - otu_richness
  - stability_score

Writes:
  - data/ood_text_pred/top_bottom30_term_size_correlation_per_term.tsv
  - data/ood_text_pred/top_bottom30_term_size_correlation_summary.txt
  - data/ood_text_pred/top_bottom30_term_sample_lookup.tsv
"""

import os
import sys
import csv

import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu


# Ensure project root on sys.path so `from scripts import utils` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


TOPTXT = 'data/ood_text_pred/term_enrichment_wilcoxon_top_bottom_notextabl.txt'
STAB = 'data/ood_text_pred/stability_vs_diversity_notextabl_max5k.tsv'
STAB_RAW = 'data/ood_text_pred/average_otu_logits_notextabl.tsv'
MAPPED = 'data/microbeatlas/samples-otus.97.mapped'
TERMS = 'data/microbeatlas/sample_terms_mapping_combined_dany_og_biome_tech.txt'

OUT_TSV = 'data/ood_text_pred/top_bottom30_term_size_correlation_per_term.tsv'
OUT_TXT = 'data/ood_text_pred/top_bottom30_term_size_correlation_summary.txt'
OUT_LOOKUP = 'data/ood_text_pred/top_bottom30_term_sample_lookup.tsv'

TOP_K = 30
MIN_N = 5
MIN_UNIQUE_X = 3


def parse_top_bottom_terms(path):
    with open(path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]

    top_terms = []
    bottom_terms = []
    mode = None

    for ln in lines:
        if ln.startswith('Top ') and 'enriched terms' in ln:
            mode = 'top'
            continue
        if ln.startswith('Top ') and 'depleted terms' in ln:
            mode = 'bottom'
            continue
        if not ln or ln.startswith('term\t'):
            continue

        term = ln.split('\t')[0].strip()
        if not term:
            continue
        if mode == 'top':
            top_terms.append(term)
        elif mode == 'bottom':
            bottom_terms.append(term)

    # preserve order, enforce exactly top/bottom 30 max
    top_terms = list(dict.fromkeys(top_terms))[:TOP_K]
    bottom_terms = list(dict.fromkeys(bottom_terms))[:TOP_K]
    return top_terms, bottom_terms


def load_sample_xy(path):
    s2xy = {}
    with open(path, 'r') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            srs = (row.get('sampleid') or '').strip()
            if not srs:
                continue
            try:
                x = float(row['otu_richness'])
                y = float(row['stability_score'])
            except Exception:
                continue
            s2xy[srs] = (x, y)
    return s2xy


def load_stability_from_raw(path):
    """Load sampleid -> stability from raw average-logit file."""
    s2y = {}
    with open(path, 'r') as f:
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


def load_richness_from_mapped(path):
    """Compute per-SRS observed OTU richness (unique 97_* IDs) from mapped file."""
    srs_to_otus = {}
    current_srs = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                header = line[1:].split()[0]
                parts = header.split('.')
                current_srs = parts[-1] if parts else header
                if current_srs and current_srs not in srs_to_otus:
                    srs_to_otus[current_srs] = set()
                continue
            if not current_srs:
                continue
            first_field = line.split()[0]
            toks = [t for t in first_field.split(';') if t]
            otu97 = next((t for t in toks if t.startswith('97_')), toks[-1] if toks else None)
            if otu97:
                srs_to_otus[current_srs].add(otu97)
    return {s: float(len(v)) for s, v in srs_to_otus.items() if v}


def build_sample_xy_manual():
    """Build sampleid -> (otu_richness, stability_score) directly from raw sources."""
    if not os.path.exists(STAB_RAW):
        raise FileNotFoundError(f'Missing raw stability table: {STAB_RAW}')
    if not os.path.exists(MAPPED):
        raise FileNotFoundError(f'Missing mapped file: {MAPPED}')

    s2y = load_stability_from_raw(STAB_RAW)
    s2x = load_richness_from_mapped(MAPPED)
    out = {}
    for srs, y in s2y.items():
        x = s2x.get(srs)
        if x is None:
            continue
        out[srs] = (x, y)
    return out


def build_term_xy_and_samples(s2xy):
    acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
    run_to_terms = shared_utils.parse_run_terms(TERMS)
    srs_to_terms = shared_utils.build_srs_terms(acc_to_srs, run_to_terms, shared_utils.MAPPED_PATH)

    term_x = {}
    term_y = {}
    term_samples = {}
    for srs, (x, y) in s2xy.items():
        terms = srs_to_terms.get(srs)
        if not terms:
            continue
        for t in terms:
            term_x.setdefault(t, []).append(x)
            term_y.setdefault(t, []).append(y)
            term_samples.setdefault(t, []).append((srs, x, y))
    return term_x, term_y, term_samples


def corr_stats(term, term_x, term_y):
    x = np.asarray(term_x.get(term, []), dtype=float)
    y = np.asarray(term_y.get(term, []), dtype=float)
    if len(x) < MIN_N or np.unique(x).size < MIN_UNIQUE_X:
        return None
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        'n': int(len(x)),
        'pearson_r': float(pr),
        'pearson_p': float(pp),
        'spearman_rho': float(sr),
        'spearman_p': float(sp),
        'mean_otu_richness': float(np.mean(x)),
        'mean_stability': float(np.mean(y)),
    }


def main():
    if not os.path.exists(TOPTXT):
        raise FileNotFoundError(f'Missing term list file: {TOPTXT}')
    if not os.path.exists(STAB_RAW):
        raise FileNotFoundError(f'Missing raw stability table: {STAB_RAW}')
    if not os.path.exists(MAPPED):
        raise FileNotFoundError(f'Missing mapped file: {MAPPED}')

    top_terms, bottom_terms = parse_top_bottom_terms(TOPTXT)
    print('Parsed terms:', 'top=', len(top_terms), 'bottom=', len(bottom_terms))
    print('Using raw sources:', STAB_RAW, 'and', MAPPED)
    s2xy = build_sample_xy_manual()
    term_x, term_y, term_samples = build_term_xy_and_samples(s2xy)

    rows = []
    top_vals = []
    bot_vals = []

    for term in top_terms:
        st = corr_stats(term, term_x, term_y)
        if st is None:
            continue
        rows.append(('top_enriched_30', term, st))
        top_vals.append(st['spearman_rho'])

    for term in bottom_terms:
        st = corr_stats(term, term_x, term_y)
        if st is None:
            continue
        rows.append(('bottom_depleted_30', term, st))
        bot_vals.append(st['spearman_rho'])

    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    with open(OUT_TSV, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow([
            'group',
            'term',
            'n_samples',
            'spearman_rho',
            'spearman_p',
            'pearson_r',
            'pearson_p',
            'mean_otu_richness',
            'mean_stability',
        ])
        for grp, term, st in rows:
            w.writerow([
                grp,
                term,
                st['n'],
                f"{st['spearman_rho']:.6f}",
                f"{st['spearman_p']:.6g}",
                f"{st['pearson_r']:.6f}",
                f"{st['pearson_p']:.6g}",
                f"{st['mean_otu_richness']:.6f}",
                f"{st['mean_stability']:.6f}",
            ])

    with open(OUT_TXT, 'w') as f:
        f.write('Top/Bottom 30 Term-Level Size-Stability Correlation Summary\n')
        f.write(f'Source list: {TOPTXT}\n')
        f.write(f'Raw stability table: {STAB_RAW}\n')
        f.write(f'Mapped OTU table: {MAPPED}\n')
        f.write(f'Top terms parsed: {len(top_terms)} | Bottom terms parsed: {len(bottom_terms)}\n')
        f.write(f'Usable top terms: {len(top_vals)}\n')
        f.write(f'Usable bottom terms: {len(bot_vals)}\n')

        if top_vals:
            f.write(
                f"Top Spearman rho mean={np.mean(top_vals):.6f} "
                f"median={np.median(top_vals):.6f}\n"
            )
        if bot_vals:
            f.write(
                f"Bottom Spearman rho mean={np.mean(bot_vals):.6f} "
                f"median={np.median(bot_vals):.6f}\n"
            )
        if top_vals and bot_vals:
            u, p = mannwhitneyu(top_vals, bot_vals, alternative='greater')
            f.write(f"Mann-Whitney U (top > bottom): U={float(u):.6f} p={float(p):.6g}\n")

    # Write direct term->sample lookup for top/bottom 30 terms.
    wanted_top = set(top_terms)
    wanted_bottom = set(bottom_terms)
    with open(OUT_LOOKUP, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['group', 'term', 'sampleid', 'otu_richness', 'stability_score'])

        for term in top_terms:
            for srs, x, y in term_samples.get(term, []):
                w.writerow(['top_enriched_30', term, srs, f'{x:.6f}', f'{y:.6f}'])

        for term in bottom_terms:
            for srs, x, y in term_samples.get(term, []):
                w.writerow(['bottom_depleted_30', term, srs, f'{x:.6f}', f'{y:.6f}'])

    print('Saved:', OUT_TSV)
    print('Saved:', OUT_TXT)
    print('Saved:', OUT_LOOKUP)


if __name__ == '__main__':
    main()
