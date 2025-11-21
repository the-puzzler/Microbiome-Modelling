#!/usr/bin/env python3
"""Summarise top/bottom stability-enriched terms to a text file.

Reads the Wilcoxon enrichment results from
`data/ood_text_pred/term_enrichment_wilcoxon.tsv` and writes a concise
summary with the strongest enriched and depleted terms (by mean stability
difference) to a plain-text file.
"""

import os
import sys
import csv

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


IN_TSV = 'data/ood_text_pred/term_enrichment_wilcoxon.tsv'
OUT_TXT = 'data/ood_text_pred/term_enrichment_wilcoxon_top_bottom.txt'
TOP_K = 30


def main():
    if not os.path.exists(IN_TSV):
        raise FileNotFoundError(f'Enrichment TSV not found: {IN_TSV}')

    terms = []
    effects = []
    qvals = []
    n_with = []
    n_without = []

    with open(IN_TSV, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                effect = float(row['effect_mean_diff'])
                q = float(row['qvalue'])
                n_t = int(row['n_with_term'])
                n_nt = int(row['n_without_term'])
            except Exception:
                continue
            terms.append(row['term'])
            effects.append(effect)
            qvals.append(q)
            n_with.append(n_t)
            n_without.append(n_nt)

    if not terms:
        raise SystemExit('No valid rows in enrichment TSV.')

    effects = np.asarray(effects, dtype=float)
    qvals = np.asarray(qvals, dtype=float)
    n_with = np.asarray(n_with, dtype=int)
    n_without = np.asarray(n_without, dtype=int)

    # Prefer to restrict to significant terms when ranking; if none, rank all.
    sig_mask = qvals < 0.05
    if not np.any(sig_mask):
        print('No terms with q < 0.05; ranking all terms by effect size.')
        sig_mask = np.ones_like(qvals, dtype=bool)

    idx = np.where(sig_mask)[0]
    # Positive effect: higher stability with term present
    sorted_pos = idx[np.argsort(effects[idx])[::-1]]
    # Negative effect: lower stability with term present
    sorted_neg = idx[np.argsort(effects[idx])]

    top_pos = sorted_pos[:TOP_K]
    top_neg = sorted_neg[:TOP_K]

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w') as f:
        f.write(f"Top {len(top_pos)} enriched terms (higher stability with term)\n")
        f.write("term\teffect_mean_diff\tqvalue\tn_with_term\tn_without_term\n")
        for i in top_pos:
            f.write(
                f"{terms[i]}\t{effects[i]:+.6f}\t{qvals[i]:.3g}\t{n_with[i]}\t{n_without[i]}\n"
            )

        f.write("\nTop {0} depleted terms (lower stability with term)\n".format(len(top_neg)))
        f.write("term\teffect_mean_diff\tqvalue\tn_with_term\tn_without_term\n")
        for i in top_neg:
            f.write(
                f"{terms[i]}\t{effects[i]:+.6f}\t{qvals[i]:.3g}\t{n_with[i]}\t{n_without[i]}\n"
            )

    print('Wrote summary of top/bottom terms to', OUT_TXT)


if __name__ == '__main__':
    main()

