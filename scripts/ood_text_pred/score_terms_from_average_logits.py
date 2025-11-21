#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

# Ensure project root on sys.path so `from scripts import utils` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


#%% Goal
# Read per-SRS average OTU logits TSV and compute per-text-annotation aggregates:
# For each term, collect the list of SRS average logits it appears in, then
# compute count (n), mean, std, and sum. Visualize top-50 and bottom-50 by mean
# with error bars (std). Linear scale by default.


#%% Paths
IN_TSV = 'data/ood_text_pred/average_otu_logits.tsv'
OUT_TSV = 'data/ood_text_pred/term_stats.tsv'
OUT_PNG = 'data/ood_text_pred/term_means_top_bottom.png'
OUT_ENRICH_TSV = 'data/ood_text_pred/term_enrichment_fisher.tsv'
# Minimum number of SRS occurrences required for a term to be included
MIN_OCCURRENCES = 10


#%% Load SRS -> terms mapping from MicrobeAtlas resources
print('Building SRS -> terms mapping...')
acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
run_to_terms = shared_utils.parse_run_terms('data/microbeatlas/sample_terms_mapping_combined_dany_og_biome_tech.txt')
srs_to_terms = shared_utils.build_srs_terms(acc_to_srs, run_to_terms, shared_utils.MAPPED_PATH)
print('SRS with terms:', len(srs_to_terms))


#%% Accumulate term scores from the per-SRS average logits TSV
if not os.path.exists(IN_TSV):
    raise FileNotFoundError(f'Input TSV not found: {IN_TSV}')

from collections import defaultdict

term_values = defaultdict(list)
sample_scores = {}
kept_rows = 0
skipped_no_terms = 0
with open(IN_TSV, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in tqdm(reader, desc='Accumulating term aggregates'):
        srs = (row.get('sampleid') or '').strip()
        try:
            avg_logit = float(row.get('average_logit'))
        except Exception:
            continue
        terms = srs_to_terms.get(srs)
        if not terms:
            skipped_no_terms += 1
            continue
        for t in terms:
            term_values[t].append(avg_logit)
        sample_scores[srs] = avg_logit
        kept_rows += 1

print('Rows used:', kept_rows, '| SRS without terms:', skipped_no_terms, '| unique terms:', len(term_values))


#%% Save full term score table (sorted desc)
os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
rows = []
for term, vals in term_values.items():
    n = len(vals)
    if n == 0:
        continue
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    stdv = float(arr.std(ddof=1)) if n > 1 else 0.0
    sm = float(arr.sum())
    rows.append((term, n, mean, stdv, sm))
print('Total terms before min-occurrence filter:', len(rows))
rows = [r for r in rows if r[1] >= MIN_OCCURRENCES]
print('Terms after filter (n >=', MIN_OCCURRENCES, '):', len(rows))
# Sort by mean descending for convenience in TSV
rows.sort(key=lambda r: r[2], reverse=True)
with open(OUT_TSV, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['term', 'n', 'mean', 'std', 'sum'])
    for term, n, mean, stdv, sm in rows:
        w.writerow([term, str(n), f'{mean:.6f}', f'{stdv:.6f}', f'{sm:.6f}'])
print('Saved term stats TSV:', OUT_TSV)


#%% Visualize top-N terms on a log scale
TOP_K = 50
if rows:
    # Top and bottom by mean
    top = rows[:TOP_K]
    bottom = sorted(rows, key=lambda r: r[2])[:TOP_K]

    plt.figure(figsize=(14, 10))
    # Top
    ax1 = plt.subplot(2, 1, 1)
    t_terms = [r[0] for r in top]
    t_means = [r[2] for r in top]
    t_stds = [r[3] for r in top]
    ax1.bar(range(len(t_means)), t_means, yerr=t_stds, ecolor='gray', alpha=0.9)
    ax1.set_title(f'Top {TOP_K} terms by mean average_logit (n ≥ {MIN_OCCURRENCES}, error bars: std)')
    ax1.set_ylabel('Mean average_logit')
    step = max(1, len(t_terms) // 20)
    ax1.set_xticks(range(0, len(t_terms), step))
    ax1.set_xticklabels([t_terms[i] for i in range(0, len(t_terms), step)], rotation=45, ha='right')

    # Bottom
    ax2 = plt.subplot(2, 1, 2)
    b_terms = [r[0] for r in bottom]
    b_means = [r[2] for r in bottom]
    b_stds = [r[3] for r in bottom]
    ax2.bar(range(len(b_means)), b_means, yerr=b_stds, ecolor='gray', alpha=0.9)
    ax2.set_title(f'Bottom {TOP_K} terms by mean average_logit (n ≥ {MIN_OCCURRENCES}, error bars: std)')
    ax2.set_xlabel('Term')
    ax2.set_ylabel('Mean average_logit')
    step2 = max(1, len(b_terms) // 20)
    ax2.set_xticks(range(0, len(b_terms), step2))
    ax2.set_xticklabels([b_terms[i] for i in range(0, len(b_terms), step2)], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print('Saved plot:', OUT_PNG)
else:
    print('No terms to plot.')

# %%
