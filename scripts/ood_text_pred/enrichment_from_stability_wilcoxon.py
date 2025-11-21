#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import csv

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Ensure project root on sys.path so `from scripts import utils` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


#%% Goal
# Load per-SRS stability scores (average OTU logits) and test whether samples
# annotated with a given metadata term have different stability distributions
# than samples without that term, using Mann–Whitney U (Wilcoxon rank-sum).
# Apply BH-FDR across terms and visualise the strongest enrichments/depletions.


#%% Paths and thresholds
IN_TSV = 'data/ood_text_pred/average_otu_logits.tsv'
OUT_TSV = 'data/ood_text_pred/term_enrichment_wilcoxon.tsv'
OUT_PNG = 'data/ood_text_pred/term_enrichment_wilcoxon_top_bottom.png'

# Minimum number of samples in which a term must appear
MIN_TERM_SAMPLES = 50


#%% Load SRS -> terms mapping from MicrobeAtlas resources
print('Building SRS -> terms mapping...')
acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
run_to_terms = shared_utils.parse_run_terms('data/microbeatlas/sample_terms_mapping_combined_dany_og_biome_tech.txt')
srs_to_terms = shared_utils.build_srs_terms(acc_to_srs, run_to_terms, shared_utils.MAPPED_PATH)
print('SRS with terms:', len(srs_to_terms))


#%% Load per-SRS stability scores (average logits)
if not os.path.exists(IN_TSV):
    raise FileNotFoundError(f'Input TSV not found: {IN_TSV}')

sample_scores = {}
with open(IN_TSV, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in tqdm(reader, desc='Loading stability scores'):
        srs = (row.get('sampleid') or '').strip()
        if not srs:
            continue
        try:
            avg_logit = float(row.get('average_logit'))
        except Exception:
            continue
        sample_scores[srs] = avg_logit

print('Loaded sample stability scores for', len(sample_scores), 'SRS')


#%% Build analysis universe: samples with both stability and at least one term
sample_ids = [s for s in sample_scores.keys() if s in srs_to_terms and srs_to_terms.get(s)]
scores = np.asarray([sample_scores[s] for s in sample_ids], dtype=np.float64)
print('Samples with stability+terms:', len(sample_ids))
if not sample_ids:
    raise SystemExit('No overlapping samples between stability scores and term annotations.')


#%% Per-term Wilcoxon enrichment
print('Running per-term Wilcoxon tests (min samples per term:', MIN_TERM_SAMPLES, ')...')
all_terms = sorted({t for s in sample_ids for t in srs_to_terms.get(s, [])})

term_stats = []
pvals = []
for term in tqdm(all_terms, desc='Terms'):
    term_mask = np.zeros(len(sample_ids), dtype=bool)
    for i, s in enumerate(sample_ids):
        if term in srs_to_terms.get(s, set()):
            term_mask[i] = True
    n_term = int(term_mask.sum())
    n_not = int((~term_mask).sum())
    if n_term < MIN_TERM_SAMPLES or n_not < MIN_TERM_SAMPLES:
        continue
    scores_term = scores[term_mask]
    scores_not = scores[~term_mask]
    if scores_term.size == 0 or scores_not.size == 0:
        continue
    try:
        stat, p = mannwhitneyu(scores_term, scores_not, alternative='two-sided')
    except Exception:
        continue
    mean_term = float(scores_term.mean())
    mean_not = float(scores_not.mean())
    effect = mean_term - mean_not
    term_stats.append((term, n_term, n_not, mean_term, mean_not, effect, stat, p))
    pvals.append(p)

print('Tested terms:', len(term_stats))
if not term_stats:
    raise SystemExit('No terms passed the MIN_TERM_SAMPLES filter for enrichment testing.')


#%% BH-FDR correction
pvals_arr = np.asarray(pvals, dtype=float)
m = len(pvals_arr)
order = np.argsort(pvals_arr)
ranked = pvals_arr[order]
bh = ranked * m / (np.arange(1, m + 1))
bh = np.minimum.accumulate(bh[::-1])[::-1]
qvals_sorted = np.minimum(bh, 1.0)
qvals = np.empty_like(qvals_sorted)
qvals[order] = qvals_sorted

print('Applied BH-FDR to', m, 'terms')


#%% Save enrichment table
os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
with open(OUT_TSV, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow([
        'term', 'n_with_term', 'n_without_term',
        'mean_with_term', 'mean_without_term',
        'effect_mean_diff', 'mannwhitney_U', 'pvalue', 'qvalue',
    ])
    for (term, n_term, n_not, mean_term, mean_not, effect, stat, p), q in zip(term_stats, qvals):
        w.writerow([
            term,
            str(n_term),
            str(n_not),
            f'{mean_term:.6f}',
            f'{mean_not:.6f}',
            f'{effect:.6f}',
            f'{stat:.6g}',
            f'{p:.6g}',
            f'{q:.6g}',
        ])
print('Saved Wilcoxon enrichment TSV:', OUT_TSV)


#%% Visualise strongest enrichments/depletions
effects = np.array([t[5] for t in term_stats], dtype=float)
qvals_arr = qvals

# Positive effect = higher stability when term present; negative = depleted
sig_mask = qvals_arr < 0.05
if not np.any(sig_mask):
    print('No terms significant at FDR < 0.05; skipping plot.')
else:
    sig_indices = np.where(sig_mask)[0]
    # Sort significant terms by effect size
    sig_sorted_pos = sig_indices[np.argsort(effects[sig_indices])[::-1]]  # largest positive
    sig_sorted_neg = sig_indices[np.argsort(effects[sig_indices])]        # most negative

    TOP_K = 30
    top_pos = sig_sorted_pos[:TOP_K]
    top_neg = sig_sorted_neg[:TOP_K]

    # Build plotting lists
    terms_pos = [term_stats[i][0] for i in top_pos]
    eff_pos = effects[top_pos]
    terms_neg = [term_stats[i][0] for i in top_neg]
    eff_neg = effects[top_neg]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    ax = axes[0]
    ax.barh(range(len(terms_pos)), eff_pos[::-1], color='#1f77b4')
    ax.set_yticks(range(len(terms_pos)))
    ax.set_yticklabels(terms_pos[::-1])
    ax.set_xlabel('Mean stability difference (with term - without term)')
    ax.set_title(f'Top {len(terms_pos)} enriched terms (q < 0.05)')

    ax = axes[1]
    ax.barh(range(len(terms_neg)), eff_neg[::-1], color='#d62728')
    ax.set_yticks(range(len(terms_neg)))
    ax.set_yticklabels(terms_neg[::-1])
    ax.set_xlabel('Mean stability difference (with term - without term)')
    ax.set_title(f'Top {len(terms_neg)} depleted terms (q < 0.05)')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print('Saved Wilcoxon enrichment plot:', OUT_PNG)

# %%

