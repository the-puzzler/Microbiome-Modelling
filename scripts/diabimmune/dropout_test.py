#!/usr/bin/env python3
"""
DIABIMMUNE: OTU dropout prediction (T1 -> T2)

Parallels gingivitis/snowmelt dropout implementations:
- Group by subject and time (age_at_collection)
- Score per-(subject,time) union-averaged OTU logits across SRS
- Build all-vs-all within-subject timepoint pairs
- Dropout label = 1 if OTU present at T1 and absent at T2
- Evaluate ROC AUC and AP using -logit as decision score (and 1 - sigmoid for AP)
- Compare no-text vs text-augmented scoring and visualize
"""

import os
import sys
import csv
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import (
    load_run_data,
    collect_micro_to_otus,
)  # noqa: E402
from scripts.gingivitis.utils import plot_dropout_summary  # noqa: E402


#%% Load DIABIMMUNE mappings: runs -> SRS, SRS -> subject/sample
run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
print('DIABIMMUNE runs:', len(run_rows), '| mapped to SRS:', len(SRA_to_micro))


#%% Parse samples.csv to get age_at_collection per sampleID
samples_csv = 'data/diabimmune/samples.csv'
samples_table = {}
with open(samples_csv) as f:
    header = None
    for line in f:
        parts = line.strip().split(',')
        if not parts:
            continue
        if header is None:
            header = parts
            header[0] = header[0].lstrip('\ufeff')
            continue
        row = dict(zip(header, parts))
        samples_table[row['sampleID']] = row
print('loaded samples records:', len(samples_table))


#%% Build grouping: (subject, age) -> list of SRS mapped to that sample/time
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


subject_time_to_srs = defaultdict(list)
subjects = set()
for srs, info in micro_to_sample.items():
    subject = info.get('subject')
    sample_id = info.get('sample')
    if not subject or not sample_id:
        continue
    rec = samples_table.get(sample_id, {})
    age = safe_float(rec.get('age_at_collection', ''))
    if age is None:
        continue
    subject_time_to_srs[(subject, age)].append(srs)
    subjects.add(subject)

print('subjects with age-resolved samples:', len(subjects))
print('subject-time groups:', len(subject_time_to_srs))


#%% Collect OTUs for needed SRS (use mapped MicrobeAtlas only)
needed_srs = set()
for srs_list in subject_time_to_srs.values():
    needed_srs.update(srs_list)
micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
print('SRS with OTUs:', len(micro_to_otus))


#%% Load model, build OTU key resolver (prefer B where both A/B exist)
model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}


#%% Score per-(subject, time): union-average logits across SRS in that group (no text)
st_otu_scores = {}
st_otu_presence = {}
with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
    emb_group = emb_file['embeddings']
    for (subject, age), srs_list in tqdm(list(subject_time_to_srs.items()), desc='Scoring subject-age groups'):
        sample_dicts = []
        for srs in srs_list:
            sdict = shared_utils.score_otus_for_srs(
                srs,
                micro_to_otus=micro_to_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
            )
            if sdict:
                sample_dicts.append(sdict)
        if not sample_dicts:
            st_otu_scores[(subject, age)] = {}
            st_otu_presence[(subject, age)] = set()
            continue
        avg = shared_utils.union_average_logits(sample_dicts)
        st_otu_scores[(subject, age)] = avg
        st_otu_presence[(subject, age)] = set(avg.keys())

print('computed group score maps:', len(st_otu_scores))


#%% Build within-subject all-vs-all timepoint pairs
def sort_times(times):
    # numeric times (age) ascending; robust to None by filtering earlier
    return sorted([t for t in times if t is not None])


pairs_by_subject = {}
total_pairs = 0
for subj in sorted(subjects):
    times = sort_times([t for (s, t) in st_otu_scores.keys() if s == subj])
    if len(times) < 2:
        continue
    pairs = [(t1, t2) for t1 in times for t2 in times if t1 != t2]
    if pairs:
        pairs_by_subject[subj] = pairs
        total_pairs += len(pairs)

print('subjects with ≥2 timepoints:', len(pairs_by_subject), '| total pairs:', total_pairs)


#%% Assemble dropout examples and evaluate (no text)
y_drop = []
y_score = []
for subj, pairs in pairs_by_subject.items():
    for t1, t2 in pairs:
        s1 = st_otu_scores.get((subj, t1), {})
        p2 = st_otu_presence.get((subj, t2), set())
        for oid, sc in s1.items():
            y_drop.append(1 if oid not in p2 else 0)
            y_score.append(sc)

y_drop = np.asarray(y_drop, dtype=np.int64)
y_score = np.asarray(y_score, dtype=np.float64)
print('total examples:', y_drop.size)

auc = roc_auc_score(y_drop, -y_score) if y_drop.size else float('nan')
ap = average_precision_score(y_drop, 1 - 1/(1 + np.exp(-y_score))) if y_drop.size else float('nan')
print(f'Overall dropout ROC AUC: {auc:.4f} | AP: {ap:.4f} | dropout rate: {y_drop.mean() if y_drop.size else float("nan"):.3f}')


#%% Text metadata integration — score with text tokens and compare
term_to_vec = shared_utils.load_term_embeddings()
run_to_terms = shared_utils.parse_run_terms()
srs_to_terms = shared_utils.build_srs_terms(SRA_to_micro, run_to_terms, shared_utils.MAPPED_PATH)

st_otu_scores_txt = {}
st_otu_presence_txt = {}
with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
    emb_group = emb_file['embeddings']
    for (subject, age), srs_list in tqdm(list(subject_time_to_srs.items()), desc='Scoring with text'):
        sample_dicts = []
        for srs in srs_list:
            sdict = shared_utils.score_otus_for_srs_with_text(
                srs,
                micro_to_otus=micro_to_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
                term_to_vec=term_to_vec,
                srs_to_terms=srs_to_terms,
            )
            if sdict:
                sample_dicts.append(sdict)
        if not sample_dicts:
            st_otu_scores_txt[(subject, age)] = {}
            st_otu_presence_txt[(subject, age)] = set()
            continue
        avg = shared_utils.union_average_logits(sample_dicts)
        st_otu_scores_txt[(subject, age)] = avg
        st_otu_presence_txt[(subject, age)] = set(avg.keys())

# Build examples for text case
y_drop_txt = []
y_score_txt = []
for subj, pairs in pairs_by_subject.items():
    for t1, t2 in pairs:
        s1 = st_otu_scores_txt.get((subj, t1), {})
        p2 = st_otu_presence_txt.get((subj, t2), set())
        for oid, sc in s1.items():
            y_drop_txt.append(1 if oid not in p2 else 0)
            y_score_txt.append(sc)

y_drop_txt = np.asarray(y_drop_txt, dtype=np.int64)
y_score_txt = np.asarray(y_score_txt, dtype=np.float64)
auc_txt = roc_auc_score(y_drop_txt, -y_score_txt) if y_drop_txt.size else float('nan')
ap_txt = average_precision_score(y_drop_txt, 1 - 1/(1 + np.exp(-y_score_txt))) if y_drop_txt.size else float('nan')

print(f'With text dropout ROC AUC: {auc_txt:.4f} | AP: {ap_txt:.4f} | dropout rate: {y_drop_txt.mean() if y_drop_txt.size else float("nan"):.3f}')
if np.isfinite(auc) and np.isfinite(auc_txt):
    print(f'Delta AUC (text - base): {auc_txt - auc:+.4f}')


#%% Plots: overlay ROC and shared-axis density histograms
import matplotlib.pyplot as plt

if y_drop.size and y_drop_txt.size:
    fpr_b, tpr_b, _ = roc_curve(y_drop, -y_score)
    fpr_t, tpr_t, _ = roc_curve(y_drop_txt, -y_score_txt)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_b, tpr_b, label=f'No text (AUC={auc:.3f})')
    plt.plot(fpr_t, tpr_t, label=f'With text (AUC={auc_txt:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('DIABIMMUNE Dropout ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# Shared-axis density comparison for logits
if y_drop.size:
    xmin = float(min(y_score.min(), y_score_txt.min())) if y_score_txt.size else float(y_score.min())
    xmax = float(max(y_score.max(), y_score_txt.max())) if y_score_txt.size else float(y_score.max())

    def _max_density(vals, labels, bins=40, rng=(xmin, xmax)):
        vals = np.asarray(vals, dtype=float)
        y = np.asarray(labels, dtype=int)
        m0 = y == 0
        m1 = y == 1
        dens_max = 0.0
        if np.any(m0):
            h0, _ = np.histogram(vals[m0], bins=bins, range=rng, density=True)
            dens_max = max(dens_max, float(h0.max()) if h0.size else 0.0)
        if np.any(m1):
            h1, _ = np.histogram(vals[m1], bins=bins, range=rng, density=True)
            dens_max = max(dens_max, float(h1.max()) if h1.size else 0.0)
        return dens_max

    ymax_base = _max_density(y_score, y_drop)
    ymax_text = _max_density(y_score_txt, y_drop_txt) if y_drop_txt.size else 0.0
    ylim = (0.0, max(ymax_base, ymax_text) * 1.05 if max(ymax_base, ymax_text) > 0 else None)

    plot_dropout_summary(y_score, y_drop, title_prefix='DIABIMMUNE', xlim=(xmin, xmax), ylim=None if ylim[1] is None else (ylim[0], ylim[1]))
    if y_drop_txt.size:
        plot_dropout_summary(y_score_txt, y_drop_txt, title_prefix='DIABIMMUNE (+ text)', xlim=(xmin, xmax), ylim=None if ylim[1] is None else (ylim[0], ylim[1]))


# %%

