#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
from collections import defaultdict
import random

import csv
import h5py
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.snowmelt.utils import load_snowmelt_metadata  # noqa: E402
from scripts.gingivitis.utils import plot_colonisation_summary  # standardized plot  # noqa: E402


#%% Parameters
N_COLONIZER_SAMPLES = 10000
N_NON_COLONIZER_SAMPLES = 10000


#%% Load snowmelt runs and group by (block, treatment, time)
run_meta, run_to_srs = load_snowmelt_metadata('data/snowmelt/snowmelt.csv')
print('parsed runs with metadata:', len(run_meta))

bt_time_to_srs = defaultdict(list)
for run, meta in run_meta.items():
    srs = run_to_srs.get(run)
    if not srs:
        continue
    key = (meta['block'], meta['treatment'], meta['time'])
    bt_time_to_srs[key].append(srs)
print('group keys (block,treatment,time):', len(bt_time_to_srs))


#%% Collect SRS -> OTUs via mapped MicrobeAtlas file
needed_srs = set()
for srs_list in bt_time_to_srs.values():
    needed_srs.update(srs_list)
micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
print('SRS with OTUs:', len(micro_to_otus))


#%% Load model and resolver (prefer bacteria keys)
model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}


#%% Load text embeddings + RunID->terms mapping and build SRS->terms via headers
term_to_vec = shared_utils.load_term_embeddings()
run_to_terms = shared_utils.parse_run_terms()
srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)


#%% Utilities
def time_order_key(t):
    order = {'A': 0, 'B': 1, 'C': 2}
    return order.get(t, 99)


#%% Collect colonizer examples (OTUs present at t2 but absent at t1) across (block,treatment)
print('Finding colonizer examples...')
bt_to_times = defaultdict(set)
for (block, trt, time) in bt_time_to_srs.keys():
    bt_to_times[(block, trt)].add(time)

all_colonizer_examples = []
for (block, trt), times_set in tqdm(bt_to_times.items(), desc='(block,treatment) pairs'):
    times_sorted = sorted(times_set, key=time_order_key)
    if len(times_sorted) < 2:
        continue
    for i in range(len(times_sorted)):
        for j in range(len(times_sorted)):
            if i == j:
                continue
            t1 = times_sorted[i]
            t2 = times_sorted[j]
            srs_t1 = bt_time_to_srs.get((block, trt, t1), [])
            srs_t2 = bt_time_to_srs.get((block, trt, t2), [])
            # Union OTUs at T1 and T2
            otus_t1 = set()
            for srs in srs_t1:
                otus_t1.update(micro_to_otus.get(srs, []))
            otus_t2 = set()
            for srs in srs_t2:
                otus_t2.update(micro_to_otus.get(srs, []))
            colonizers = otus_t2 - otus_t1
            for target in colonizers:
                all_colonizer_examples.append({
                    'block': block,
                    'treatment': trt,
                    't1': t1,
                    't2': t2,
                    'srs_t1': list(srs_t1),
                    'target_otu': target,
                    'is_colonizer': True,
                })

print('Total colonizer examples found:', len(all_colonizer_examples))


#%% Sample colonizer and non-colonizer examples
random.seed(42)
if len(all_colonizer_examples) > N_COLONIZER_SAMPLES:
    sampled_colonizers = random.sample(all_colonizer_examples, N_COLONIZER_SAMPLES)
else:
    sampled_colonizers = all_colonizer_examples
    print(f'Using all {len(sampled_colonizers)} colonizer examples')

print('Preparing non-colonizer examples...')
all_possible_otus = set()
for srs, otus in micro_to_otus.items():
    all_possible_otus.update(otus)

sampled_non_colonizers = []
for ex in tqdm(sampled_colonizers, desc='Sampling non-colonizers'):
    if len(sampled_non_colonizers) >= N_NON_COLONIZER_SAMPLES:
        break
    block, trt, t1, t2 = ex['block'], ex['treatment'], ex['t1'], ex['t2']
    srs_t1 = bt_time_to_srs.get((block, trt, t1), [])
    srs_t2 = bt_time_to_srs.get((block, trt, t2), [])
    otus_t1 = set()
    for srs in srs_t1:
        otus_t1.update(micro_to_otus.get(srs, []))
    otus_t2 = set()
    for srs in srs_t2:
        otus_t2.update(micro_to_otus.get(srs, []))
    absent_both = list(all_possible_otus - otus_t1 - otus_t2)
    if not absent_both:
        continue
    target = random.choice(absent_both)
    sampled_non_colonizers.append({
        'block': block,
        'treatment': trt,
        't1': t1,
        't2': t2,
        'srs_t1': list(srs_t1),
        'target_otu': target,
        'is_colonizer': False,
    })

print('Sampled non-colonizers:', len(sampled_non_colonizers))


#%% Score augmented tasks (baseline and with text)
examples = sampled_colonizers + sampled_non_colonizers
print('Total examples to process:', len(examples))

target_scores = []
target_scores_txt = []
target_labels = []

with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
    emb_group = emb_file['embeddings']
    for ex in tqdm(examples, desc='Scoring augmented T1 inputs'):
        srs_t1 = ex['srs_t1']
        target = ex['target_otu']
        per_sample_scores = []
        per_sample_scores_text = []
        for srs in srs_t1:
            base_otus = micro_to_otus.get(srs, [])
            aug_otus = list(base_otus)
            if target not in aug_otus:
                aug_otus.append(target)
            # Baseline (no text)
            logits_map = shared_utils.score_otu_list(
                aug_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
            )
            if target in logits_map:
                per_sample_scores.append(logits_map[target])
            # With text
            logits_map_txt = shared_utils.score_otu_list_with_text(
                srs,
                aug_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
                term_to_vec=term_to_vec,
                srs_to_terms=srs_to_terms,
            )
            if target in logits_map_txt:
                per_sample_scores_text.append(logits_map_txt[target])
        if per_sample_scores:
            target_scores.append(float(np.mean(per_sample_scores)))
            target_labels.append(1 if ex['is_colonizer'] else 0)
        if per_sample_scores_text:
            target_scores_txt.append(float(np.mean(per_sample_scores_text)))

print('Scored examples (baseline/text):', len(target_scores), '/', len(target_scores_txt))


#%% Evaluate colonisation prediction
y = np.array(target_labels, dtype=np.int64)
logits = np.array(target_scores, dtype=np.float32)
probs = 1 / (1 + np.exp(-logits))
logits_txt = np.array(target_scores_txt, dtype=np.float32)
probs_txt = 1 / (1 + np.exp(-logits_txt)) if logits_txt.size else np.array([])

if y.size and (y.min() != y.max()):
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    print(f'Colonisation (no text) — AUC: {auc:.4f} | AP: {ap:.4f} | pos_rate: {y.mean():.3f}')
    if probs_txt.size:
        auc_txt = roc_auc_score(y, probs_txt)
        ap_txt = average_precision_score(y, probs_txt)
        print(f'Colonisation (+ text) — AUC: {auc_txt:.4f} | AP: {ap_txt:.4f} | ΔAUC: {auc_txt - auc:+.4f}')
else:
    print('Insufficient class variation for AUC/AP.')


#%% Visualize
plot_colonisation_summary(logits, y, title_prefix='Snowmelt Colonisation')
if probs_txt.size:
    fpr_b, tpr_b, _ = roc_curve(y, probs)
    fpr_t, tpr_t, _ = roc_curve(y, probs_txt)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_b, tpr_b, label=f'No text (AUC={auc:.3f})')
    plt.plot(fpr_t, tpr_t, label=f'With text (AUC={auc_txt:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Snowmelt Colonisation ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# %%

