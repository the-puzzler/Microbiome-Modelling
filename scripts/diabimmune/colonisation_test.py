#!/usr/bin/env python3
"""
DIABIMMUNE: OTU colonisation prediction (T1 -> T2)

Parallels gingivitis/snowmelt colonisation experiments:
- Group by subject and time (age_at_collection)
- Identify colonizer OTUs (present at T2 but absent at T1) and non-colonizers
- For each example, augment T1 inputs with the target OTU and score with the
  pretrained model (no task-specific training)
- Evaluate ROC AUC/AP for distinguishing colonizers vs non-colonizers, with
  and without text tokens, and visualize
"""

import os
import sys
import csv
from collections import defaultdict
import random

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch
# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import (
    load_run_data,
)  # noqa: E402
from scripts.gingivitis.utils import plot_colonisation_summary  # noqa: E402

OUT_DIR = os.path.join('data', 'paper_figures', 'drop_col_figures')

# If True, append 4 zero-valued scratch tokens per SRS to the transformer input
USE_ZERO_SCRATCH_TOKENS = True
SCRATCH_TOKENS_PER_SRS = 16


#%% Parameters
N_COLONIZER_SAMPLES = 20000
N_NON_COLONIZER_SAMPLES = 20000


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


#%% Collect SRS -> OTUs via mapped MicrobeAtlas file
needed_srs = set()
for srs_list in subject_time_to_srs.values():
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
srs_to_terms = shared_utils.build_srs_terms(SRA_to_micro, run_to_terms, shared_utils.MAPPED_PATH)


#%% Collect colonizer examples (OTUs present at t2 but absent at t1)
print('Finding colonizer examples...')
timepoints_by_subject = defaultdict(dict)
for (subj, age), srs_list in subject_time_to_srs.items():
    timepoints_by_subject[subj][age] = srs_list

all_colonizer_examples = []
for subject, timepoints in tqdm(timepoints_by_subject.items(), desc='Subjects'):
    ages = list(timepoints.keys())
    if len(ages) < 2:
        continue
    for i in range(len(ages)):
        for j in range(len(ages)):
            if i == j:
                continue
            t1 = ages[i]
            t2 = ages[j]
            srs_t1 = timepoints[t1]
            srs_t2 = timepoints[t2]
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
                    'subject': subject,
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
    subject = ex['subject']
    t1 = ex['t1']
    t2 = ex['t2']
    srs_t1 = timepoints_by_subject[subject][t1]
    srs_t2 = timepoints_by_subject[subject][t2]
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
        'subject': subject,
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
        per_sample_scores_txt = []
        for srs in srs_t1:
            base_otus = micro_to_otus.get(srs, [])
            aug_otus = list(base_otus)
            if target not in aug_otus:
                aug_otus.append(target)
            # Baseline (no text; optionally with zero scratch tokens)
            if not USE_ZERO_SCRATCH_TOKENS:
                logits_map = shared_utils.score_otu_list(
                    aug_otus,
                    resolver=resolver,
                    model=model,
                    device=device,
                    emb_group=emb_group,
                )
            else:
                vecs = []
                keep = []
                for oid in aug_otus:
                    key_emb = resolver.get(oid, oid) if resolver else oid
                    if key_emb in emb_group:
                        vecs.append(torch.tensor(emb_group[key_emb][()], dtype=torch.float32, device=device))
                        keep.append(oid)
                if vecs:
                    x1 = torch.stack(vecs, dim=0).unsqueeze(0)
                    n1 = x1.shape[1]
                    with torch.no_grad():
                        h1 = model.input_projection_type1(x1)
                        if SCRATCH_TOKENS_PER_SRS > 0:
                            z = torch.zeros((1, SCRATCH_TOKENS_PER_SRS, shared_utils.D_MODEL), dtype=torch.float32, device=device)
                            h = torch.cat([h1, z], dim=1)
                            mask = torch.ones((1, n1 + SCRATCH_TOKENS_PER_SRS), dtype=torch.bool, device=device)
                        else:
                            h = h1
                            mask = torch.ones((1, n1), dtype=torch.bool, device=device)
                        h = model.transformer(h, src_key_padding_mask=~mask)
                        logits_all = model.output_projection(h).squeeze(-1).squeeze(0).cpu().numpy()
                    logits_map = dict(zip(keep, logits_all[:len(keep)]))
                else:
                    logits_map = {}
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
                per_sample_scores_txt.append(logits_map_txt[target])
        if per_sample_scores:
            target_scores.append(float(np.mean(per_sample_scores)))
            target_labels.append(1 if ex['is_colonizer'] else 0)
        if per_sample_scores_txt:
            target_scores_txt.append(float(np.mean(per_sample_scores_txt)))

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
os.makedirs(OUT_DIR, exist_ok=True)
plot_colonisation_summary(
    logits,
    y,
    title_prefix='DIABIMMUNE Colonisation',
    save_path=os.path.join(OUT_DIR, 'diabimmune_colonisation_density_roc_base.png'),
)
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
    plt.title('DIABIMMUNE Colonisation ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'diabimmune_colonisation_roc_base_vs_text.png'), dpi=300)
    plt.close()

# %%
