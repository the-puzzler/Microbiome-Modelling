#!/usr/bin/env python3
#%% Imports and setup
import csv
import os
from collections import defaultdict

import h5py
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Ensure project root on sys.path
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import (
    load_gingivitis_run_data,
    collect_micro_to_otus,
    plot_colonisation_summary,
)  # noqa: E402

OUT_DIR = os.path.join('data', 'paper_figures', 'drop_col_figures')

# If True, append 4 zero-valued scratch tokens per SRS to the transformer input
USE_ZERO_SCRATCH_TOKENS = True
SCRATCH_TOKENS_PER_SRS = 16


#%% Parameters (simple, explicit)
N_COLONIZER_SAMPLES = 10000
N_NON_COLONIZER_SAMPLES = 10000


#%% Load gingivitis runs and map to MicrobeAtlas SRS
gingivitis_csv = 'data/gingivitis/gingiva.csv'
run_ids, SRA_to_micro = load_gingivitis_run_data(
    gingivitis_path=gingivitis_csv,
)
print(f"loaded runs: {len(run_ids)} | mapped to SRS: {len(SRA_to_micro)}")


#%% Collect SRS -> OTUs from mapped file
micro_to_otus = collect_micro_to_otus(SRA_to_micro)
if micro_to_otus:
    lens = [len(v) for v in micro_to_otus.values()]
    print(
        "SRS→OTUs prepared:",
        len(micro_to_otus),
        "samples | avg OTUs per SRS:",
        round(sum(lens) / max(1, len(lens)), 1),
    )
else:
    print("WARNING: No SRS→OTUs were found. Check mappings and mapped file.")


#%% Read gingivitis metadata to group by patient and timepoint
records = []
with open(gingivitis_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        run = row.get('Run', '').strip()
        subj = row.get('subject_code', '').strip()
        tcode = row.get('time_code', '').strip()
        if not run or not subj or not tcode:
            continue
        srs = SRA_to_micro.get(run)
        if not srs:
            continue
        records.append({'run': run, 'srs': srs, 'subject': subj, 'time': tcode})

patient_timepoint_samples = defaultdict(lambda: defaultdict(list))
for r in records:
    patient_timepoint_samples[r['subject']][r['time']].append(r['srs'])

multi_timepoint_patients = {p: tp for p, tp in patient_timepoint_samples.items() if len(tp) > 1}
print(f"Patients with multiple timepoints: {len(multi_timepoint_patients)}")


#%% Load model and rename map + resolver (prefer bacteria keys)
model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}


#%% Load text embeddings + RunID->terms mapping and build SRS->terms via headers
term_to_vec = shared_utils.load_term_embeddings()
run_to_terms = shared_utils.parse_run_terms()
run_to_srs = SRA_to_micro
srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)


## Scoring without text moved to shared utils: shared_utils.score_otu_list

#%% Collect colonizer examples (T2 present, T1 absent)
print("Finding all possible colonizer examples...")
all_colonizer_examples = []
for patient, timepoints in tqdm(multi_timepoint_patients.items()):
    timepoint_list = list(timepoints.keys())
    for i in range(len(timepoint_list)):
        for j in range(len(timepoint_list)):
            if i == j:
                continue
            t1 = timepoint_list[i]
            t2 = timepoint_list[j]
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
                    'patient': patient,
                    't1': t1,
                    't2': t2,
                    'srs_t1': list(srs_t1),
                    'target_otu': target,
                    'is_colonizer': True,
                })

print(f"Found {len(all_colonizer_examples)} colonizer examples")


#%% Sample colonizer examples
import random
random.seed(42)
if len(all_colonizer_examples) > N_COLONIZER_SAMPLES:
    sampled_colonizers = random.sample(all_colonizer_examples, N_COLONIZER_SAMPLES)
else:
    sampled_colonizers = all_colonizer_examples
    print(f"Using all {len(sampled_colonizers)} colonizer examples")


#%% Prepare non-colonizer examples from absent-both OTUs
print("Preparing non-colonizer examples...")
all_possible_otus = set()
for srs, otus in micro_to_otus.items():
    all_possible_otus.update(otus)

sampled_non_colonizers = []
for ex in tqdm(sampled_colonizers):
    if len(sampled_non_colonizers) >= N_NON_COLONIZER_SAMPLES:
        break
    patient = ex['patient']
    t1 = ex['t1']
    t2 = ex['t2']
    srs_t1 = patient_timepoint_samples[patient][t1]
    srs_t2 = patient_timepoint_samples[patient][t2]
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
        'patient': patient,
        't1': t1,
        't2': t2,
        'srs_t1': list(srs_t1),
        'target_otu': target,
        'is_colonizer': False,
    })

print(f"Sampled non-colonizers: {len(sampled_non_colonizers)}")


#%% Build and score augmented tasks
examples = sampled_colonizers + sampled_non_colonizers
print(f"Total examples to process: {len(examples)}")

target_scores = []
target_labels = []

with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
    emb_group = emb_file['embeddings']
    for ex in tqdm(examples, desc='Scoring augmented T1 inputs'):
        srs_t1 = ex['srs_t1']
        target = ex['target_otu']
        per_sample_scores = []
        per_sample_scores_txt = []
        for srs in srs_t1:
            # Original OTUs for this SRS
            base_otus = micro_to_otus.get(srs, [])
            aug_otus = list(base_otus)
            if target not in aug_otus:
                aug_otus.append(target)
            # No-text logits (optionally with zero scratch tokens)
            if not USE_ZERO_SCRATCH_TOKENS:
                logits_map = shared_utils.score_otu_list(
                    aug_otus,
                    resolver=resolver,
                    model=model,
                    device=device,
                    emb_group=emb_group,
                )
            else:
                # Implement score_otu_list + zero scratch tokens locally
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
            # With text tokens (shared utils)
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
            target_scores_txt = locals().setdefault('target_scores_txt', [])
            target_scores_txt.append(float(np.mean(per_sample_scores_txt)))

print(f"Scored examples: {len(target_scores)}")


#%% Evaluate colonization prediction
y = np.array(target_labels, dtype=np.int64)
logits = np.array(target_scores, dtype=np.float32)
probs = 1 / (1 + np.exp(-logits))

# With text (if computed)
logits_txt = np.array(locals().get('target_scores_txt', []), dtype=np.float32)
probs_txt = 1 / (1 + np.exp(-logits_txt)) if logits_txt.size else np.array([])

if y.size and (y.min() != y.max()):
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    print(f"Colonization (no text) — AUC: {auc:.4f} | AP: {ap:.4f} | pos_rate: {y.mean():.3f}")
    if probs_txt.size:
        auc_txt = roc_auc_score(y, probs_txt)
        ap_txt = average_precision_score(y, probs_txt)
        print(f"Colonization (+ text) — AUC: {auc_txt:.4f} | AP: {ap_txt:.4f} | ΔAUC: {auc_txt - auc:+.4f}")
else:
    print("Insufficient class variation for AUC/AP.")


#%% Visualize
os.makedirs(OUT_DIR, exist_ok=True)
plot_colonisation_summary(
    logits,
    y,
    title_prefix='Gingiva Colonisation',
    save_path=os.path.join(OUT_DIR, 'gingivitis_colonisation_density_roc_base.png'),
)
if logits_txt.size:
    # Overlay ROC curves
    fpr_b, tpr_b, _ = roc_curve(y, probs)
    fpr_t, tpr_t, _ = roc_curve(y, probs_txt)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.plot(fpr_b, tpr_b, label=f'No text (AUC={auc:.3f})')
    plt.plot(fpr_t, tpr_t, label=f'With text (AUC={auc_txt:.3f})')
    plt.plot([0,1],[0,1],'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gingiva Colonisation ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gingivitis_colonisation_roc_base_vs_text.png'), dpi=300)
    plt.close()

# %%
