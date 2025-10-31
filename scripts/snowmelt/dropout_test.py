#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import h5py
import torch

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.snowmelt.utils import load_snowmelt_metadata  # noqa: E402

#%%

run_meta, run_to_srs = load_snowmelt_metadata('data/snowmelt/snowmelt.csv')
print('parsed runs with metadata:', len(run_meta))

# Build grouping: (block, treatment, time) -> list of SRS
bt_time_to_srs = {}
for run, meta in run_meta.items():
    srs = run_to_srs.get(run)
    if not srs:
        continue
    key = (meta['block'], meta['treatment'], meta['time'])
    bt_time_to_srs.setdefault(key, []).append(srs)

print('group keys (block,treatment,time):', len(bt_time_to_srs))

# Collect OTUs for all needed SRS
needed_srs = set()
for srs_list in bt_time_to_srs.values():
    needed_srs.update(srs_list)
micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
print('SRS with OTUs:', len(micro_to_otus))

# Load model + resolver
model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}


# Build per (block,treatment,time): union-average OTU logits across SRS
bt_time_logits = {}
with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
    emb_group = emb_file['embeddings']
    for key, srs_list in bt_time_to_srs.items():
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
            continue
        bt_time_logits[key] = shared_utils.union_average_logits(sample_dicts)

print('groups with averaged logits:', len(bt_time_logits))

# Build within-(block,treatment) all-vs-all time pairs and assemble dropout examples
from sklearn.metrics import roc_auc_score, average_precision_score

y_true_all = []
y_score_all = []

# Group keys by (block,treatment)
bt_to_times = {}
for (block, trt, time) in bt_time_logits.keys():
    bt_to_times.setdefault((block, trt), []).append(time)

def time_order_key(t):
    order = {'A': 0, 'B': 1, 'C': 2}
    return order.get(t, 99)

for bt, times_list in bt_to_times.items():
    times_sorted = sorted(set(times_list), key=time_order_key)
    if len(times_sorted) < 2:
        continue
    # all-vs-all pairs
    for t1 in times_sorted:
        for t2 in times_sorted:
            if t1 == t2:
                continue
            key1 = (bt[0], bt[1], t1)
            key2 = (bt[0], bt[1], t2)
            if key1 not in bt_time_logits or key2 not in bt_time_logits:
                continue
            logits_t1 = bt_time_logits[key1]
            present_t2 = set(bt_time_logits[key2].keys())
            for oid, score in logits_t1.items():
                y_true_all.append(0 if oid in present_t2 else 1)  # dropout=1
                y_score_all.append(score)

y_true_all = np.array(y_true_all, dtype=np.int64)
y_score_all = np.array(y_score_all, dtype=np.float32)
print('total examples:', len(y_true_all))

auc = roc_auc_score(y_true_all, -y_score_all)  # higher score -> persistence, so dropout uses -score
ap = average_precision_score(y_true_all, 1 / (1 + np.exp(-y_score_all)))
print(f'Overall dropout ROC AUC: {auc:.4f} | AP: {ap:.4f} | dropout rate: {y_true_all.mean():.3f}')


# %%
