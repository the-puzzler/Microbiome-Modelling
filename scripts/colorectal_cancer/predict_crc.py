#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import csv
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Ensure project root on sys.path (future extensions may import shared utils)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils


#%% Paths (two SRA tables to combine by Run)
CRC_TBL_A = 'data/colorectal_cancer/SraRunTable (10).csv'  # has 'diagnosis', rich metadata
CRC_TBL_B = 'data/colorectal_cancer/SraRunTable (11).csv'  # has 'sample_type', neoplasia fields


#%% Fresh start: load, merge, and derive the broadest possible labels
def _nz(s):
    return (s or '').strip()

def load_by_run(path):
    data = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for r in reader:
            run = _nz(r.get('Run'))
            if not run:
                continue
            data[run] = {k.strip(): (_nz(v) if isinstance(v, str) else v) for k, v in r.items()}
    return data

by_run_a = load_by_run(CRC_TBL_A)
by_run_b = load_by_run(CRC_TBL_B)

runs_all = sorted(set(by_run_a) | set(by_run_b))
runs_both = sorted(set(by_run_a) & set(by_run_b))
print('Runs — total:', len(runs_all), '| A_only:', len(set(by_run_a) - set(by_run_b)), '| B_only:', len(set(by_run_b) - set(by_run_a)), '| both:', len(runs_both))

# Merge rows (prefer non-empty from A, then fill from B)
merged = {}
for run in runs_all:
    row = {}
    if run in by_run_a:
        row.update(by_run_a[run])
    if run in by_run_b:
        for k, v in by_run_b[run].items():
            if not row.get(k):
                row[k] = v
    merged[run] = row

# Subject id: prefer global BioSample, then Sample Name, sample_title, else Run
def subject_id(row):
    bios = _nz(row.get('BioSample'))
    if bios:
        return bios
    sname = _nz(row.get('Sample Name'))
    if sname:
        return sname
    stit = _nz(row.get('sample_title'))
    if stit:
        return stit
    return _nz(row.get('Run'))

# Helpers for yes/no
YES = {'yes', 'y', '1', 'true', 't'}
NO = {'no', 'n', '0', 'false', 'f'}
def yn(val):
    v = _nz(val).lower()
    if not v:
        return None
    if v in YES:
        return True
    if v in NO:
        return False
    return None

# Derive primary label using both files
# Priority: if diagnosis present → map to CRC/Adenoma/Control per paper.
# If diagnosis missing, use table (11) signals: Noeoplasia/Advanced_Adenoma/High_Risk_Adenoma,
# and sample_type suffix (_Polyp_Y vs _Polyp_N) to derive Adenoma vs Control.
def label_from_row(row):
    diag = _nz(row.get('diagnosis'))
    if diag and diag.upper() != 'NA':
        if diag.lower() == 'cancer':
            return 'CRC', 'diagnosis'
        if diag in {'Adenoma', 'adv Adenoma'}:
            return 'Adenoma', 'diagnosis'
        if diag in {'Normal', 'High Risk Normal'}:
            # Use B flags only to drop contradictions
            neo = yn(row.get('Noeoplasia'))
            adv = yn(row.get('Advanced_Adenoma'))
            high = yn(row.get('High_Risk_Adenoma'))
            st = _nz(row.get('sample_type'))
            polyp_yes = st.endswith('_Polyp_Y') if st else False
            if any(x is True for x in [neo, adv, high]) or polyp_yes:
                return None, 'contradiction_drop'
            return 'Control', 'diagnosis'
        return None, 'diagnosis_other_drop'

    # No diagnosis → derive from (11)
    st = _nz(row.get('sample_type'))
    if st:
        if st.endswith('_Polyp_Y'):
            return 'Adenoma', 'sample_type'
        if st.endswith('_Polyp_N'):
            # tentatively Control; still check explicit yes flags
            neo = yn(row.get('Noeoplasia'))
            adv = yn(row.get('Advanced_Adenoma'))
            high = yn(row.get('High_Risk_Adenoma'))
            if any(x is True for x in [neo, adv, high]):
                return 'Adenoma', 'flags_override'
            return 'Control', 'sample_type'
    # If sample_type not decisive, use yes/no flags
    neo = yn(row.get('Noeoplasia'))
    adv = yn(row.get('Advanced_Adenoma'))
    high = yn(row.get('High_Risk_Adenoma'))
    if any(x is True for x in [neo, adv, high]):
        return 'Adenoma', 'flags'
    if all(x is False for x in [neo, adv, high]) and any(x is not None for x in [neo, adv, high]):
        return 'Control', 'flags'
    return None, 'unlabeled'

# Assign labels at run level and then collapse to subjects (drop conflicts)
run_labels = {}
source_counts = Counter()
for run, row in merged.items():
    lab, src = label_from_row(row)
    if lab is not None:
        run_labels[run] = lab
    source_counts[src] += 1

print('\nRun-level labeling sources:', dict(source_counts))
print('Run-level labeled count:', len(run_labels), '/', len(merged))
print('Run-level label distribution:', dict(Counter(run_labels.values())))

runs_by_subject = defaultdict(list)
subject_to_label = {}
label_conflicts = 0
for run, row in merged.items():
    if run not in run_labels:
        continue
    sid = subject_id(row)
    lbl = run_labels[run]
    if sid in subject_to_label and subject_to_label[sid] != lbl:
        label_conflicts += 1
        # keep the first observed; drop conflicting run
        continue
    subject_to_label.setdefault(sid, lbl)
    runs_by_subject[sid].append(run)

duplicates = sum(1 for lst in runs_by_subject.values() if len(lst) > 1)
print(f"\nSubject-level aggregation: subjects={len(runs_by_subject)} | subjects with >1 run={duplicates} | label_conflicts={label_conflicts}")
print('Primary labels (subjects):', dict(Counter(subject_to_label.values())))

# Build common tasks and print sizes
def build_task(task):
    X_ids = []
    y = []
    if task == 'CRC_vs_Control':
        for sid, lbl in subject_to_label.items():
            if lbl in {'CRC', 'Control'}:
                X_ids.append(sid)
                y.append(1 if lbl == 'CRC' else 0)
    elif task == 'Adenoma_vs_Control':
        for sid, lbl in subject_to_label.items():
            if lbl in {'Adenoma', 'Control'}:
                X_ids.append(sid)
                y.append(1 if lbl == 'Adenoma' else 0)
    elif task == 'Adenoma_vs_CRC':
        for sid, lbl in subject_to_label.items():
            if lbl in {'Adenoma', 'CRC'}:
                X_ids.append(sid)
                y.append(1 if lbl == 'CRC' else 0)
    elif task == 'AdenomaCRC_vs_Control':
        for sid, lbl in subject_to_label.items():
            if lbl in {'Adenoma', 'CRC', 'Control'}:
                X_ids.append(sid)
                y.append(1 if lbl in {'Adenoma', 'CRC'} else 0)
    return X_ids, y

for task in ['CRC_vs_Control', 'Adenoma_vs_Control', 'Adenoma_vs_CRC', 'AdenomaCRC_vs_Control']:
    ids, y = build_task(task)
    if y:
        print(f"Task {task}: n={len(ids)} | positives={sum(y)} | negatives={len(y)-sum(y)}")
    else:
        print(f"Task {task}: n=0")


#%% Label selection criteria (documentation)
# We assign subject-level labels using the broadest available signals from both tables:
# - If 'diagnosis' is present (table 10):
#     Cancer → CRC; Adenoma/adv Adenoma → Adenoma; Normal/High Risk Normal → Control,
#     unless contradicted by neoplasia/adenoma flags or sample_type indicating polyp.
# - If 'diagnosis' is missing (table 11 only):
#     sample_type *_Polyp_Y → Adenoma; *_Polyp_N → Control (unless flags indicate Yes → Adenoma).
#     If sample_type not decisive, use flags: any Yes → Adenoma; all No (with any flag present) → Control.
# Conflicts across runs for the same subject are dropped, keeping the first observed label.
# Subject identity prefers BioSample, then Sample Name, then sample_title, else Run.


#%% Build subject embeddings and run Logistic Regression baselines
# One SRS per subject (prefer a Run that maps via MicrobeAtlas), OTU-only embeddings, 5-fold CV AUC.

# Map subject → chosen Run → SRS
acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
subject_to_run = {}
for sid, runs in runs_by_subject.items():
    chosen = None
    for run in runs:
        if run in acc_to_srs:
            chosen = run
            break
    if chosen is None and runs:
        chosen = runs[0]
    subject_to_run[sid] = chosen

subject_to_srs = {sid: acc_to_srs.get(run) for sid, run in subject_to_run.items()}
needed_srs = {s for s in subject_to_srs.values() if s}
print('Subjects with mapped SRS:', sum(1 for s in subject_to_srs.values() if s), '/', len(subject_to_srs))

# Collect OTUs, load foundation model, build resolver, compute per-SRS embeddings (no text)
micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}

sample_embeddings, missing_otus = shared_utils.build_sample_embeddings(
    micro_to_otus,
    model,
    device,
    rename_map=rename_map,
    resolver=resolver,
    srs_to_terms=None,
    term_to_vec=None,
    include_text=False,
)
print('Embeddings ready for SRS:', len(sample_embeddings), '| missing OTU vecs:', missing_otus)


def run_lr_task(task_name, ids, y):
    # Assemble X, y for subjects with available embeddings
    X_list, y_list = [], []
    for sid, yi in zip(ids, y):
        srs = subject_to_srs.get(sid)
        if srs in sample_embeddings:
            X_list.append(sample_embeddings[srs].numpy())
            y_list.append(int(yi))
    if not X_list or len(set(y_list)) < 2:
        print(f"[LR] {task_name}: not enough data (usable={len(X_list)})")
        return
    X = np.stack(X_list)
    y_arr = np.asarray(y_list, dtype=np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs_all = np.zeros_like(y_arr, dtype=np.float32)
    for tr, te in skf.split(X, y_arr):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=4000, solver='lbfgs', class_weight='balanced')
        clf.fit(Xtr, y_arr[tr])
        probs = clf.predict_proba(Xte)[:, 1]
        probs_all[te] = probs
    auc = roc_auc_score(y_arr, probs_all)
    pos = int(y_arr.sum())
    neg = int(len(y_arr) - pos)
    print(f"[LR] {task_name}: AUC={auc:.3f} | n={len(y_arr)} (pos={pos}, neg={neg})")


# Evaluate LR baselines on all tasks
for task in ['CRC_vs_Control', 'Adenoma_vs_Control', 'Adenoma_vs_CRC', 'AdenomaCRC_vs_Control']:
    ids, y = build_task(task)
    if y:
        run_lr_task(task, ids, y)

# %%
