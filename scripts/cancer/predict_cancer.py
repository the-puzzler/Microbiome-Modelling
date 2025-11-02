#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import csv
import re
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

# Ensure project root is on sys.path so `from scripts import utils` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


#%% Paths
CANCER_RUN_TABLE = 'data/cancer/SraRunTableCancer.csv'


#%% Load cancer run table and build binary labels from env_medium
# Positive = Tumor (env_medium in {'Tumor','tumor'})
# Negative = Not-Tumor (env_medium in {'NAT','Normal','Normal-FA','N- fallop','N'})
# Controls are excluded (e.g., 'empty control','Paraffin Control','Extraction control').
rows = []
with open(CANCER_RUN_TABLE) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})

POSITIVE = {'Tumor', 'tumor'}
NEGATIVE = {'NAT', 'Normal', 'Normal-FA', 'N- fallop', 'N'}
EXCLUDE = {'empty control', 'Paraffin Control', 'Extraction control'}

run_to_label = {}
for r in rows:
    rid = r.get('Run', '')
    lab = r.get('env_medium', '')
    if not rid or not lab or lab in EXCLUDE:
        continue
    if lab in POSITIVE:
        run_to_label[rid] = 1
    elif lab in NEGATIVE:
        run_to_label[rid] = 0

print('Usable labeled runs:', len(run_to_label))


#%% Map Run -> SRS and resolve SRS labels (drop conflicts)
acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
srs_labels = {}
conflicts = 0
for rid, lab in run_to_label.items():
    if rid not in acc_to_srs:
        continue
    srs = acc_to_srs[rid]
    if srs in srs_labels and srs_labels[srs] != lab:
        conflicts += 1
        srs_labels.pop(srs, None)
    else:
        srs_labels[srs] = lab
print('Labeled SRS:', len(srs_labels), '| conflicts dropped:', conflicts)


#%% Collect SRS -> OTUs, load model, build resolver (prefer bacteria keys)
needed_srs = set(srs_labels.keys())
micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
print('SRS with OTUs:', len(micro_to_otus))

model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}


#%% Build per-SRS embeddings from the foundation model (no text)
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
print('Embeddings built for SRS:', len(sample_embeddings), '| missing OTU vecs:', missing_otus)


#%% Assemble dataset (X, y) for SRS with both label and embedding
X, y, keep_srs = [], [], []
for srs, lab in srs_labels.items():
    if srs in sample_embeddings:
        X.append(sample_embeddings[srs].numpy())
        y.append(lab)
        keep_srs.append(srs)
X = np.stack(X) if X else np.zeros((0, int(shared_utils.D_MODEL)))
y = np.asarray(y, dtype=np.int64)
print('Dataset:', X.shape[0], 'samples | pos_rate:', (float(y.mean()) if y.size else float('nan')))


#%% Build organ labels (strip status from env_local_scale, e.g., 'Breast (T)' -> 'Breast')
run_to_env = {r.get('Run', ''): r.get('env_local_scale', '') for r in rows if r.get('Run') and r.get('env_local_scale')}
srs_env_counts = defaultdict(Counter)
for rid, val in run_to_env.items():
    if rid in acc_to_srs:
        srs_env_counts[acc_to_srs[rid]][val] += 1

def to_organ(label: str) -> str:
    return re.sub(r'\s*\([^)]*\)', '', label).strip() if label else 'unknown'

srs_to_organ = {srs: (to_organ(max(cnt, key=cnt.get)) if cnt else 'unknown') for srs, cnt in srs_env_counts.items()}
organs_all = sorted({srs_to_organ.get(srs, 'unknown') for srs in keep_srs} - {'unknown'})
# Focus this demo on Breast only (others often too small here)
organs = ['Breast'] if 'Breast' in organs_all else organs_all
print('Organs evaluated:', organs)


#%% Per-organ demo: 5-fold Stratified CV (StandardScaler + Elastic Net LR), plots
if X.shape[0] == 0 or y.min() == y.max():
    raise SystemExit('Not enough labeled examples to evaluate.')

print('\nPer-organ evaluation (Elastic Net LR, standardized):')
for organ in organs:
    mask = np.array([srs_to_organ.get(s, 'unknown') == organ for s in keep_srs])
    if mask.sum() < 30:
        continue
    y_o = y[mask]
    if y_o.min() == y_o.max():
        continue
    X_o = X[mask]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = np.zeros_like(y_o, dtype=np.float32)
    y_pred = np.zeros_like(y_o, dtype=np.int64)
    for tr, te in skf.split(X_o, y_o):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                max_iter=4000,
                solver='saga',
                penalty='elasticnet',
                l1_ratio=0.5,
                C=1.0,
                class_weight='balanced',
            )),
        ])
        pipe.fit(X_o[tr], y_o[tr])
        prob = pipe.predict_proba(X_o[te])[:, 1]
        y_prob[te] = prob
        y_pred[te] = (prob >= 0.5).astype(np.int64)

    auc = roc_auc_score(y_o, y_prob)
    print(f'- {organ}: AUC={auc:.3f}, n={mask.sum()}')

    # Subplots: Confusion matrix and ROC
    cm = confusion_matrix(y_o, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not', 'Tumor'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title(f'{organ}: Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    fpr, tpr, _ = roc_curve(y_o, y_prob)
    axes[1].plot(fpr, tpr, label=f'AUC={auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{organ}: ROC Curve')
    axes[1].legend(loc='lower right')
    plt.tight_layout()
    plt.show()

#%% Notes
# - This demo builds a tumor vs not-tumor classifier using env_medium mapping
#   (Tumor vs NAT/Normal variants), excluding technical controls.
# - Embeddings come from the foundation model with OTU tokens only (no text),
#   pooled into per-SRS vectors.
# - We report per-organ performance via within-organ 5-fold Stratified CV and
#   show Confusion Matrix + ROC for each organ with both classes.
