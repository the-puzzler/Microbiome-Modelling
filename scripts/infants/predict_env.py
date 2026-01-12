#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import numpy as np
import re
import matplotlib.pyplot as plt

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.infants.utils import (
    load_infants_meta,
    load_infants_otus_tsv,
)
from scripts import utils as shared_utils

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# If True, append 8 zero-valued scratch tokens per sample when building embeddings
USE_ZERO_SCRATCH_TOKENS = True
SCRATCH_TOKENS_PER_SAMPLE = 16


#%% Load metadata
meta = load_infants_meta('data/infants/meta_withbirth.csv')
print('loaded meta entries:', len(meta))

#%% Load infants otus directly
sample_to_otus = load_infants_otus_tsv('data/infants/infants_otus.tsv')
print('otus rows:', len(sample_to_otus))

#%% Build per-SRS embeddings (frozen backbone, optional scratch tokens)
model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = shared_utils.build_otu_key_resolver(sample_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}

if not USE_ZERO_SCRATCH_TOKENS:
    sample_embeddings, _ = shared_utils.build_sample_embeddings(
        sample_to_otus,
        model,
        device,
        prokbert_path=shared_utils.PROKBERT_PATH,
        txt_emb=shared_utils.TXT_EMB,
        rename_map=rename_map,
        resolver=resolver,
    )
else:
    # Manually mirror build_sample_embeddings but append zero scratch tokens before pooling
    import h5py
    import torch
    sample_embeddings = {}
    missing_otus = 0
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        embedding_group = emb_file['embeddings']
        for sample_key, otu_list in sample_to_otus.items():
            otu_vectors = []
            for otu_id in otu_list:
                candidates = []
                if resolver and otu_id in resolver:
                    candidates.append(resolver[otu_id])
                candidates.append(otu_id)
                key_found = None
                for key in candidates:
                    if key in embedding_group:
                        key_found = key
                        break
                if key_found is not None:
                    vec = embedding_group[key_found][()]
                    otu_vectors.append(torch.tensor(vec, dtype=torch.float32, device=device))
                else:
                    missing_otus += 1
            if not otu_vectors:
                continue
            otu_tensor = torch.stack(otu_vectors, dim=0).unsqueeze(0)
            type2_tensor = torch.zeros((1, 0, shared_utils.TXT_EMB), dtype=torch.float32, device=device)
            with torch.no_grad():
                hidden_type1 = model.input_projection_type1(otu_tensor)
                hidden_type2 = model.input_projection_type2(type2_tensor)
                combined_hidden = torch.cat([hidden_type1, hidden_type2], dim=1)
                if SCRATCH_TOKENS_PER_SAMPLE > 0:
                    z = torch.zeros((1, SCRATCH_TOKENS_PER_SAMPLE, shared_utils.D_MODEL), dtype=torch.float32, device=device)
                    combined_hidden = torch.cat([combined_hidden, z], dim=1)
                mask = torch.ones((1, combined_hidden.shape[1]), dtype=torch.bool, device=device)
                hidden = model.transformer(combined_hidden, src_key_padding_mask=~mask)
                sample_vec = hidden.mean(dim=1).squeeze(0).cpu()
            sample_embeddings[sample_key] = sample_vec
    print('embeddings built with scratch tokens:', len(sample_embeddings), '| missing otu embeddings:', missing_otus)
print('embeddings built:', len(sample_embeddings))

#%% Assemble dataset X, y
X_list = []
y_list = []
kept = 0
for sid, label in meta.items():
    emb = sample_embeddings.get(sid)
    if emb is None:
        continue
    X_list.append(emb.numpy())
    y_list.append(label)
    kept += 1

print('usable samples:', kept)
if kept == 0:
    raise SystemExit('No usable samples (check mappings and embeddings).')

X = np.stack(X_list)
labels = np.array(y_list)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)
print('classes:', classes)

#%% 5x Stratified CV logistic regression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_reports = []
all_acc = []
all_auc_macro = []
# Collect per-class metrics across folds
per_class_f1 = {c: [] for c in []}
per_class_auc = {c: [] for c in []}

for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
    # Use selected configuration: lbfgs + L2, C=10, multinomial, no class weighting
    clf = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        C=10,
        class_weight=None,
        multi_class='multinomial',
        max_iter=2000,
    )
    clf.fit(X[tr], y[tr])
    y_pred = clf.predict(X[te])
    acc = accuracy_score(y[te], y_pred)
    all_acc.append(acc)
    # Macro ROC AUC (one-vs-rest) if >2 classes
    try:
        y_score = clf.predict_proba(X[te])
        auc_macro = roc_auc_score(y[te], y_score, multi_class='ovr', average='macro') if len(classes) > 2 else roc_auc_score(y[te], y_score[:,1])
        all_auc_macro.append(auc_macro)
    except Exception:
        pass
    # Per-class F1 from report dict
    rep_dict = classification_report(
        le.inverse_transform(y[te]),
        le.inverse_transform(y_pred),
        labels=classes,
        zero_division=0,
        output_dict=True,
    )
    # Initialise dicts on first fold
    if not per_class_f1:
        for c in classes:
            per_class_f1[c] = []
            per_class_auc[c] = []
    for idx, cname in enumerate(classes):
        f1 = rep_dict.get(cname, {}).get('f1-score')
        if f1 is not None:
            per_class_f1[cname].append(float(f1))
        try:
            # Compute per-class AUC (one-vs-rest) when both classes present
            y_true_bin = (y[te] == idx).astype(int)
            if y_true_bin.min() != y_true_bin.max():
                auc_c = roc_auc_score(y_true_bin, y_score[:, idx])
                per_class_auc[cname].append(float(auc_c))
        except Exception:
            pass
    rep = classification_report(le.inverse_transform(y[te]), le.inverse_transform(y_pred), labels=classes, zero_division=0)
    all_reports.append((fold, acc, rep))
    print(f'Fold {fold} accuracy: {acc:.4f}')

print('\nCV accuracy (mean±std): {:.4f} ± {:.4f}'.format(float(np.mean(all_acc)), float(np.std(all_acc))))
if all_auc_macro:
    print('CV macro ROC AUC (mean±std): {:.4f} ± {:.4f}'.format(float(np.mean(all_auc_macro)), float(np.std(all_auc_macro))))

print('\nClassification report (last fold):')
print(all_reports[-1][2])

#%% Visualisation: per-class boxplots of ROC AUC (blue) and F1 (yellow)
def _age_sort_key(label: str):
    # Expect formats like 'B(C)', 'M(V)', '4M(C)', '12M(V)', '3Y(C)', '5Y(V)'
    m = re.match(r"^([0-9]+M|[0-9]+Y|B|M)\((C|V)\)$", label)
    age = m.group(1) if m else label
    cat = m.group(2) if m else ''
    order = {'B': 0, 'M': 1, '4M': 2, '12M': 3, '3Y': 4, '5Y': 5}
    cat_order = {'C': 0, 'V': 1}
    return (order.get(age, 999), cat_order.get(cat, 9), label)

classes_sorted = sorted(classes, key=_age_sort_key)
auc_data = [per_class_auc.get(c, []) for c in classes_sorted]
f1_data = [per_class_f1.get(c, []) for c in classes_sorted]

fig, ax = plt.subplots(figsize=(12, 5))
positions_auc = np.arange(len(classes_sorted)) - 0.15
positions_f1 = np.arange(len(classes_sorted)) + 0.15

bp1 = ax.boxplot(auc_data, positions=positions_auc, widths=0.25, patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('#1f77b4')  # blue
    patch.set_alpha(0.6)
bp2 = ax.boxplot(f1_data, positions=positions_f1, widths=0.25, patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('#f1c40f')  # yellow
    patch.set_alpha(0.6)

ax.set_xticks(np.arange(len(classes_sorted)))
ax.set_xticklabels(classes_sorted, rotation=45, ha='right')
ax.set_ylabel('Score')
ax.set_title('Per-class ROC AUC (blue) and F1 (yellow) across CV folds')

# Horizontal lines at overall means
all_auc_vals = [v for lst in auc_data for v in lst]
all_f1_vals = [v for lst in f1_data for v in lst]
if all_auc_vals:
    ax.axhline(np.mean(all_auc_vals), color='#1f77b4', linestyle='--', linewidth=1, label='AUC mean')
if all_f1_vals:
    ax.axhline(np.mean(all_f1_vals), color='#f1c40f', linestyle='--', linewidth=1, label='F1 mean')

ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

# %%
