"""
DIABIMMUNE baselines notebook-style: binary taxa vs model embeddings.

Cells are structured with #%% for a user-friendly, linear workflow.
"""

#%% Imports & Config
import os, sys, csv
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils
from scripts.diabimmune.utils import (
    load_run_data,
    collect_micro_to_otus,
    load_microbiome_model as diab_load_model,
    build_sample_embeddings as diab_build_sample_embeddings,
)

SEED = 42
MIN_PREVALENCE = 5
D_TARGET = 100
RUN_CAPACITY_MATCHED = False
USE_GROUPED_CV = True


#%% Load run/sample maps and MicrobeAtlas OTUs
_, SRA_to_micro, _, micro_to_subject, micro_to_sample = load_run_data()
micro_to_otus = collect_micro_to_otus(SRA_to_micro, micro_to_subject)


#%% Compute eligible SRS (at least one resolvable embedding)
try:
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
except Exception:
    rename_map = None
try:
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}
except Exception:
    resolver = {}
import h5py
eligible_srs = set()
with h5py.File(shared_utils.PROKBERT_PATH) as f:
    emb = f['embeddings']
    for srs, otus in micro_to_otus.items():
        for oid in otus:
            key = resolver.get(oid, oid) if resolver else oid
            if key in emb:
                eligible_srs.add(srs)
                break
print('eligible SRS:', len(eligible_srs))


#%% Build binary presence/absence matrix (prevalence filtered)
sample_srs = [s for s in micro_to_otus if s in micro_to_sample and s in eligible_srs]
prev = Counter()
for srs in sample_srs:
    prev.update(set(micro_to_otus.get(srs, [])))
kept_otus = sorted([otu for otu, c in prev.items() if c >= MIN_PREVALENCE])
otu_index = {otu: i for i, otu in enumerate(kept_otus)}
X_bin = np.zeros((len(sample_srs), len(kept_otus)), dtype=np.float32)
for r, srs in enumerate(sample_srs):
    for otu in micro_to_otus.get(srs, []):
        j = otu_index.get(otu)
        if j is not None:
            X_bin[r, j] = 1.0
print('binary matrix:', X_bin.shape)


#%% Build model embeddings with text tokens
model, device = diab_load_model()
term_to_vec = shared_utils.load_term_embeddings()
run_to_terms = shared_utils.parse_run_terms()
srs_to_terms = shared_utils.build_srs_terms(SRA_to_micro, run_to_terms, shared_utils.MAPPED_PATH)
sample_embeddings, _ = diab_build_sample_embeddings(
    micro_to_otus, model, device, srs_to_terms=srs_to_terms, term_to_vec=term_to_vec, include_text=True
)
print('sample embeddings:', len(sample_embeddings))


#%% Load labels (samples, milk, hla)
samples_path = 'data/diabimmune/samples.csv'
milk_path = 'data/diabimmune/milk.csv'
hla_path = 'data/diabimmune/pregnancy_birth.csv'

samples_table = {}
with open(samples_path) as f:
    header = None
    for line in f:
        parts = line.strip().split(',')
        if not parts or len(parts) < 2:
            continue
        if header is None:
            header = parts
            header[0] = header[0].lstrip('\ufeff')
            continue
        row = dict(zip(header, parts))
        samples_table[row['sampleID']] = row

milk_labels = {}
with open(milk_path) as f:
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
        lab = row['milk_first_three_days']
        if lab:
            milk_labels[row['subjectID']] = 'mothers_breast_milk' if lab == 'mothers_breast_milk' else 'other'

hla_labels = {}
with open(hla_path) as f:
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
        lab = row['HLA_risk_class']
        if lab in {'2', '3'}:
            hla_labels[row['subjectID']] = lab


#%% Build per-task records (only SRS we can map and that are eligible)
def build_records(label_map):
    recs = []
    for srs in sample_srs:
        sample_info = micro_to_sample.get(srs, {})
        subject = micro_to_subject.get(srs)
        sample_id = sample_info.get('sample')
        if not subject or not sample_id:
            continue
        label = label_map.get(subject)
        if not label:
            continue
        age = None
        age_str = samples_table.get(sample_id, {}).get('age_at_collection', '').strip()
        if age_str:
            try:
                age = float(age_str)
            except ValueError:
                pass
        recs.append({'srs': srs, 'label': label, 'age': age, 'subject': subject})
    return recs

milk_records_all = build_records(milk_labels)
hla_records_all = build_records(hla_labels)
print('records: milk', len(milk_records_all), 'hla', len(hla_records_all))


#%% Downsample to balance classes and align to intersection (binary ∩ embeddings)
def balance_and_align(records):
    by_label = defaultdict(list)
    for r in records:
        if r['srs'] in sample_embeddings and r['srs'] in sample_srs:
            by_label[r['label']].append(r)
    counts = {k: len(v) for k, v in by_label.items()}
    if not counts:
        return [], [], np.array([])
    rng = np.random.default_rng(SEED)
    minority = min(counts.values())
    out = []
    for k, items in by_label.items():
        if len(items) > minority:
            idx = rng.choice(len(items), minority, replace=False)
            out.extend(items[i] for i in idx)
        else:
            out.extend(items)
    rng.shuffle(out)
    srs_to_row = {s: i for i, s in enumerate(sample_srs)}
    rows = [srs_to_row[r['srs']] for r in out]
    groups = np.array([r['subject'] for r in out])
    return out, rows, groups

milk_records, milk_idx, milk_groups = balance_and_align(milk_records_all)
hla_records, hla_idx, hla_groups = balance_and_align(hla_records_all)
print('aligned: milk', len(milk_records), 'hla', len(hla_records))


#%% Build X/y for each task
def build_xy(records, idx):
    labels = [r['label'] for r in records]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    Xb = X_bin[idx]
    Xe = np.stack([sample_embeddings[r['srs']].numpy() for r in records])
    return Xb, Xe, y

milk_Xb, milk_Xe, milk_y = build_xy(milk_records, milk_idx)
hla_Xb, hla_Xe, hla_y = build_xy(hla_records, hla_idx)


#%% Cross-validation helper (inline)
def run_cv(X, y, groups=None, scale=False, rf_constrained=False):
    # Splitter
    if groups is not None and USE_GROUPED_CV:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            splits = splitter.split(X, y, groups)
        except Exception:
            from sklearn.model_selection import GroupKFold
            splitter = GroupKFold(n_splits=5)
            splits = splitter.split(X, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = splitter.split(X, y)

    lr = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
    if scale:
        lr = make_pipeline(StandardScaler(), lr)
    rf = RandomForestClassifier(
        n_estimators=300 if not rf_constrained else 100,
        max_depth=None if not rf_constrained else 6,
        min_samples_leaf=1 if not rf_constrained else 20,
        max_leaf_nodes=None if not rf_constrained else 64,
        n_jobs=-1,
        random_state=SEED,
        class_weight='balanced_subsample',
    )
    results = {'LR': {'auc': [], 'acc': []}, 'RF': {'auc': [], 'acc': []}}
    for tr, te in splits:
        for name, model in [('LR', lr), ('RF', rf)]:
            model.fit(X[tr], y[tr])
            prob = model.predict_proba(X[te])
            auc = roc_auc_score(y[te], prob[:, 1])
            pred = model.predict(X[te])
            acc = accuracy_score(y[te], pred)
            results[name]['auc'].append(float(auc))
            results[name]['acc'].append(float(acc))
    return {k: {'auc_mean': float(np.mean(v['auc'])), 'auc_std': float(np.std(v['auc'])),
                'acc_mean': float(np.mean(v['acc'])), 'acc_std': float(np.std(v['acc']))}
            for k, v in results.items()}


#%% Run CV for binary vs embeddings (milk)
milk_bin = run_cv(milk_Xb, milk_y, groups=milk_groups, scale=False, rf_constrained=False)
milk_emb = run_cv(milk_Xe, milk_y, groups=milk_groups, scale=True, rf_constrained=False)
print(f"\n== MILK (n={len(milk_y)}) ==")
print(f"Split: {'StratifiedGroupKFold' if USE_GROUPED_CV else 'StratifiedKFold'}")
print(f"Binary d={milk_Xb.shape[1]}, Emb d={milk_Xe.shape[1]}")
for m in ['LR', 'RF']:
    b, e = milk_bin[m], milk_emb[m]
    print(f"{m}: bin AUC {b['auc_mean']:.3f}±{b['auc_std']:.3f}, emb AUC {e['auc_mean']:.3f}±{e['auc_std']:.3f}; bin ACC {b['acc_mean']:.3f}±{b['acc_std']:.3f}, emb ACC {e['acc_mean']:.3f}±{e['acc_std']:.3f}")


#%% Run CV for binary vs embeddings (hla)
hla_bin = run_cv(hla_Xb, hla_y, groups=hla_groups, scale=False, rf_constrained=False)
hla_emb = run_cv(hla_Xe, hla_y, groups=hla_groups, scale=True, rf_constrained=False)
print(f"\n== HLA (n={len(hla_y)}) ==")
print(f"Split: {'StratifiedGroupKFold' if USE_GROUPED_CV else 'StratifiedKFold'}")
print(f"Binary d={hla_Xb.shape[1]}, Emb d={hla_Xe.shape[1]}")
for m in ['LR', 'RF']:
    b, e = hla_bin[m], hla_emb[m]
    print(f"{m}: bin AUC {b['auc_mean']:.3f}±{b['auc_std']:.3f}, emb AUC {e['auc_mean']:.3f}±{e['auc_std']:.3f}; bin ACC {b['acc_mean']:.3f}±{b['acc_std']:.3f}, emb ACC {e['acc_mean']:.3f}±{e['acc_std']:.3f}")


#%% Plot AUC comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, task, res_bin, res_emb in [
    (axes[0], 'milk', milk_bin, milk_emb),
    (axes[1], 'hla', hla_bin, hla_emb)
]:
    labels = ['LR', 'RF']
    x = np.arange(len(labels))
    w = 0.35
    bin_means = [res_bin[m]['auc_mean'] for m in labels]
    bin_errs = [res_bin[m]['auc_std'] for m in labels]
    emb_means = [res_emb[m]['auc_mean'] for m in labels]
    emb_errs = [res_emb[m]['auc_std'] for m in labels]
    ax.bar(x - w/2, bin_means, w, yerr=bin_errs, capsize=4, label='Binary')
    ax.bar(x + w/2, emb_means, w, yerr=emb_errs, capsize=4, label='Embeddings')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('ROC AUC')
    ax.set_title(task)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.legend()
plt.tight_layout()
plt.show()


#%% Optional: capacity-matched (binary SVD d=100 vs embeddings d=100, constrained RF)
if RUN_CAPACITY_MATCHED:
    svd_milk = TruncatedSVD(n_components=D_TARGET, random_state=SEED)
    milk_Xb_red = svd_milk.fit_transform(milk_Xb)
    svd_hla = TruncatedSVD(n_components=D_TARGET, random_state=SEED)
    hla_Xb_red = svd_hla.fit_transform(hla_Xb)

    milk_bin_cap = run_cv(milk_Xb_red, milk_y, groups=milk_groups, scale=True, rf_constrained=True)
    milk_emb_cap = run_cv(milk_Xe, milk_y, groups=milk_groups, scale=True, rf_constrained=True)
    print(f"\n== Capacity-matched MILK (d={D_TARGET}) ==")
    for m in ['LR', 'RF']:
        b, e = milk_bin_cap[m], milk_emb_cap[m]
        print(f"{m}: bin AUC {b['auc_mean']:.3f}±{b['auc_std']:.3f}, emb AUC {e['auc_mean']:.3f}±{e['auc_std']:.3f}")

    hla_bin_cap = run_cv(hla_Xb_red, hla_y, groups=hla_groups, scale=True, rf_constrained=True)
    hla_emb_cap = run_cv(hla_Xe, hla_y, groups=hla_groups, scale=True, rf_constrained=True)
    print(f"\n== Capacity-matched HLA (d={D_TARGET}) ==")
    for m in ['LR', 'RF']:
        b, e = hla_bin_cap[m], hla_emb_cap[m]
        print(f"{m}: bin AUC {b['auc_mean']:.3f}±{b['auc_std']:.3f}, emb AUC {e['auc_mean']:.3f}±{e['auc_std']:.3f}")

#%% Standard Stratified CV (no grouping) for comparison
milk_bin_std = run_cv(milk_Xb, milk_y, groups=None, scale=False, rf_constrained=False)
milk_emb_std = run_cv(milk_Xe, milk_y, groups=None, scale=True, rf_constrained=False)
print(f"\n== MILK (standard StratifiedKFold, n={len(milk_y)}) ==")
print(f"Binary d={milk_Xb.shape[1]}, Emb d={milk_Xe.shape[1]}")
for m in ['LR', 'RF']:
    b, e = milk_bin_std[m], milk_emb_std[m]
    print(f"{m}: bin AUC {b['auc_mean']:.3f}±{b['auc_std']:.3f}, emb AUC {e['auc_mean']:.3f}±{e['auc_std']:.3f}; bin ACC {b['acc_mean']:.3f}±{b['acc_std']:.3f}, emb ACC {e['acc_mean']:.3f}±{e['acc_std']:.3f}")

hla_bin_std = run_cv(hla_Xb, hla_y, groups=None, scale=False, rf_constrained=False)
hla_emb_std = run_cv(hla_Xe, hla_y, groups=None, scale=True, rf_constrained=False)
print(f"\n== HLA (standard StratifiedKFold, n={len(hla_y)}) ==")
print(f"Binary d={hla_Xb.shape[1]}, Emb d={hla_Xe.shape[1]}")
for m in ['LR', 'RF']:
    b, e = hla_bin_std[m], hla_emb_std[m]
    print(f"{m}: bin AUC {b['auc_mean']:.3f}±{b['auc_std']:.3f}, emb AUC {e['auc_mean']:.3f}±{e['auc_std']:.3f}; bin ACC {b['acc_mean']:.3f}±{b['acc_std']:.3f}, emb ACC {e['acc_mean']:.3f}±{e['acc_std']:.3f}")

#%% Note on splits and memorization
print(
    """
Observation:
- Under standard StratifiedKFold (no grouping), binary one-hot taxa features show strong AUC/ACC.
- With subject-grouped CV (StratifiedGroupKFold/GroupKFold), both binary and embeddings drop toward chance.

Interpretation:
- Multiple timepoints per subject allow models to partially memorize subject-specific signatures when subjects are split across folds.
- Grouping all samples from the same subject into a single fold removes this leakage and yields a stricter, more realistic estimate.

Recommendation:
- Prefer grouped CV by subject for reporting; show non-grouped results as sensitivity only.
"""
)
