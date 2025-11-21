"""
DIABIMMUNE: Multilabel Colonisation (t1 -> t2) using binary taxa presence, matching the
gingivitis/snowmelt baseline structure with grouped 5-fold CV by subject.
"""

#%% Imports & config
import os, sys, csv
from collections import defaultdict
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils
from scripts.diabimmune.utils import (
    load_run_data,
)

SEED = 42
MIN_PREVALENCE = 10
# Optional subject-level subsampling to keep the DIABIMMUNE baseline tractable
# Set to None to use all subjects, or an integer to cap the number of subjects.
N_SUBJECTS_SUBSAMPLE = 25


#%% Load DIABIMMUNE mappings and group SRS by (subject, age)
run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()

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

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

subject_time_to_srs = defaultdict(list)
for srs, info in micro_to_sample.items():
    subject = info.get('subject')
    sample_id = info.get('sample')
    if not subject or not sample_id:
        continue
    age = safe_float(samples_table.get(sample_id, {}).get('age_at_collection', ''))
    if age is None:
        continue
    subject_time_to_srs[(subject, age)].append(srs)


#%% Optional subsampling of subjects to reduce dataset size
all_subjects = sorted({subj for (subj, _age) in subject_time_to_srs.keys()})
print(f"Total DIABIMMUNE subjects with age-resolved samples: {len(all_subjects)}")
if N_SUBJECTS_SUBSAMPLE is not None and len(all_subjects) > N_SUBJECTS_SUBSAMPLE:
    rng = np.random.default_rng(SEED)
    keep = set(rng.choice(all_subjects, N_SUBJECTS_SUBSAMPLE, replace=False))
    subject_time_to_srs = defaultdict(list,
        {k: v for k, v in subject_time_to_srs.items() if k[0] in keep}
    )
    print(f"Subsampling DIABIMMUNE subjects to n={N_SUBJECTS_SUBSAMPLE}; "
          f"subject-time groups retained: {len(subject_time_to_srs)}")
else:
    print(f"Using all DIABIMMUNE subjects; subject-time groups: {len(subject_time_to_srs)}")


#%% Collect SRS → OTUs and build presence matrix
needed_srs = {s for lst in subject_time_to_srs.values() for s in lst}
micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)

group_to_srs = {k: v for k, v in subject_time_to_srs.items()}
X, kept_otus, otu_index, keys, st_presence, key_to_row = shared_utils.build_presence_matrix(
    group_to_srs, micro_to_otus, min_prevalence=MIN_PREVALENCE
)


#%% Build multilabel colonisation dataset: all t1->t2 pairs within subject (t1 != t2)
X_rows, Y_ml, M_ml, groups_ml = shared_utils.build_multilabel_pairs(
    keys=keys,
    presence_by_key=st_presence,
    kept_otus=kept_otus,
    key_to_row=key_to_row,
    mode='colonisation',
    group_id_func=lambda k: k[0],  # subject
)

if X_rows.size == 0:
    print('No t1->t2 pairs available.')
else:
    X_ml = X[X_rows]
    print(f"Multilabel colonisation: n={len(X_ml)}, d={X_ml.shape[1]}, labels={Y_ml.shape[1]}")

    # Single random split (not grouped) diagnostic
    rng = np.random.default_rng(SEED)
    n_samples = X_ml.shape[0]
    idx = rng.permutation(n_samples)
    n_train = max(1, int(0.8 * n_samples))
    tr_idx = idx[:n_train]
    te_idx = idx[n_train:]
    tr_mask = np.zeros(n_samples, dtype=bool); tr_mask[tr_idx] = True
    te_mask = np.zeros(n_samples, dtype=bool); te_mask[te_idx] = True

    no_train_examples = 0
    single_class_train = 0
    for j in range(Y_ml.shape[1]):
        elig = M_ml[tr_mask, j]
        if not np.any(elig):
            no_train_examples += 1
            continue
        y_tr = Y_ml[tr_mask, j][elig]
        if np.unique(y_tr).size < 2:
            single_class_train += 1

    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=SEED))
    clf.fit(X_ml[tr_mask], Y_ml[tr_mask])
    prob = clf.predict_proba(X_ml[te_mask])
    per_label = []
    for j in range(Y_ml.shape[1]):
        mask = M_ml[te_mask, j]
        if not np.any(mask):
            continue
        y_true = Y_ml[te_mask, j][mask]
        if np.unique(y_true).size < 2:
            continue
        try:
            per_label.append(float(roc_auc_score(y_true, prob[:, j][mask])))
        except Exception:
            pass
    if per_label:
        macro = float(np.mean(per_label))
        total_labels = Y_ml.shape[1]
        print(f"Random split macro AUC: {macro:.3f} (labels with both classes: {len(per_label)})")
        print(f"Label diagnostics (train): no-train-examples={no_train_examples}/{total_labels}, single-class={single_class_train}/{total_labels}")
    else:
        print('No evaluable labels in the test split (all single-class after masking).')

# ---- Define models ----
lr_est = LogisticRegression(
    max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=SEED
)

rf_est = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=SEED
)

# ---- Grouped 5-fold CV by subject ----
shared_utils.eval_masked_ovr("LogReg (OvR) — Colonisation", lr_est, X_ml, Y_ml, M_ml, groups_ml, n_splits=5, seed=SEED)
shared_utils.eval_masked_ovr("RandForest (OvR) — Colonisation", rf_est, X_ml, Y_ml, M_ml, groups_ml, n_splits=5, seed=SEED)

# %%
