#%% Gingiva: Multilabel Colonisation (t1 -> t2)
"""
Single task (clean baseline):
- Input X: binary taxa presence at (subject, t1) [union across SRS]
- Target Y: for each kept OTU j, Yj=1 if absent at t1 and present at t2 (colonisation),
            Yj=0 if absent at both; OTUs present at t1 are masked when scoring.
- CV: Grouped 5-fold by subject, One-vs-Rest Logistic Regression, macro AUC over
       labels with both classes within masked examples.
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
from scripts.gingivitis.utils import (
    load_gingivitis_run_data,
    collect_micro_to_otus,
)

SEED = 42
MIN_PREVALENCE = 10


#%% Load runs → map to SRS → SRS→OTUs
gingivitis_csv = 'data/gingivitis/gingiva.csv'
run_ids, SRA_to_micro = load_gingivitis_run_data(
    gingivitis_path=gingivitis_csv,
    microbeatlas_path=shared_utils.MICROBEATLAS_SAMPLES,
)
micro_to_otus = collect_micro_to_otus(SRA_to_micro)


#%% Group SRS by (subject, time)
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
        if srs:
            records.append({'srs': srs, 'subject': subj, 'time': tcode})

subject_time_to_srs = defaultdict(list)
for r in records:
    subject_time_to_srs[(r['subject'], r['time'])].append(r['srs'])


#%% Build binary presence per (subject, time)
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

    # Single random split (not grouped) for symmetry with dropout baseline
    rng = np.random.default_rng(SEED)
    n_samples = X_ml.shape[0]
    idx = rng.permutation(n_samples)
    n_train = max(1, int(0.8 * n_samples))
    tr_idx = idx[:n_train]
    te_idx = idx[n_train:]
    tr_mask = np.zeros(n_samples, dtype=bool); tr_mask[tr_idx] = True
    te_mask = np.zeros(n_samples, dtype=bool); te_mask[te_idx] = True

    # Label diagnostics on training set
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


#%% Here we do not allow a subject to be both in train and test.
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

# ---- Run both ----
shared_utils.eval_masked_ovr("LogReg (OvR)", lr_est, X_ml, Y_ml, M_ml, groups_ml, n_splits=5, seed=SEED)
shared_utils.eval_masked_ovr("RandForest (OvR)", rf_est, X_ml, Y_ml, M_ml, groups_ml, n_splits=5, seed=SEED)



# %%
