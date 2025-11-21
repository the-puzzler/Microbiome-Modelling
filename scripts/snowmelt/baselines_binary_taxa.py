#%% Snowmelt: Multilabel Dropout (t1 -> t2)
"""
Single task (clean baseline):
- Input X: binary taxa presence at (block, treatment, t1) [union across SRS]
- Target Y: for each kept OTU j, Yj=1 if present at t1 and absent at t2 (dropout),
            Yj=0 if present at both; OTUs absent at t1 are masked when scoring.
- CV: Grouped 5-fold by (block|treatment), One-vs-Rest LR, macro AUC over
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
from scripts.snowmelt.utils import load_snowmelt_metadata

SEED = 42
MIN_PREVALENCE = 10


#%% Load runs & group SRS by (block, treatment, time)
run_meta, run_to_srs = load_snowmelt_metadata('data/snowmelt/snowmelt.csv')
bt_time_to_srs = defaultdict(list)
for run, meta in run_meta.items():
    srs = run_to_srs.get(run)
    if not srs:
        continue
    key = (meta['block'], meta['treatment'], meta['time'])
    bt_time_to_srs[key].append(srs)


#%% Build SRS→OTUs and binary presence per group
needed_srs = {s for lst in bt_time_to_srs.values() for s in lst}
micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)

group_to_srs = {k: v for k, v in bt_time_to_srs.items()}
X, kept_otus, otu_index, keys, bt_presence, key_to_row = shared_utils.build_presence_matrix(
    group_to_srs, micro_to_otus, min_prevalence=MIN_PREVALENCE
)


#%% Build multilabel dropout dataset: all t1->t2 pairs within (block, treatment)
X_rows, Y_ml, M_ml, groups_ml = shared_utils.build_multilabel_pairs(
    keys=keys,
    presence_by_key=bt_presence,
    kept_otus=kept_otus,
    key_to_row=key_to_row,
    mode='dropout',
    group_id_func=lambda k: (k[0], k[1]),  # (block, treatment)
)

if X_rows.size == 0:
    print('No t1->t2 pairs available.')
else:
    X_ml = X[X_rows]
    print(f"Multilabel dropout: n={len(X_ml)}, d={X_ml.shape[1]}, labels={Y_ml.shape[1]}")

    # Single random split (not grouped) to mirror Gingiva baseline
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

    # ---- Grouped 5-fold CV by (block, treatment) ----
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

    shared_utils.eval_masked_ovr(
        "LogReg (OvR) — Dropout", lr_est, X_ml, Y_ml, M_ml, groups_ml, n_splits=5, seed=SEED
    )
    shared_utils.eval_masked_ovr(
        "RandForest (OvR) — Dropout", rf_est, X_ml, Y_ml, M_ml, groups_ml, n_splits=5, seed=SEED
    )



# %%
