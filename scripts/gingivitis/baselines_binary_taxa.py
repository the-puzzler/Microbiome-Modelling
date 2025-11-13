#%% Gingiva: Multilabel Dropout (t1 -> t2)
"""
Single task (clean baseline):
- Input X: binary taxa presence at (subject, t1) [union across SRS]
- Target Y: for each kept OTU j, Yj=1 if present at t1 and absent at t2 (dropout),
            Yj=0 if present at both; OTUs absent at t1 are masked when scoring.
- CV: Grouped 5-fold by subject, One-vs-Rest Logistic Regression, macro AUC over
       labels with both classes within the masked examples.
"""

#%% Imports & config
import os, sys, csv
from collections import Counter, defaultdict
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
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
def union_presence(srs_list):
    seen = set()
    for srs in srs_list:
        for otu in micro_to_otus.get(srs, []):
            seen.add(otu)
    return seen

st_presence = {key: union_presence(srs_list) for key, srs_list in subject_time_to_srs.items()}

# Prevalence filter across groups
prev = Counter()
for pres in st_presence.values():
    prev.update(pres)
kept_otus = sorted([otu for otu, c in prev.items() if c >= MIN_PREVALENCE])
otu_index = {otu: i for i, otu in enumerate(kept_otus)}

keys = sorted(st_presence.keys())
X = np.zeros((len(keys), len(kept_otus)), dtype=np.float32)
for i, key in enumerate(keys):
    for otu in st_presence[key]:
        j = otu_index.get(otu)
        if j is not None:
            X[i, j] = 1.0


#%% Build multilabel dropout dataset: all t1->t2 pairs within subject (t1 != t2)
key_to_row = {k: i for i, k in enumerate(keys)}
times_by_subject = defaultdict(list)
for (subj, t) in keys:
    times_by_subject[subj].append(t)

X_pairs, Y_pairs, M_pairs, groups_pairs = [], [], [], []
for subj, times in times_by_subject.items():
    for i in range(len(times)):
        for j in range(len(times)):
            if i == j:
                continue
            t1, t2 = times[i], times[j]
            k1 = (subj, t1)
            k2 = (subj, t2)
            row = key_to_row[k1]
            pres1 = st_presence.get(k1, set())
            pres2 = st_presence.get(k2, set())
            yrow = np.zeros(len(kept_otus), dtype=np.int64)
            mrow = np.zeros(len(kept_otus), dtype=bool)
            for idx, otu in enumerate(kept_otus):
                if otu in pres1:
                    mrow[idx] = True
                    yrow[idx] = 1 if otu not in pres2 else 0
            X_pairs.append(row)
            Y_pairs.append(yrow)
            M_pairs.append(mrow)
            groups_pairs.append(subj)

if not X_pairs:
    print('No t1->t2 pairs available.')
else:
    X_ml = X[np.array(X_pairs, dtype=np.int64)]
    Y_ml = np.stack(Y_pairs)
    M_ml = np.stack(M_pairs)
    groups_ml = np.array(groups_pairs, dtype=object)
    print(f"Multilabel dropout: n={len(X_ml)}, d={X_ml.shape[1]}, labels={Y_ml.shape[1]}")

    # Single random split (not grouped) to increase label support
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
        print(f"Random split macro AUC: {macro:.3f} (labels with both classes: {len(per_label)})")
        total_labels = Y_ml.shape[1]
        print(f"Label diagnostics (train): no-train-examples={no_train_examples}/{total_labels}, single-class={single_class_train}/{total_labels}")
    else:
        print('No evaluable labels in the test split (all single-class after masking).')

# %%
# --- Grouped 5-fold CV (by subject), evaluate LogReg and RandomForest with masked AUC ---
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import math

gkf = GroupKFold(n_splits=5)

def eval_model(name, base_estimator, X_ml, Y_ml, M_ml, groups_ml, SEED):
    fold_aucs = []
    total_labels = Y_ml.shape[1]

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_ml, groups=groups_ml), 1):
        tr_mask = np.zeros(X_ml.shape[0], dtype=bool); tr_mask[tr_idx] = True
        te_mask = np.zeros(X_ml.shape[0], dtype=bool); te_mask[te_idx] = True

        # ---- Label diagnostics on TRAIN split (eligible = present at t1 for DROPOUT) ----
        no_train_examples = 0
        single_class_train = 0
        for j in range(total_labels):
            elig = M_ml[tr_mask, j]  # eligible positions for label j in TRAIN
            if not np.any(elig):
                no_train_examples += 1
                continue
            y_tr = Y_ml[tr_mask, j][elig]
            if np.unique(y_tr).size < 2:
                single_class_train += 1

        # ---- Fit & predict (OvR wrapper) ----
        clf = OneVsRestClassifier(base_estimator)
        clf.fit(X_ml[tr_mask], Y_ml[tr_mask])
        prob = clf.predict_proba(X_ml[te_mask])  # probabilities per label

        # ---- Masked per-label AUC on TEST split (evaluate only eligible: present at t1) ----
        per_label = []
        for j in range(total_labels):
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

        fold_auc = float(np.mean(per_label)) if per_label else float("nan")
        fold_aucs.append(fold_auc)

        print(f"[{name}] Fold {fold}: macro AUC={fold_auc:.3f} over {len(per_label)} evaluable labels")
        print(f"[{name}] Label diagnostics (train): no-train-examples={no_train_examples}/{total_labels}, "
              f"single-class={single_class_train}/{total_labels}")

    valid = [a for a in fold_aucs if not (math.isnan(a) or math.isinf(a))]
    if valid:
        mean_auc = np.mean(valid)
        std_auc = np.std(valid)
        print(f"[{name}] Grouped 5-fold macro AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    else:
        print(f"[{name}] No evaluable labels across CV folds.")

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

# ---- Run both baselines on the DROPOUT dataset ----
eval_model("LogReg (OvR) — Dropout", lr_est, X_ml, Y_ml, M_ml, groups_ml, SEED)
eval_model("RandForest (OvR) — Dropout", rf_est, X_ml, Y_ml, M_ml, groups_ml, SEED)

# %%
