#%% Snowmelt: Multilabel Colonisation (t1 -> t2)
"""
Single task (clean baseline):
- Input X: binary taxa presence at (block, treatment, t1) [union across SRS]
- Target Y: for each kept OTU j, Yj=1 if absent at t1 and present at t2 (colonisation),
            Yj=0 if absent at both; OTUs present at t1 are masked when scoring.
- CV: Grouped 5-fold by (block|treatment), One-vs-Rest LR, macro AUC over
       labels with both classes within masked examples.
"""

#%% Imports & config
import os, sys
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
    bt_time_to_srs[(meta['block'], meta['treatment'], meta['time'])].append(srs)


#%% Build SRS→OTUs and binary presence per group
needed_srs = {s for lst in bt_time_to_srs.values() for s in lst}
micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)

def union_presence(srs_list):
    seen = set()
    for srs in srs_list:
        for otu in micro_to_otus.get(srs, []):
            seen.add(otu)
    return seen

bt_presence = {k: union_presence(v) for k, v in bt_time_to_srs.items()}

prev = Counter()
for pres in bt_presence.values():
    prev.update(pres)
kept_otus = sorted([o for o, c in prev.items() if c >= MIN_PREVALENCE])
otu_index = {o: i for i, o in enumerate(kept_otus)}

keys = sorted(bt_presence.keys())
X = np.zeros((len(keys), len(kept_otus)), dtype=np.float32)
for i, key in enumerate(keys):
    for otu in bt_presence[key]:
        j = otu_index.get(otu)
        if j is not None:
            X[i, j] = 1.0


#%% Build multilabel colonisation dataset: all t1->t2 pairs within (block, treatment)
key_to_row = {k: i for i, k in enumerate(keys)}
bt_to_times = defaultdict(list)
for (b, tr, t) in keys:
    bt_to_times[(b, tr)].append(t)

X_pairs, Y_pairs, M_pairs, groups_pairs = [], [], [], []
for (b, tr), times in bt_to_times.items():
    for i in range(len(times)):
        for j in range(len(times)):
            if i == j:
                continue
            t1, t2 = times[i], times[j]
            k1 = (b, tr, t1)
            k2 = (b, tr, t2)
            row = key_to_row[k1]
            pres1 = bt_presence.get(k1, set())
            pres2 = bt_presence.get(k2, set())
            yrow = np.zeros(len(kept_otus), dtype=np.int64)
            mrow = np.zeros(len(kept_otus), dtype=bool)
            for idx, otu in enumerate(kept_otus):
                if otu not in pres1:  # absent at t1 → eligible for colonisation
                    mrow[idx] = True
                    yrow[idx] = 1 if otu in pres2 else 0
            X_pairs.append(row)
            Y_pairs.append(yrow)
            M_pairs.append(mrow)
            groups_pairs.append(f"{b}|{tr}")

if not X_pairs:
    print('No t1->t2 pairs available.')
else:
    X_ml = X[np.array(X_pairs, dtype=np.int64)]
    Y_ml = np.stack(Y_pairs)
    M_ml = np.stack(M_pairs)
    groups_ml = np.array(groups_pairs, dtype=object)
    print(f"Multilabel colonisation: n={len(X_ml)}, d={X_ml.shape[1]}, labels={Y_ml.shape[1]}")

    # Single random split (not grouped) for symmetry
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
