#!/usr/bin/env python3
"""
Gingiva: Multilabel Colonisation and Dropout (t1 -> t2) baselines
using a small MLP classifier instead of Logistic Regression / Random Forest.

The data preprocessing and masking logic follow the existing
`baselines_colonisation.py` and `baselines_binary_taxa.py` scripts exactly.
"""

#%% Imports & config
import os
import sys
import csv
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils
from scripts.gingivitis.utils import (
    load_gingivitis_run_data,
    collect_micro_to_otus,
)

SEED = 42
MIN_PREVALENCE = 10


def _load_gingivitis_presence():
    """Build binary taxa presence per (subject, time) as in the baseline scripts."""
    gingivitis_csv = "data/gingivitis/gingiva.csv"
    run_ids, SRA_to_micro = load_gingivitis_run_data(
        gingivitis_path=gingivitis_csv,
        microbeatlas_path=shared_utils.MICROBEATLAS_SAMPLES,
    )
    micro_to_otus = collect_micro_to_otus(SRA_to_micro)

    records = []
    with open(gingivitis_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("Run", "").strip()
            subj = row.get("subject_code", "").strip()
            tcode = row.get("time_code", "").strip()
            if not run or not subj or not tcode:
                continue
            srs = SRA_to_micro.get(run)
            if srs:
                records.append({"srs": srs, "subject": subj, "time": tcode})

    subject_time_to_srs = defaultdict(list)
    for r in records:
        subject_time_to_srs[(r["subject"], r["time"])].append(r["srs"])

    group_to_srs = {k: v for k, v in subject_time_to_srs.items()}
    X, kept_otus, otu_index, keys, st_presence, key_to_row = shared_utils.build_presence_matrix(
        group_to_srs, micro_to_otus, min_prevalence=MIN_PREVALENCE
    )
    return X, kept_otus, keys, st_presence, key_to_row


def _build_multilabel_task(mode):
    """
    Build multilabel colonisation/dropout dataset using shared_utils.build_multilabel_pairs,
    matching gingivitis baseline scripts.
    """
    X, kept_otus, keys, st_presence, key_to_row = _load_gingivitis_presence()
    X_rows, Y_ml, M_ml, groups_ml = shared_utils.build_multilabel_pairs(
        keys=keys,
        presence_by_key=st_presence,
        kept_otus=kept_otus,
        key_to_row=key_to_row,
        mode=mode,
        group_id_func=lambda k: k[0],  # subject
    )
    if X_rows.size == 0:
        return None, None, None, None
    X_ml = X[X_rows]
    return X_ml, Y_ml, M_ml, groups_ml


def _make_mlp(hidden_mult=1.0):
    """
    Small MLP classifier wrapped in OneVsRest, mirroring the interface of the
    Logistic Regression / Random Forest baselines.
    """
    # Hidden size scaled as a small multiple of input dim; actual dim is set at fit time.
    # We keep architecture modest to mirror "baseline" complexity.
    base_hidden = 128
    hidden_units = max(16, int(base_hidden * hidden_mult))
    est = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=SEED,
    )
    return OneVsRestClassifier(est)


def _diagnostic_split_and_report(X_ml, Y_ml, M_ml, label):
    """
    Run the same 80/20 random split diagnostic as in the LR/RF baselines
    but using the MLP, and report macro AUC over evaluable labels.
    """
    rng = np.random.default_rng(SEED)
    n_samples = X_ml.shape[0]
    idx = rng.permutation(n_samples)
    n_train = max(1, int(0.8 * n_samples))
    tr_idx = idx[:n_train]
    te_idx = idx[n_train:]
    tr_mask = np.zeros(n_samples, dtype=bool)
    te_mask = np.zeros(n_samples, dtype=bool)
    tr_mask[tr_idx] = True
    te_mask[te_idx] = True

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

    clf = _make_mlp()
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
        print(
            "[{}] Random split macro AUC (MLP): {:.3f} (labels with both classes: {})".format(
                label, macro, len(per_label)
            )
        )
        print(
            "[{}] Label diagnostics (train): no-train-examples={}/{}, single-class={}/{}".format(
                label, no_train_examples, total_labels, single_class_train, total_labels
            )
        )
    else:
        print(
            "[{}] No evaluable labels in the test split (all single-class after masking).".format(
                label
            )
        )


def run_gingivitis_mlp():
    # Colonisation
    print("=== Gingivitis: Multilabel Colonisation (MLP) ===")
    X_ml, Y_ml, M_ml, groups_ml = _build_multilabel_task(mode="colonisation")
    if X_ml is None:
        print("No t1->t2 pairs available for colonisation.")
    else:
        print(
            "Multilabel colonisation (MLP): n={}, d={}, labels={}".format(
                len(X_ml), X_ml.shape[1], Y_ml.shape[1]
            )
        )
        _diagnostic_split_and_report(X_ml, Y_ml, M_ml, label="Colonisation")
        mlp_est = _make_mlp()
        shared_utils.eval_masked_ovr(
            "MLP (OvR) — Colonisation",
            mlp_est,
            X_ml,
            Y_ml,
            M_ml,
            groups_ml,
            n_splits=5,
            seed=SEED,
        )

    # Dropout
    print("\n=== Gingivitis: Multilabel Dropout (MLP) ===")
    X_ml, Y_ml, M_ml, groups_ml = _build_multilabel_task(mode="dropout")
    if X_ml is None:
        print("No t1->t2 pairs available for dropout.")
    else:
        print(
            "Multilabel dropout (MLP): n={}, d={}, labels={}".format(
                len(X_ml), X_ml.shape[1], Y_ml.shape[1]
            )
        )
        _diagnostic_split_and_report(X_ml, Y_ml, M_ml, label="Dropout")
        mlp_est = _make_mlp()
        shared_utils.eval_masked_ovr(
            "MLP (OvR) — Dropout",
            mlp_est,
            X_ml,
            Y_ml,
            M_ml,
            groups_ml,
            n_splits=5,
            seed=SEED,
        )


if __name__ == "__main__":
    run_gingivitis_mlp()
