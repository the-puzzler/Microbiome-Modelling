#!/usr/bin/env python3
"""IBS prediction across countries using frozen microbiome embeddings.

Tasks:
  - Binary IBS label: "Diagnosed by a medical professional" vs "I do not have this condition".
  - Countries: USA, United Kingdom, Australia (others ignored if present).
  - Build per-sample embeddings from the frozen backbone (no fine-tuning).
  - Train simple logistic regression heads on top of embeddings.
  - Evaluate cross-country generalisation: train on one country, test on
    held-out countries (5 repeats with stratified 50/50 split in the source).

Relies on shared utils for MicrobeAtlas mappings, OTU→embedding resolution,
and model loading.
"""

import os
import sys
import csv
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


SEED = 42
IBS_META_CSV = 'data/IBS/final_metadata.csv'


def load_ibs_metadata(path=IBS_META_CSV):
    """Load IBS metadata and return list of (run_id, country, label_text).

    - Excludes self-diagnosed IBS entries.
    - Keeps only runs where IBS is either "I do not have this condition" or
      "Diagnosed by a medical professional (doctor, physician assistant)".
    """
    records = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get('run_id', '').strip()
            country = row.get('country', '').strip()
            ibs = row.get('ibs', '').strip()
            if not run_id or not country or not ibs:
                continue
            if 'Self-diagnosed' in ibs:
                continue
            if ibs not in {
                'I do not have this condition',
                'Diagnosed by a medical professional (doctor, physician assistant)',
            }:
                continue
            records.append((run_id, country, ibs))
    print('Loaded IBS metadata records (filtered):', len(records))
    return records


def map_runs_to_srs(run_ids):
    """Map SRA run IDs to MicrobeAtlas SRS via the mapped headers.

    Returns dict run_id -> srs.
    """
    acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
    run_to_srs = {rid: acc_to_srs[rid] for rid in run_ids if rid in acc_to_srs}
    print('Runs mapped to SRS:', len(run_to_srs), 'of', len(run_ids))
    return run_to_srs


def build_sample_embeddings_for_runs(run_to_srs):
    """Build frozen sample embeddings for the SRS used by IBS runs.

    Returns dict srs -> embedding (torch tensor on CPU).
    """
    needed_srs = set(run_to_srs.values())
    micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
    print('SRS with OTUs for IBS:', len(micro_to_otus))

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}
    sample_embeddings, _ = shared_utils.build_sample_embeddings(
        micro_to_otus,
        model,
        device,
        prokbert_path=shared_utils.PROKBERT_PATH,
        txt_emb=shared_utils.TXT_EMB,
        rename_map=rename_map,
        resolver=resolver,
    )
    print('IBS SRS embeddings:', len(sample_embeddings))
    return sample_embeddings


def assemble_country_datasets(records, run_to_srs, sample_embeddings):
    """Build per-country X, y arrays from IBS records.

    Label mapping:
        1 = medically diagnosed IBS
        0 = no IBS
    Self-diagnosed cases were already filtered out.
    """
    by_country = defaultdict(lambda: {'X': [], 'y': []})

    def label_from_text(txt):
        return 1 if 'Diagnosed by a medical professional' in txt else 0

    for run_id, country, ibs in records:
        srs = run_to_srs.get(run_id)
        if not srs:
            continue
        emb = sample_embeddings.get(srs)
        if emb is None:
            continue
        y = label_from_text(ibs)
        by_country[country]['X'].append(emb.numpy())
        by_country[country]['y'].append(y)

    # Convert to numpy arrays
    datasets = {}
    for country, data in by_country.items():
        if not data['X']:
            continue
        X = np.stack(data['X'])
        y = np.asarray(data['y'], dtype=np.int64)
        datasets[country] = (X, y)
        pos = int(y.sum())
        neg = int((y == 0).sum())
        print(f"{country}: {X.shape[0]} samples (pos={pos}, neg={neg})")

    return datasets


def evaluate_cross_country(datasets, seeds=5):
    """Train logistic regression on one country, evaluate on all countries.

    For each source country and each seed:
      - Stratified 50/50 split of that country's data into train/test.
      - Train a logistic regression on the source train set.
      - Evaluate AUC on:
          * source test set (source -> source)
          * full target dataset for every other country (source -> target).
    Returns dict (src, tgt) -> list of AUCs.
    """
    countries = sorted(datasets.keys())
    results = defaultdict(list)

    for seed in range(seeds):
        for src in countries:
            X_src, y_src = datasets[src]
            if len(np.unique(y_src)) < 2:
                print(f"Skipping {src} for seed {seed}: only one class present.")
                continue
            idx = np.arange(len(X_src))
            tr_idx, te_idx = train_test_split(
                idx,
                test_size=0.5,
                stratify=y_src,
                random_state=SEED + seed,
            )
            X_tr, y_tr = X_src[tr_idx], y_src[tr_idx]
            X_te, y_te = X_src[te_idx], y_src[te_idx]

            # Simple logistic regression head on frozen embeddings
            clf = LogisticRegression(
                solver='lbfgs',
                penalty='l2',
                C=1.0,
                class_weight='balanced',
                max_iter=2000,
            )
            clf.fit(X_tr, y_tr)

            # Source -> source AUC
            try:
                p_src = clf.predict_proba(X_te)[:, 1]
                auc_src = roc_auc_score(y_te, p_src)
            except Exception:
                auc_src = float('nan')
            results[(src, src)].append(auc_src)

            # Source -> target AUCs
            for tgt in countries:
                if tgt == src:
                    continue
                X_tgt, y_tgt = datasets[tgt]
                if len(np.unique(y_tgt)) < 2:
                    continue
                try:
                    p_tgt = clf.predict_proba(X_tgt)[:, 1]
                    auc_tgt = roc_auc_score(y_tgt, p_tgt)
                except Exception:
                    auc_tgt = float('nan')
                results[(src, tgt)].append(auc_tgt)

    # Print summary matrix
    print('\nCross-country mean AUC (source rows → target columns):')
    print('Countries:', countries)
    mat = np.full((len(countries), len(countries)), np.nan)
    for i, src in enumerate(countries):
        for j, tgt in enumerate(countries):
            vals = results.get((src, tgt), [])
            if vals:
                mat[i, j] = float(np.nanmean(vals))
    print(mat)

    return results


def main():
    records = load_ibs_metadata(IBS_META_CSV)
    if not records:
        raise SystemExit('No IBS records after filtering; aborting.')
    run_ids = [r[0] for r in records]

    run_to_srs = map_runs_to_srs(run_ids)
    if not run_to_srs:
        raise SystemExit('No runs could be mapped to SRS; aborting.')

    sample_embeddings = build_sample_embeddings_for_runs(run_to_srs)
    if not sample_embeddings:
        raise SystemExit('No embeddings built for IBS SRS; aborting.')

    datasets = assemble_country_datasets(records, run_to_srs, sample_embeddings)
    if len(datasets) < 2:
        raise SystemExit('Need at least two countries with data for cross-country evaluation.')

    _ = evaluate_cross_country(datasets, seeds=5)


if __name__ == '__main__':
    main()

