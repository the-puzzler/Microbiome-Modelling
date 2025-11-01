#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import csv
from collections import Counter

# Ensure project root is on sys.path so `from scripts import utils` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


#%% Paths
CANCER_RUN_TABLE = 'data/cancer/SraRunTableCancer.csv'


#%% Load cancer run table and explore candidate label columns
rows = []
with open(CANCER_RUN_TABLE) as f:
    reader = csv.DictReader(f)
    for r in reader:
        # Normalize keys slightly
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})

print('Cancer table rows:', len(rows))

run_ids = [r.get('Run', '') for r in rows if r.get('Run')]
print('Unique Run IDs:', len(set(run_ids)))

# Candidate columns for classification (presence depends on incoming table)
candidate_cols = [
    'env_local_scale',
    'env_medium',
    'env_broad_scale',
    'Assay Type',
]

def summarize_col(col):
    vals = [r.get(col, '') for r in rows]
    vals = [v for v in vals if v]
    if not vals:
        print(f"- {col}: (absent or empty)")
        return None
    cnt = Counter(vals)
    total = sum(cnt.values())
    top = ', '.join([f"{k}={cnt[k]}" for k in sorted(cnt, key=cnt.get, reverse=True)[:10]])
    print(f"- {col}: {len(cnt)} classes | total labeled={total} | top: {top}")
    return cnt

print('Candidate label columns:')
col_summaries = {col: summarize_col(col) for col in candidate_cols}

# Also show distribution restricted to runs that map to MicrobeAtlas SRS (usable subset)
acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
mapped_runs = [rid for rid in run_ids if rid in acc_to_srs]
print('Runs mapped to MicrobeAtlas SRS:', len(set(mapped_runs)))

print('Mapped-run label distributions:')
for col in candidate_cols:
    vals = [r.get(col, '') for r in rows if r.get('Run') in acc_to_srs]
    vals = [v for v in vals if v]
    cnt = Counter(vals)
    if cnt:
        top = ', '.join([f"{k}={cnt[k]}" for k in sorted(cnt, key=cnt.get, reverse=True)[:10]])
        print(f"- {col}: {len(cnt)} classes | top: {top}")
    else:
        print(f"- {col}: (no labels among mapped runs)")


#%% Next steps (placeholder)
# After choosing a target column (e.g., env_local_scale vs env_medium), the flow will be:
# 1) Build SRS -> OTUs for mapped runs
# 2) Load model and build per-SRS embeddings (optionally with text terms)
# 3) Assemble (X, y) and run a logistic regression with CV
# 4) Report AUC/AP and confusion matrices

# %%
