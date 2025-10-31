#%% Imports and setup
import csv
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure project root is on sys.path so `from scripts import utils` works when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import (
    load_gingivitis_run_data,
    collect_micro_to_otus,
    plot_dropout_summary,
)  # noqa: E402


#%% Load gingivitis runs and map to MicrobeAtlas SRS
gingivitis_csv = 'data/gingivitis/gingiva.csv'
run_ids, SRA_to_micro = load_gingivitis_run_data(
    gingivitis_path=gingivitis_csv,
    microbeatlas_path=shared_utils.MICROBEATLAS_SAMPLES,
)
print(f"loaded runs: {len(run_ids)} | mapped to SRS: {len(SRA_to_micro)}")


#%% Collect SRS -> OTUs from mapped file (no BIOM fallback)
micro_to_otus = collect_micro_to_otus(SRA_to_micro)
if micro_to_otus:
    lens = [len(v) for v in micro_to_otus.values()]
    print(
        "SRS→OTUs prepared:",
        len(micro_to_otus),
        "samples | avg OTUs per SRS:",
        round(sum(lens) / max(1, len(lens)), 1),
    )
else:
    print("WARNING: No SRS→OTUs were found. Check mappings and mapped file.")


#%% Read gingivitis metadata to group by patient and timepoint
records = []
with open(gingivitis_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        run = row.get('Run', '').strip()
        subj = row.get('subject_code', '').strip()
        tcode = row.get('time_code', '').strip()
        if not run or not subj or not tcode:
            continue
        # Map run -> SRS; some runs may not map
        srs = SRA_to_micro.get(run)
        if not srs:
            continue
        records.append({'run': run, 'srs': srs, 'subject': subj, 'time': tcode})

# Build grouping: per subject, per time, list of SRS samples
subject_time_to_srs = defaultdict(list)
for r in records:
    subject_time_to_srs[(r['subject'], r['time'])].append(r['srs'])

subjects = sorted({r['subject'] for r in records})
print('subjects with mapped runs:', len(subjects))
print('subject-time groups:', len(subject_time_to_srs))


#%% Load model checkpoint
model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)


#%% Load OTU rename map for translating new IDs to old ProkBERT IDs
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = None
if rename_map:
    # Gingiva is bacterial; prefer B when both exist
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B')


#%% Helper: score OTUs for a single SRS via shared util (raw logits)
def score_otus_for_srs(srs, prokbert_file, model, device, resolver=None):
    emb_group = prokbert_file['embeddings']
    return shared_utils.score_otus_for_srs(
        srs,
        micro_to_otus=micro_to_otus,
        resolver=resolver,
        model=model,
        device=device,
        emb_group=emb_group,
    )


#%% Compute per subject-time per-OTU averaged logits across samples
subject_time_otu_scores = {}
subject_time_otu_presence = {}

with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
    print("opened ProkBERT embeddings store")
    for (subject, time_code), srs_list in tqdm(
        list(subject_time_to_srs.items()),
        desc='Scoring subject-time groups',
    ):
        # Collect per-sample score dicts
        sample_dicts = []
        for srs in tqdm(srs_list, desc='SRS in group', leave=False):
            sdict = score_otus_for_srs(srs, emb_file, model, device, resolver=resolver)
            if sdict:
                sample_dicts.append(sdict)
        if not sample_dicts:
            subject_time_otu_scores[(subject, time_code)] = {}
            subject_time_otu_presence[(subject, time_code)] = set()
            continue
        # Union-average logits across SRS in the group
        avg_scores = shared_utils.union_average_logits(sample_dicts)
        subject_time_otu_scores[(subject, time_code)] = avg_scores
        subject_time_otu_presence[(subject, time_code)] = set(avg_scores.keys())

print('computed score maps for groups:', len(subject_time_otu_scores))


#%% Binary prediction within-person: T1 logits predict T2 presence
def safe_float_time(t):
    try:
        return float(t)
    except Exception:
        return None


# Build all-vs-all timepoint pairs per subject (t1 != t2)
pairs_by_subject = {}
total_pairs = 0
for subject in subjects:
    times = sorted({t for (s, t) in subject_time_otu_scores.keys() if s == subject},
                   key=lambda x: (safe_float_time(x) is None, safe_float_time(x), x))
    if len(times) < 2:
        continue
    pairs = [(t1, t2) for t1 in times for t2 in times if t1 != t2]
    if pairs:
        pairs_by_subject[subject] = pairs
        total_pairs += len(pairs)

print('subjects with ≥2 timepoints:', len(pairs_by_subject), '| total pairs:', total_pairs)

# Assemble binary classification dataset: from T1 logits predict presence at T2
y_true_all = []
y_score_all = []
for subject, pairs in tqdm(pairs_by_subject.items(), desc='Building T1→T2 examples'):
    n_examples = 0
    for t1, t2 in pairs:
        scores_t1 = subject_time_otu_scores.get((subject, t1), {})
        present_t2 = subject_time_otu_presence.get((subject, t2), set())
        for otu, score in scores_t1.items():
            # Only consider OTUs that exist at T1
            label = 1 if otu in present_t2 else 0
            y_true_all.append(label)
            y_score_all.append(score)
            n_examples += 1
    # no per-subject reporting

y_true_all = np.array(y_true_all, dtype=np.int64)
y_score_all = np.array(y_score_all, dtype=np.float32)
print('total examples:', len(y_true_all))

 
# Dropout perspective only: label=1 means dropped out at T2
y_drop = 1 - y_true_all
# Use negative logits as decision scores (monotonic to 1 - prob)
scores_drop = -y_score_all
probs = 1 / (1 + np.exp(-y_score_all))
prob_drop = 1 - probs
overall_auc = roc_auc_score(y_drop, scores_drop)
overall_ap = average_precision_score(y_drop, prob_drop)
pos_frac = float(y_drop.mean())
print(f'Overall ROC AUC: {overall_auc:.4f} | AP: {overall_ap:.4f} | dropout rate: {pos_frac:.3f}')
# Visualise distributions and ROC
plot_dropout_summary(y_score_all, y_drop, title_prefix='Gingiva')

        



# %%
