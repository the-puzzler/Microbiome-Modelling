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

        


#%% Text metadata integration — clean comparison (no fallbacks)
import pickle as _pkl
import matplotlib.pyplot as _plt

TERMS_MAP_TSV = 'data/microbeatlas/sample_terms_mapping_combined_dany_og_biome_tech.txt'
TEXT_EMB_PKL = 'data/microbeatlas/word_embeddings_dany_biomes_combined_dany_og_biome_tech.pkl'

# Load term embeddings (term -> np.array), lowercased, tensors on device
with open(TEXT_EMB_PKL, 'rb') as _f:
    _term_to_vec_np = _pkl.load(_f)
term_to_vec = {str(k).lower(): torch.tensor(np.asarray(v), dtype=torch.float32, device=device)
               for k, v in _term_to_vec_np.items()}

# Parse RunID -> terms (tab-separated, 1st col is RunID, rest are terms)
run_to_terms = {}
with open(TERMS_MAP_TSV, 'r', errors='replace') as _f:
    _f.readline()  # header
    for line in _f:
        parts = line.rstrip('\n').split('\t')
        if not parts:
            continue
        rid = parts[0].strip()
        terms = [t.strip().lower() for t in parts[1:] if t and t.strip()]
        if rid and terms:
            run_to_terms[rid] = terms

# Build SRS -> terms by expanding all accessions mapping to the same SRS
acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
srs_to_accs = defaultdict(list)
for acc, srs in acc_to_srs.items():
    srs_to_accs[srs].append(acc)

srs_to_terms = defaultdict(set)
for r in records:
    srs = r['srs']
    for acc in srs_to_accs.get(srs, []):
        if acc in run_to_terms:
            srs_to_terms[srs].update(run_to_terms[acc])

# Helper: score SRS with both OTU and text embeddings (text active)
def score_otus_for_srs_with_text(srs, prokbert_file, model, device, resolver=None):
    otu_ids = micro_to_otus.get(srs, [])
    if not otu_ids:
        return {}
    emb_group = prokbert_file['embeddings']
    vecs, keep = [], []
    for oid in otu_ids:
        key = resolver.get(oid, oid) if resolver else oid
        if key in emb_group:
            vecs.append(torch.tensor(emb_group[key][()], dtype=torch.float32, device=device))
            keep.append(oid)
    if not vecs:
        return {}
    x1 = torch.stack(vecs, dim=0).unsqueeze(0)
    t_terms = [t for t in sorted(srs_to_terms.get(srs, set())) if t in term_to_vec]
    x2 = torch.stack([term_to_vec[t] for t in t_terms], dim=0).unsqueeze(0) if t_terms else torch.empty((1, 0, int(shared_utils.TXT_EMB)), dtype=torch.float32, device=device)
    n1, n2 = x1.shape[1], x2.shape[1]
    mask = torch.ones((1, n1 + n2), dtype=torch.bool, device=device)
    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        h2 = model.input_projection_type2(x2)
        h = torch.cat([h1, h2], dim=1)
        h = model.transformer(h, src_key_padding_mask=~mask)
        out = model.output_projection(h).squeeze(-1)
    logits_type1 = out[:, :n1].squeeze(0).detach().cpu().numpy()
    return dict(zip(keep, logits_type1))

# Per-(subject,time) averaged logits with text
subject_time_otu_scores_text = {}
subject_time_otu_presence_text = {}
with h5py.File(shared_utils.PROKBERT_PATH) as _embf:
    for (subject, time_code), srs_list in tqdm(list(subject_time_to_srs.items()), desc='Scoring with text'):
        sdicts = [score_otus_for_srs_with_text(srs, _embf, model, device, resolver=resolver) for srs in srs_list]
        sdicts = [d for d in sdicts if d]
        avg = shared_utils.union_average_logits(sdicts) if sdicts else {}
        subject_time_otu_scores_text[(subject, time_code)] = avg
        subject_time_otu_presence_text[(subject, time_code)] = set(avg.keys())

# Build examples and compute AUC/AP (all-pairs)
y_true_txt = []
y_score_txt = []
for subject, pairs in pairs_by_subject.items():
    for t1, t2 in pairs:
        s1 = subject_time_otu_scores_text.get((subject, t1), {})
        p2 = subject_time_otu_presence_text.get((subject, t2), set())
        for otu, sc in s1.items():
            y_true_txt.append(1 if otu in p2 else 0)
            y_score_txt.append(sc)

y_true_txt = np.asarray(y_true_txt, dtype=np.int64)
y_drop_txt = 1 - y_true_txt
y_score_txt = np.asarray(y_score_txt, dtype=np.float64)

auc_txt = roc_auc_score(y_drop_txt, -y_score_txt) if y_true_txt.size else float('nan')
ap_txt = average_precision_score(y_drop_txt, 1 - 1/(1 + np.exp(-y_score_txt))) if y_true_txt.size else float('nan')

print(f'Baseline AUC/AP (no text): {overall_auc:.4f} / {overall_ap:.4f}')
print(f'With text AUC/AP        : {auc_txt:.4f} / {ap_txt:.4f}')
if np.isfinite(auc_txt):
    print(f'Delta AUC (text - base) : {auc_txt - overall_auc:+.4f}')

# Overlay ROC curves for comparison
from sklearn.metrics import roc_curve
fpr_b, tpr_b, _ = roc_curve(y_drop, -y_score_all)
fpr_t, tpr_t, _ = roc_curve(y_drop_txt, -y_score_txt)
_plt.figure(figsize=(6, 5))
_plt.plot(fpr_b, tpr_b, label=f'No text (AUC={overall_auc:.3f})')
_plt.plot(fpr_t, tpr_t, label=f'With text (AUC={auc_txt:.3f})')
_plt.plot([0,1],[0,1],'k--', linewidth=1)
_plt.xlabel('False Positive Rate')
_plt.ylabel('True Positive Rate')
_plt.title('Dropout ROC: no text vs with text')
_plt.legend(loc='lower right')
_plt.tight_layout()
_plt.show()

# Shared-axis density comparison for logits (no text vs with text)
try:
    xmin = float(min(y_score_all.min(), y_score_txt.min())) if y_score_txt.size else float(y_score_all.min())
    xmax = float(max(y_score_all.max(), y_score_txt.max())) if y_score_txt.size else float(y_score_all.max())
    # Estimate a common y-limit by computing density maxima over both cases
    def _max_density(vals, labels, bins=40, rng=(xmin, xmax)):
        vals = np.asarray(vals, dtype=float)
        y = np.asarray(labels, dtype=int)
        m0 = y == 0
        m1 = y == 1
        dens_max = 0.0
        if np.any(m0):
            h0, _ = np.histogram(vals[m0], bins=bins, range=rng, density=True)
            dens_max = max(dens_max, float(h0.max()) if h0.size else 0.0)
        if np.any(m1):
            h1, _ = np.histogram(vals[m1], bins=bins, range=rng, density=True)
            dens_max = max(dens_max, float(h1.max()) if h1.size else 0.0)
        return dens_max
    ymax_base = _max_density(y_score_all, y_drop)
    ymax_text = _max_density(y_score_txt, y_drop_txt) if y_score_txt.size else 0.0
    ylim = (0.0, max(ymax_base, ymax_text) * 1.05 if max(ymax_base, ymax_text) > 0 else None)

    plot_dropout_summary(y_score_all, y_drop, title_prefix='Gingiva', xlim=(xmin, xmax), ylim=None if ylim[1] is None else (ylim[0], ylim[1]))
    if y_score_txt.size:
        plot_dropout_summary(y_score_txt, y_drop_txt, title_prefix='Gingiva (+ text)', xlim=(xmin, xmax), ylim=None if ylim[1] is None else (ylim[0], ylim[1]))
except Exception as _e_shared:
    print(f"Shared-axis density plotting failed: {_e_shared}")

# %%
