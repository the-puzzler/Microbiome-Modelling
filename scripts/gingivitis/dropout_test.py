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

OUT_DIR = os.path.join('data', 'paper_figures', 'drop_col_figures')

# If True, append 8 zero-valued scratch tokens per SRS to the transformer input
USE_ZERO_SCRATCH_TOKENS = True
SCRATCH_TOKENS_PER_SRS = 16

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


#%% Helper: score OTUs for a single SRS (optionally with zero scratch tokens)
def score_otus_for_srs(srs, prokbert_file, model, device, resolver=None):
    emb_group = prokbert_file['embeddings']
    if not USE_ZERO_SCRATCH_TOKENS:
        return shared_utils.score_otus_for_srs(
            srs,
            micro_to_otus=micro_to_otus,
            resolver=resolver,
            model=model,
            device=device,
            emb_group=emb_group,
        )

    otu_ids = micro_to_otus.get(srs, [])
    if not otu_ids:
        return {}
    vecs = []
    keep = []
    for oid in otu_ids:
        key = resolver.get(oid, oid) if resolver else oid
        if key in emb_group:
            vecs.append(torch.tensor(emb_group[key][()], dtype=torch.float32, device=device))
            keep.append(oid)
    if not vecs:
        return {}
    x1 = torch.stack(vecs, dim=0).unsqueeze(0)
    n1 = x1.shape[1]
    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        if SCRATCH_TOKENS_PER_SRS > 0:
            z = torch.zeros((1, SCRATCH_TOKENS_PER_SRS, shared_utils.D_MODEL), dtype=torch.float32, device=device)
            h = torch.cat([h1, z], dim=1)
            mask = torch.ones((1, n1 + SCRATCH_TOKENS_PER_SRS), dtype=torch.bool, device=device)
        else:
            h = h1
            mask = torch.ones((1, n1), dtype=torch.bool, device=device)
        h = model.transformer(h, src_key_padding_mask=~mask)
        logits = model.output_projection(h).squeeze(-1)
    logits_type1 = logits[:, :n1].squeeze(0).cpu().numpy()
    return dict(zip(keep, logits_type1))


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

        


#%% Text metadata integration — clean comparison (shared utils)
import matplotlib.pyplot as _plt

# Load text embeddings and RunID->terms mapping
term_to_vec = shared_utils.load_term_embeddings()
run_to_terms = shared_utils.parse_run_terms()

# Build SRS -> terms using MicrobeAtlas mapped headers expansion
run_to_srs = SRA_to_micro  # reuse mapping loaded above
srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)

# Per-(subject,time) averaged logits with text
subject_time_otu_scores_text = {}
subject_time_otu_presence_text = {}
with h5py.File(shared_utils.PROKBERT_PATH) as _embf:
    for (subject, time_code), srs_list in tqdm(list(subject_time_to_srs.items()), desc='Scoring with text'):
        sdicts = [shared_utils.score_otus_for_srs_with_text(
            srs,
            micro_to_otus=micro_to_otus,
            resolver=resolver,
            model=model,
            device=device,
            emb_group=_embf['embeddings'],
            term_to_vec=term_to_vec,
            srs_to_terms=srs_to_terms,
        ) for srs in srs_list]
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
os.makedirs(OUT_DIR, exist_ok=True)
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
_plt.savefig(os.path.join(OUT_DIR, 'gingivitis_dropout_roc_base_vs_text.png'), dpi=300)
_plt.close()

# Shared-axis density comparison for logits (no text vs with text)

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

plot_dropout_summary(
    y_score_all,
    y_drop,
    title_prefix='Gingiva',
    xlim=(xmin, xmax),
    ylim=None if ylim[1] is None else (ylim[0], ylim[1]),
    save_path=os.path.join(OUT_DIR, 'gingivitis_dropout_density_roc_base.png'),
)
if y_score_txt.size:
    plot_dropout_summary(
        y_score_txt,
        y_drop_txt,
        title_prefix='Gingiva (+ text)',
        xlim=(xmin, xmax),
        ylim=None if ylim[1] is None else (ylim[0], ylim[1]),
        save_path=os.path.join(OUT_DIR, 'gingivitis_dropout_density_roc_text.png'),
    )

# %%
#%% Term impact analysis (ablation on ROC AUC)
# For each term, remove it from all SRS term-sets, rescore, and measure
# AUC drop: contribution(term) = AUC_full_text - AUC_without_term
# Positive values indicate the term helps; negative indicates it hurts.
import matplotlib.pyplot as plt

# Select candidate terms observed in gingiva runs only, then rank by frequency
# Build set of terms attached to the gingiva dataset Run IDs
gingiva_runs = {r['run'] for r in records}
gingiva_terms = set()
for run in gingiva_runs:
    if run in run_to_terms:
        gingiva_terms.update(run_to_terms[run])

# Count frequencies across the SRS in this experiment, but only for gingiva-linked terms
term_counts = {}
for terms in srs_to_terms.values():
    for t in terms:
        if t in gingiva_terms:
            term_counts[t] = term_counts.get(t, 0) + 1

min_freq = 5
max_terms = 20
candidates = [t for t, c in sorted(term_counts.items(), key=lambda kv: kv[1], reverse=True) if c >= min_freq]
if len(candidates) > max_terms:
    candidates = candidates[:max_terms]

contrib = {}
with h5py.File(shared_utils.PROKBERT_PATH) as _embf:
    emb_group = _embf['embeddings']
    for term in candidates:
        # Build SRS -> terms with the term removed
        srs_to_terms_minus = {srs: (terms - {term}) if term in terms else terms for srs, terms in srs_to_terms.items()}
        # Recompute per-(subject,time) averages with the term ablated
        st_scores = {}
        st_presence = {}
        for (subject, time_code), srs_list in subject_time_to_srs.items():
            sdicts = [shared_utils.score_otus_for_srs_with_text(
                srs,
                micro_to_otus=micro_to_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
                term_to_vec=term_to_vec,
                srs_to_terms=srs_to_terms_minus,
            ) for srs in srs_list]
            sdicts = [d for d in sdicts if d]
            avg = shared_utils.union_average_logits(sdicts) if sdicts else {}
            st_scores[(subject, time_code)] = avg
            st_presence[(subject, time_code)] = set(avg.keys())
        # Build examples and compute AUC with the term removed
        y_true_m = []
        y_score_m = []
        for subject, pairs in pairs_by_subject.items():
            for t1, t2 in pairs:
                s1 = st_scores.get((subject, t1), {})
                p2 = st_presence.get((subject, t2), set())
                for otu, sc in s1.items():
                    y_true_m.append(1 if otu in p2 else 0)
                    y_score_m.append(sc)
        y_true_m = np.asarray(y_true_m, dtype=np.int64)
        y_score_m = np.asarray(y_score_m, dtype=np.float64)
        auc_minus = roc_auc_score(1 - y_true_m, -y_score_m) if y_true_m.size else float('nan')
        contrib[term] = float(auc_txt - auc_minus) if np.isfinite(auc_minus) and np.isfinite(auc_txt) else float('nan')

# Plot contributions
terms_sorted = [t for t, v in sorted(contrib.items(), key=lambda kv: (np.isnan(kv[1]), -kv[1] if not np.isnan(kv[1]) else 0)) if not np.isnan(v)]
vals_sorted = [contrib[t] for t in terms_sorted]

plt.figure(figsize=(10, 4.5))
plt.bar(range(len(terms_sorted)), vals_sorted, color=['#1f77b4' if v >= 0 else '#d62728' for v in vals_sorted])
plt.xticks(range(len(terms_sorted)), terms_sorted, rotation=45, ha='right')
plt.ylabel('AUC contribution (Δ AUC)')
plt.title('Per-term impact on dropout AUC (ablation)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'gingivitis_dropout_term_impact.png'), dpi=300)
plt.close()

# %%
