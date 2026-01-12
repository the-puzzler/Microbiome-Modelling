#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
import h5py
import random
import torch

import umap.umap_ as umap


# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import (
    load_gingivitis_run_data,
    collect_micro_to_otus,
)  # noqa: E402


#%% Load gingivitis mapping and group by subject × time
gingivitis_csv = 'data/gingivitis/gingiva.csv'
run_ids, SRA_to_micro = load_gingivitis_run_data(gingivitis_path=gingivitis_csv)
print(f"loaded runs: {len(run_ids)} | mapped to SRS: {len(SRA_to_micro)}")

# Read subject/time per run and build subject-time grouping of SRS
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
        if not srs:
            continue
        records.append({'srs': srs, 'subject': subj, 'time': tcode})

subject_time_to_srs = {}
subjects = set()
times = set()
for r in records:
    key = (r['subject'], r['time'])
    subject_time_to_srs.setdefault(key, []).append(r['srs'])
    subjects.add(r['subject'])
    times.add(r['time'])

print('subject-time groups:', len(subject_time_to_srs))


#%% Collect SRS → OTUs and build per-SRS embeddings
micro_to_otus = collect_micro_to_otus(SRA_to_micro)
print('SRS with OTUs:', len(micro_to_otus))

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
print('SRS embeddings:', len(sample_embeddings))


#%% New visual: pick a random SRS, add random DNA embeddings, and visualise per-layer outputs
# Pick a random SRS with OTUs
srs_candidates = [s for s, otus in micro_to_otus.items() if otus]
if not srs_candidates:
    raise SystemExit('No SRS with OTUs found.')
rand_srs = random.choice(srs_candidates)
otus = micro_to_otus[rand_srs]
print('selected SRS:', rand_srs, 'with OTUs:', len(otus))

def resolve_embedding_vectors(oid_list, emb_group):
    vecs = []
    for oid in oid_list:
        key = resolver.get(oid, oid)
        if key in emb_group:
            vecs.append(emb_group[key][()])
    return vecs

# Collect DNA (ProkBERT) embeddings for original OTUs and added OTUs from other gingiva
dna_vecs = []
added_vecs = []
with h5py.File(shared_utils.PROKBERT_PATH) as f:
    emb = f['embeddings']
    # Original SRS OTUs
    dna_vecs = resolve_embedding_vectors(otus, emb)
    # Sample added OTUs randomly from the whole embedding dataset (15% of originals)
    n_rand = max(1, int(round(0.15 * len(dna_vecs))))
    # Draw a larger pool then trim after filtering
    try:
        all_keys = list(emb.keys())
    except Exception:
        all_keys = []
        # Fallback: iterate first N
        for i, k in enumerate(emb.keys()):
            all_keys.append(k)
            if i > 200000:
                break
    random.shuffle(all_keys)
    added_vecs = []
    for k in all_keys:
        if isinstance(k, (bytes, bytearray)):
            key = k.decode('utf-8', errors='ignore')
        else:
            key = str(k)
        try:
            vec = emb[key][()]
            added_vecs.append(vec)
        except Exception:
            continue
        if len(added_vecs) >= n_rand:
            break

if not dna_vecs:
    raise SystemExit('No DNA embeddings found for selected SRS.')

dna = np.stack(dna_vecs)
if not added_vecs:
    # Fallback: synthesize noise if we couldn't fetch from other samples
    mu = float(dna.mean())
    sigma = float(dna.std()) if dna.std() > 0 else 1.0
    n_rand_fallback = max(1, int(round(0.15 * dna.shape[0])))
    added_vecs = [np.random.normal(mu, sigma, size=(dna.shape[1],)) for _ in range(n_rand_fallback)]

added = np.stack(added_vecs)
is_original = np.array([True] * dna.shape[0] + [False] * added.shape[0], dtype=bool)
dna_all = np.vstack([dna, added])

# Build model pass capturing per-layer outputs
x1 = torch.tensor(dna_all, dtype=torch.float32, device=device).unsqueeze(0)
x2 = torch.empty((1, 0, shared_utils.TXT_EMB), dtype=torch.float32, device=device)
mask = torch.ones((1, x1.shape[1]), dtype=torch.bool, device=device)

with torch.no_grad():
    h1 = model.input_projection_type1(x1)
    hidden = h1  # no type2
    layer_outputs = []
    for enc in model.transformer.layers:
        hidden = enc(hidden, src_key_padding_mask=~mask)
        layer_outputs.append(hidden.squeeze(0).cpu().numpy())
    # Final logits for coloring
    final_logits = model.output_projection(hidden).squeeze(-1).squeeze(0).cpu().numpy()

# Prepare PCA per layer and for DNA
n_layers = len(layer_outputs)
plots = n_layers + 1
ncols = 3
nrows = (plots + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols + 2, 3.5 * nrows), squeeze=False)

# Match logit colour scheme from paper_umap_logit_colours (coolwarm, red = higher)
cmap = plt.cm.coolwarm
norm = Normalize(vmin=float(final_logits.min()), vmax=float(final_logits.max()))

# Dimensionality reducer
def reduce2d(X):
    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1).fit_transform(X)

# First subplot: DNA (input)
XY0 = reduce2d(dna_all)
ax = axes[0][0]
for flag, marker in ((True, 'o'), (False, 'x')):
    idx = np.where(is_original == flag)[0]
    if idx.size:
        ax.scatter(
            XY0[idx, 0],
            XY0[idx, 1],
            s=(8 if flag else 14),
            alpha=0.85,
            c=cmap(norm(final_logits[idx])),
            marker=marker,
            label=('original' if flag else 'imposter'),
        )
ax.set_title('Input DNA')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Subsequent subplots: each transformer layer output
for li, out in enumerate(layer_outputs, start=1):
    r, c = divmod(li, ncols)
    ax = axes[r][c]
    XY = reduce2d(out)
    for flag, marker in ((True, 'o'), (False, 'x')):
        idx = np.where(is_original == flag)[0]
        if idx.size:
            ax.scatter(
                XY[idx, 0],
                XY[idx, 1],
                s=(8 if flag else 14),
                alpha=0.85,
                c=cmap(norm(final_logits[idx])),
                marker=marker,
            )
    ax.set_title(f'Layer {li}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

# Turn off any unused axes
for pi in range(plots, nrows * ncols):
    r, c = divmod(pi, ncols)
    axes[r][c].axis('off')

# Add a shared colorbar to explain logit color mapping
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Reserve right margin for a dedicated colorbar axis
plt.tight_layout(rect=(0, 0, 0.92, 0.95))

# Dedicated colorbar axis: [left, bottom, width, height] in figure coords
# Move the bar down slightly and shorten it to leave more room for the legend above.
cax = fig.add_axes([0.935, 0.15, 0.02, 0.6])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Final logit (Stability Score)')

# Marker legend on top of the colorbar
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='black', markeredgecolor='black', markersize=4, label='original'),
    Line2D([0], [0], marker='x', linestyle='None', markeredgecolor='black', markersize=4, label='imposter'),
]
fig.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(0.92, 0.88),
    borderaxespad=0.0,
    frameon=False,
)

#fig.suptitle(f'Gingivitis — per-layer OTU embeddings for {rand_srs}\nColor = final logit, marker: o=original, x=added', fontsize=12)
plt.show()

# %%
