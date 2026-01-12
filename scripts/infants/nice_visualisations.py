#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.infants.utils import (
    load_infants_meta,
    load_infants_otus_tsv,
)
from scripts import utils as shared_utils


#%% Load metadata and OTUs (simple, linear)
meta = load_infants_meta('data/infants/meta_withbirth.csv')
print('loaded meta entries:', len(meta))

sample_to_otus = load_infants_otus_tsv('data/infants/infants_otus.tsv')
print('otus rows:', len(sample_to_otus))


#%% Build frozen per-sample embeddings using exact key resolver
model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
resolver = shared_utils.build_otu_key_resolver(sample_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B') if rename_map else {}
sample_embeddings, _ = shared_utils.build_sample_embeddings(
    sample_to_otus,
    model,
    device,
    prokbert_path=shared_utils.PROKBERT_PATH,
    txt_emb=shared_utils.TXT_EMB,
    rename_map=rename_map,
    resolver=resolver,
)
print('embeddings built:', len(sample_embeddings))


#%% Assemble X and labels
def parse_env_label(label: str):
    m = re.match(r'^([0-9]+M|[0-9]+Y|B|M)\((C|V)\)$', label.strip())
    if not m:
        return label, ''
    return m.group(1), m.group(2)

X_list = []
age_labels = []
mode_labels = []
sample_ids = []
kept = 0
for sid, label in meta.items():
    emb = sample_embeddings.get(sid)
    if emb is None:
        continue
    age, mode = parse_env_label(label)
    X_list.append(emb.numpy())
    age_labels.append(age)
    mode_labels.append(mode)
    sample_ids.append(sid)
    kept += 1

print('usable samples for viz:', kept)
if kept == 0:
    raise SystemExit('No usable samples for visualisation.')

# Build presence/absence matrix: samples x OTUs (0/1)
all_otus = sorted({otu for sid in sample_ids for otu in sample_to_otus.get(sid, [])})
print('unique OTUs across usable samples:', len(all_otus))
otu_to_idx = {otu: i for i, otu in enumerate(all_otus)}
presence_X = np.zeros((kept, len(all_otus)), dtype=np.float32)
for row_idx, sid in enumerate(sample_ids):
    for otu in sample_to_otus.get(sid, []):
        col_idx = otu_to_idx.get(otu)
        if col_idx is not None:
            presence_X[row_idx, col_idx] = 1.0

X = np.stack(X_list)


#%% UMAP reduction to 2D/3D
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=43, metric='cosine')
X_umap = reducer.fit_transform(X)
X2 = X_umap[:, :2]
X3 = X_umap[:, :3]


#%% Helper for coloring and ordering
def age_sort_key(val: str):
    order = {'B': 0, '4M': 1, '12M': 2, '3Y': 3, '5Y': 4, 'M': 5}
    return order.get(val, 999)

uniq_ages = sorted(sorted(set(age_labels)), key=age_sort_key)
age_to_color = {age: plt.cm.tab20(i % 20) for i, age in enumerate(uniq_ages)}

uniq_modes = ['C', 'V']
mode_to_color = {'C': '#1f77b4', 'V': '#ff7f0e'}


#%% 2D scatter: colored by age
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for age in uniq_ages:
    idx = [i for i, a in enumerate(age_labels) if a == age]
    if not idx:
        continue
    ax.scatter(X2[idx, 0], X2[idx, 1], s=6, alpha=0.8, c=[age_to_color[age]], label=age)
#ax.set_title('UMAP 2D — colored by age')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% 3D scatter: colored by age
fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
for age in uniq_ages:
    idx = [i for i, a in enumerate(age_labels) if a == age]
    if not idx:
        continue
    ax1.scatter(X3[idx, 0], X3[idx, 1], X3[idx, 2], s=4, alpha=0.7, c=[age_to_color[age]], label=age)
ax1.set_title('UMAP 3D — colored by age')
ax1.set_xlabel('UMAP1')
ax1.set_ylabel('UMAP2')
ax1.set_zlabel('UMAP3')
ax1.legend(markerscale=1.2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% 2D scatter: colored by delivery mode (C vs V)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for mode in uniq_modes:
    idx = [i for i, m in enumerate(mode_labels) if m == mode]
    if not idx:
        continue
    ax.scatter(X2[idx, 0], X2[idx, 1], s=6, alpha=0.8, c=[mode_to_color[mode]], label=mode)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.legend(title='Mode', markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% 3D scatter: colored by delivery mode (C vs V)
fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
for mode in uniq_modes:
    idx = [i for i, m in enumerate(mode_labels) if m == mode]
    if not idx:
        continue
    ax1.scatter(X3[idx, 0], X3[idx, 1], X3[idx, 2], s=4, alpha=0.7, c=[mode_to_color[mode]], label=mode)
ax1.set_title('UMAP 3D — colored by delivery mode')
ax1.set_xlabel('UMAP1')
ax1.set_ylabel('UMAP2')
ax1.set_zlabel('UMAP3')
ax1.legend(title='Mode', markerscale=1.2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% 2D UMAP on presence/absence OTU vectors (colored by age)
reducer_pa_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,metric='cosine', random_state=43)
X_pa_2d = reducer_pa_2d.fit_transform(presence_X)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for age in uniq_ages:
    idx = [i for i, a in enumerate(age_labels) if a == age]
    if not idx:
        continue
    ax.scatter(X_pa_2d[idx, 0], X_pa_2d[idx, 1], s=6, alpha=0.8, c=[age_to_color[age]], label=age)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% 3D UMAP on presence/absence OTU vectors (colored by age)
reducer_pa_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=43)
X_pa_3d = reducer_pa_3d.fit_transform(presence_X)

fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
for age in uniq_ages:
    idx = [i for i, a in enumerate(age_labels) if a == age]
    if not idx:
        continue
    ax1.scatter(X_pa_3d[idx, 0], X_pa_3d[idx, 1], X_pa_3d[idx, 2], s=4, alpha=0.7, c=[age_to_color[age]], label=age)
ax1.set_title('UMAP 3D — presence/absence (age)')
ax1.set_xlabel('UMAP1')
ax1.set_ylabel('UMAP2')
ax1.set_zlabel('UMAP3')
ax1.legend(markerscale=1.2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% 2D UMAP on presence/absence OTU vectors (colored by delivery mode C vs V)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
for mode in uniq_modes:
    idx = [i for i, m in enumerate(mode_labels) if m == mode]
    if not idx:
        continue
    ax.scatter(X_pa_2d[idx, 0], X_pa_2d[idx, 1], s=6, alpha=0.8, c=[mode_to_color[mode]], label=mode)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.legend(title='Mode', markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% 3D UMAP on presence/absence OTU vectors (colored by delivery mode C vs V)
fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
for mode in uniq_modes:
    idx = [i for i, m in enumerate(mode_labels) if m == mode]
    if not idx:
        continue
    ax1.scatter(X_pa_3d[idx, 0], X_pa_3d[idx, 1], X_pa_3d[idx, 2], s=4, alpha=0.7, c=[mode_to_color[mode]], label=mode)
ax1.set_title('UMAP 3D — presence/absence (delivery mode)')
ax1.set_xlabel('UMAP1')
ax1.set_ylabel('UMAP2')
ax1.set_zlabel('UMAP3')
ax1.legend(title='Mode', markerscale=1.2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%
