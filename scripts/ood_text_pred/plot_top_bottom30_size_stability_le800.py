#!/usr/bin/env python3
"""Plot size vs stability for top/bottom 30 terms (samples with <=800 OTUs).

Creates two side-by-side subplots:
1) Samples carrying any of the top 30 enriched terms
2) Samples carrying any of the top 30 depleted terms

Each plotted point corresponds to a (term, sample) pair.
So if one sample matches multiple terms in a group, it can appear multiple times.
Requires precomputed lookup file from:
  scripts/ood_text_pred/analyze_top_bottom30_term_size_correlation.py
"""

import os
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root on sys.path so `from scripts import utils` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402

LOOKUP_TSV = 'data/ood_text_pred/top_bottom30_term_sample_lookup.tsv'
STAB_RAW = 'data/ood_text_pred/average_otu_logits_notextabl.tsv'
MAPPED = 'data/microbeatlas/samples-otus.97.mapped'

OUT_PNG = 'data/ood_text_pred/top_bottom30_size_vs_stability_le800.png'
OUT_PNG_SIGMOID = 'data/ood_text_pred/top_bottom30_size_vs_sigmoid_stability_le800.png'
OUT_TSV = 'data/ood_text_pred/top_bottom30_size_vs_stability_le800_points.tsv'
RANDOM_TSV = 'data/ood_text_pred/random_otu_size_matched_4000_points.tsv'

MAX_OTU = 800.0
RANDOM_N = 4000
RANDOM_SEED = 42
BATCH_SIZE = 32
USE_ZERO_SCRATCH_TOKENS = True
SCRATCH_TOKENS_PER_SAMPLE = 16


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_stability_from_raw(path):
    s2y = {}
    with open(path, 'r') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            srs = (row.get('sampleid') or '').strip()
            if not srs:
                continue
            try:
                s2y[srs] = float(row['average_logit'])
            except Exception:
                continue
    return s2y


def load_richness_from_mapped(path, needed):
    s2otu = {}
    current = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                header = line[1:].split()[0]
                parts = header.split('.')
                srs = parts[-1] if parts else header
                current = srs if srs in needed else None
                if current and current not in s2otu:
                    s2otu[current] = set()
                continue
            if not current:
                continue
            first = line.split()[0]
            toks = [t for t in first.split(';') if t]
            otu = next((t for t in toks if t.startswith('97_')), toks[-1] if toks else None)
            if otu:
                s2otu[current].add(otu)
    return {s: float(len(v)) for s, v in s2otu.items() if v}


def build_all_samples_xy():
    if not os.path.exists(STAB_RAW):
        raise FileNotFoundError(f'Missing raw stability table: {STAB_RAW}')
    if not os.path.exists(MAPPED):
        raise FileNotFoundError(f'Missing mapped OTU file: {MAPPED}')

    s2y = load_stability_from_raw(STAB_RAW)
    s2x = load_richness_from_mapped(MAPPED, set(s2y))
    all_xy = []
    for srs, y in s2y.items():
        x = s2x.get(srs)
        if x is None or x > MAX_OTU:
            continue
        all_xy.append((srs, x, y))
    return all_xy


def make_resolver(emb_group, rename_map, prefer_domain='B'):
    cache = {}
    prefer = str(prefer_domain or 'B').upper()

    def resolve(new97):
        if new97 in cache:
            return cache[new97]
        candidates = []
        if rename_map:
            a_old = rename_map.get('new97_to_oldA97', {}).get(new97)
            b_old = rename_map.get('new97_to_oldB97', {}).get(new97)
            if prefer == 'B' and b_old:
                candidates.append(b_old)
            if a_old:
                candidates.append(a_old)
            if prefer != 'B' and b_old:
                candidates.append(b_old)
        if str(new97).startswith('97_'):
            num = str(new97).split('_', 1)[1]
            if prefer == 'B':
                candidates.extend([f'B97_{num}', f'A97_{num}'])
            else:
                candidates.extend([f'A97_{num}', f'B97_{num}'])
        candidates.append(new97)
        chosen = None
        for key in candidates:
            if key in emb_group:
                chosen = key
                break
        cache[new97] = chosen
        return chosen

    return resolve


def build_otu_universe(needed_srs):
    s2otu = {}
    universe = set()
    current = None
    with open(MAPPED, 'r', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                header = line[1:].split()[0]
                parts = header.split('.')
                srs = parts[-1] if parts else header
                current = srs if srs in needed_srs else None
                if current and current not in s2otu:
                    s2otu[current] = set()
                continue
            if not current:
                continue
            first = line.split()[0]
            toks = [t for t in first.split(';') if t]
            otu = next((t for t in toks if t.startswith('97_')), toks[-1] if toks else None)
            if not otu:
                continue
            s2otu[current].add(otu)
            universe.add(otu)
    return s2otu, sorted(universe)


def generate_random_panel_points(all_xy):
    if os.path.exists(RANDOM_TSV):
        out = []
        with open(RANDOM_TSV, 'r') as f:
            r = csv.DictReader(f, delimiter='\t')
            for row in r:
                try:
                    out.append((float(row['stability_score']), float(row['otu_richness'])))
                except Exception:
                    continue
        if len(out) == RANDOM_N:
            return out

    if not os.path.exists(STAB_RAW):
        raise FileNotFoundError(f'Missing raw stability table: {STAB_RAW}')
    if not os.path.exists(MAPPED):
        raise FileNotFoundError(f'Missing mapped OTU file: {MAPPED}')

    s2y = load_stability_from_raw(STAB_RAW)
    needed_srs = set(s2y.keys())
    _, otu_universe = build_otu_universe(needed_srs)
    if not otu_universe:
        raise RuntimeError('No OTUs found for random panel generation.')

    rng = np.random.default_rng(RANDOM_SEED)
    sizes = np.asarray([int(x) for _, x in all_xy], dtype=int)
    if sizes.size == 0:
        raise RuntimeError('No size distribution available from all-sample panel.')
    draw_sizes = rng.choice(sizes, size=RANDOM_N, replace=True)

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = (
        shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
        if os.path.exists(shared_utils.RENAME_MAP_PATH)
        else None
    )

    import h5py
    import torch

    results = []
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file['embeddings']
        resolve = make_resolver(emb_group, rename_map)

        resolved_pool = []
        for oid in otu_universe:
            key = resolve(oid)
            if key is not None:
                resolved_pool.append(key)
        resolved_pool = np.asarray(resolved_pool, dtype=object)
        if resolved_pool.size == 0:
            raise RuntimeError('No OTUs could be resolved to embedding keys.')

        batch_sizes = []
        batch_key_lists = []

        def score_batch():
            if not batch_sizes:
                return
            b = len(batch_sizes)
            max_len = max(batch_sizes)
            x1_cpu = torch.zeros((b, max_len, shared_utils.OTU_EMB), dtype=torch.float32)
            mask = torch.zeros((b, max_len), dtype=torch.bool)
            for bi, keys in enumerate(batch_key_lists):
                for j, key in enumerate(keys):
                    x1_cpu[bi, j, :].copy_(torch.from_numpy(emb_group[key][()]).to(torch.float32))
                mask[bi, :len(keys)] = True

            x1 = x1_cpu.to(device)
            mask_dev_otus = mask.to(device)
            with torch.no_grad():
                h1 = model.input_projection_type1(x1)
                if USE_ZERO_SCRATCH_TOKENS and SCRATCH_TOKENS_PER_SAMPLE > 0:
                    z = torch.zeros(
                        (b, SCRATCH_TOKENS_PER_SAMPLE, shared_utils.D_MODEL),
                        dtype=torch.float32,
                        device=device,
                    )
                    h = torch.cat([h1, z], dim=1)
                    mask_dev = torch.ones(
                        (b, max_len + SCRATCH_TOKENS_PER_SAMPLE),
                        dtype=torch.bool,
                        device=device,
                    )
                else:
                    h = h1
                    mask_dev = mask_dev_otus
                h = model.transformer(h, src_key_padding_mask=~mask_dev)
                logits = model.output_projection(h).squeeze(-1)

            for bi, n in enumerate(batch_sizes):
                y = float(logits[bi, :n].mean().item())
                x = float(n)
                results.append((y, x))

            batch_sizes.clear()
            batch_key_lists.clear()

        for n in draw_sizes:
            n = int(max(1, n))
            replace = n > resolved_pool.size
            idx = rng.choice(resolved_pool.size, size=n, replace=replace)
            keys = resolved_pool[idx].tolist()
            batch_sizes.append(n)
            batch_key_lists.append(keys)
            if len(batch_sizes) >= BATCH_SIZE:
                score_batch()
        score_batch()

    os.makedirs(os.path.dirname(RANDOM_TSV), exist_ok=True)
    with open(RANDOM_TSV, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['point_id', 'otu_richness', 'stability_score'])
        for i, (y, x) in enumerate(results, 1):
            w.writerow([i, f'{x:.0f}', f'{y:.6f}'])
    return results


def main():
    if not os.path.exists(LOOKUP_TSV):
        raise FileNotFoundError(
            f'Missing lookup input: {LOOKUP_TSV}. '
            'Run analyze_top_bottom30_term_size_correlation.py first.'
        )

    # Build point table directly from precomputed lookup:
    # one point per (group, term, sample) with richness<=MAX_OTU
    points = []
    with open(LOOKUP_TSV, 'r') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            try:
                grp = (row.get('group') or '').strip()
                term = (row.get('term') or '').strip()
                srs = (row.get('sampleid') or '').strip()
                x = float(row['otu_richness'])
                y = float(row['stability_score'])
            except Exception:
                continue
            if grp not in {'top_enriched_30', 'bottom_depleted_30'}:
                continue
            if not term or not srs:
                continue
            if x > MAX_OTU:
                continue
            points.append((grp, term, srs, x, y))

    if not points:
        raise SystemExit('No points available after filtering.')

    # Save filtered term points snapshot used in plot.
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    with open(OUT_TSV, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['group', 'term', 'sampleid', 'otu_richness', 'stability_score'])
        for g, t, s, x, y in points:
            w.writerow([g, t, s, f'{x:.0f}', f'{y:.6f}'])
    print('Saved points TSV:', OUT_TSV, '| rows:', len(points))

    # store as (stability, richness) for plotting with stability on x-axis
    top_xy = [(y, x) for g, t, s, x, y in points if g == 'top_enriched_30']
    bot_xy = [(y, x) for g, t, s, x, y in points if g == 'bottom_depleted_30']
    all_xy = [(y, x) for _, x, y in build_all_samples_xy()]

    rand_xy = generate_random_panel_points(all_xy)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), sharex=True, sharey=True)

    # Left panel: top 30 enriched terms
    if top_xy:
        x = np.asarray([p[0] for p in top_xy], dtype=float)  # stability
        y = np.asarray([p[1] for p in top_xy], dtype=float)  # richness
        axes[0].scatter(x, y, s=8, alpha=0.22, color='#d62728', edgecolors='none')
        axes[0].set_title(f'Top 30 Enriched Terms (n={len(top_xy)} points)')
    else:
        axes[0].set_title('Top 30 Enriched Terms (n=0)')
    axes[0].grid(True, alpha=0.25)

    # Right panel: bottom 30 depleted terms
    if bot_xy:
        x = np.asarray([p[0] for p in bot_xy], dtype=float)  # stability
        y = np.asarray([p[1] for p in bot_xy], dtype=float)  # richness
        axes[1].scatter(x, y, s=8, alpha=0.22, color='#1f77b4', edgecolors='none')
        axes[1].set_title(f'Bottom 30 Depleted Terms (n={len(bot_xy)} points)')
    else:
        axes[1].set_title('Bottom 30 Depleted Terms (n=0)')
    axes[1].grid(True, alpha=0.25)

    # Third panel: all samples at <=800 richness
    if all_xy:
        x = np.asarray([p[0] for p in all_xy], dtype=float)  # stability
        y = np.asarray([p[1] for p in all_xy], dtype=float)  # richness
        axes[2].scatter(x, y, s=8, alpha=0.15, color='#4d4d4d', edgecolors='none')
        axes[2].set_title(f'All Samples <=800 OTUs (n={len(all_xy)} samples)')
    else:
        axes[2].set_title('All Samples <=800 OTUs (n=0)')
    axes[2].grid(True, alpha=0.25)

    # Fourth panel: random OTU communities size-matched to all-sample distribution
    if rand_xy:
        x = np.asarray([p[0] for p in rand_xy], dtype=float)  # stability
        y = np.asarray([p[1] for p in rand_xy], dtype=float)  # richness
        axes[3].scatter(x, y, s=8, alpha=0.15, color='#2ca02c', edgecolors='none')
        axes[3].set_title(f'Random OTU Sets (n={len(rand_xy)} points)')
    else:
        axes[3].set_title('Random OTU Sets (n=0)')
    axes[3].grid(True, alpha=0.25)

    for ax in axes:
        ax.set_ylim(0, MAX_OTU)
        ax.set_xlabel('Stability score (average OTU logit)')
    axes[0].set_ylabel('OTU count (<= 800)')

    fig.suptitle('Stability vs OTU count (<=800) for Top/Bottom Terms and All Samples')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=350)
    print('Saved figure:', OUT_PNG)

    # Sigmoid-transformed stability version
    fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5.2), sharex=True, sharey=True)

    if top_xy:
        x = sigmoid([p[0] for p in top_xy])
        y = np.asarray([p[1] for p in top_xy], dtype=float)
        axes2[0].scatter(x, y, s=8, alpha=0.22, color='#d62728', edgecolors='none')
        axes2[0].set_title(f'Top 30 Enriched Terms (n={len(top_xy)} points)')
    else:
        axes2[0].set_title('Top 30 Enriched Terms (n=0)')
    axes2[0].grid(True, alpha=0.25)

    if bot_xy:
        x = sigmoid([p[0] for p in bot_xy])
        y = np.asarray([p[1] for p in bot_xy], dtype=float)
        axes2[1].scatter(x, y, s=8, alpha=0.22, color='#1f77b4', edgecolors='none')
        axes2[1].set_title(f'Bottom 30 Depleted Terms (n={len(bot_xy)} points)')
    else:
        axes2[1].set_title('Bottom 30 Depleted Terms (n=0)')
    axes2[1].grid(True, alpha=0.25)

    if all_xy:
        x = sigmoid([p[0] for p in all_xy])
        y = np.asarray([p[1] for p in all_xy], dtype=float)
        axes2[2].scatter(x, y, s=8, alpha=0.15, color='#4d4d4d', edgecolors='none')
        axes2[2].set_title(f'All Samples <=800 OTUs (n={len(all_xy)} samples)')
    else:
        axes2[2].set_title('All Samples <=800 OTUs (n=0)')
    axes2[2].grid(True, alpha=0.25)

    if rand_xy:
        x = sigmoid([p[0] for p in rand_xy])
        y = np.asarray([p[1] for p in rand_xy], dtype=float)
        axes2[3].scatter(x, y, s=8, alpha=0.15, color='#2ca02c', edgecolors='none')
        axes2[3].set_title(f'Random OTU Sets (n={len(rand_xy)} points)')
    else:
        axes2[3].set_title('Random OTU Sets (n=0)')
    axes2[3].grid(True, alpha=0.25)

    for ax in axes2:
        ax.set_ylim(0, MAX_OTU)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Sigmoid(stability score)')
    axes2[0].set_ylabel('OTU count (<= 800)')

    fig2.suptitle('Sigmoid(stability) vs OTU count (<=800) for Top/Bottom Terms and All Samples')
    plt.tight_layout()
    plt.savefig(OUT_PNG_SIGMOID, dpi=350)
    print('Saved figure:', OUT_PNG_SIGMOID)


if __name__ == '__main__':
    main()
