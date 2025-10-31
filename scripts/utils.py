import csv
import os
import sys
import re

import h5py
import torch
from tqdm import tqdm

# Ensure project root is on sys.path so we can import model.py from the repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import MicrobiomeTransformer


# Default paths shared across tasks
MICROBEATLAS_SAMPLES = 'data/diabimmune/microbeatlas_samples.tsv'
MAPPED_PATH = 'data/microbeatlas/samples-otus.97.mapped'
CHECKPOINT_PATH = 'data/model/checkpoint_epoch_0_final_epoch3_conf00.pt'
PROKBERT_PATH = 'data/model/prokbert_embeddings.h5'
RENAME_MAP_PATH = 'data/microbeatlas/otus.rename.map1'

# Model hyperparameters (shared)
D_MODEL = 100
NHEAD = 5
NUM_LAYERS = 5
DROPOUT = 0.1
DIM_FF = 400
OTU_EMB = 384
TXT_EMB = 1536


# Note: MicrobeAtlas run→SRS mapping is DIABIMMUNE-specific and implemented in
# scripts/diabimmune/utils.py to keep shared utils dataset-agnostic.


def collect_micro_to_otus_mapped(needed_srs, mapped_path=MAPPED_PATH):
    """
    Stream the large mapped MicrobeAtlas file and build SRS -> list of OTU ids.
    The mapped file blocks start with lines like:
        >SRR2459896.SRS1074972    66481   23845   497
    followed by lines like:
        90_246;96_8626;97_10374   4920
    We take the 97%-identity token (e.g., '97_10374') as the OTU id to match
    ProkBERT embedding keys (which often use 97%-level clusters, sometimes
    with an 'A97_' prefix in the embedding file).
    Only SRS IDs present in needed_srs are collected.
    """
    needed_srs = set(needed_srs)
    micro_to_otus = {}
    current_srs = None
    if not os.path.exists(mapped_path):
        raise FileNotFoundError(f"Mapped file not found: {mapped_path}")
    with open(mapped_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                header = line[1:].split()[0]  # e.g., SRR... .SRS...
                parts = header.split('.')
                srs = parts[-1] if parts else header
                current_srs = srs if srs in needed_srs else None
                if current_srs is not None and current_srs not in micro_to_otus:
                    micro_to_otus[current_srs] = []
                continue
            if current_srs is None:
                continue
            # Parse OTU triplet and take the 97_* component
            first_field = line.split()[0]
            parts = first_field.split(';')
            otu97 = None
            for tok in parts:
                if tok.startswith('97_'):
                    otu97 = tok
                    break
            if otu97 is None and parts:
                # Fallback: if 97_ not present, keep the first token
                otu97 = parts[-1]
            micro_to_otus[current_srs].append(otu97)

    # Deduplicate OTUs per SRS while preserving some order
    for k, v in list(micro_to_otus.items()):
        seen = set()
        deduped = []
        for oid in v:
            if oid not in seen:
                seen.add(oid)
                deduped.append(oid)
        micro_to_otus[k] = deduped

    print('otus (mapped) for', len(micro_to_otus), 'samples')
    missing = needed_srs - set(micro_to_otus)
    if missing:
        print('missing SRS in mapped file:', len(missing))
    if micro_to_otus:
        ex = next(iter(micro_to_otus))
        print('example', ex, 'otus:', micro_to_otus[ex][:5])
    return micro_to_otus


def build_accession_to_srs_from_mapped(mapped_path=MAPPED_PATH):
    """
    Build a mapping from accession prefix (e.g., SRR/ERR/ERS/DRR...) to SRS by
    scanning only header lines in the mapped file. Efficient streaming parse.
    Header example:
        >SRR2459896.SRS1074972    66481   23845   497
    We map 'SRR2459896' -> 'SRS1074972'. Works for ERR/ERS too.
    """
    if not os.path.exists(mapped_path):
        raise FileNotFoundError(f"Mapped file not found: {mapped_path}")
    acc_to_srs = {}
    with open(mapped_path, 'r', errors='replace') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            header = line[1:].split()[0]  # e.g., SRR... .SRS...
            parts = header.split('.')
            if len(parts) < 2:
                continue
            acc = parts[0]
            srs = parts[-1]
            acc_to_srs[acc] = srs
    return acc_to_srs


def load_microbiome_model(checkpoint_path=CHECKPOINT_PATH):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    model = MicrobiomeTransformer(
        input_dim_type1=OTU_EMB,
        input_dim_type2=TXT_EMB,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT
    )

    load_info = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print('model ready on', device)
    # Attach load info for debugging downstream
    try:
        model._load_info = load_info
        if hasattr(load_info, 'missing_keys') and load_info.missing_keys:
            mk = load_info.missing_keys
            print('missing checkpoint keys (first):', mk[:5] if len(mk) > 5 else mk)
        if hasattr(load_info, 'unexpected_keys') and load_info.unexpected_keys:
            uk = load_info.unexpected_keys
            print('unexpected checkpoint keys (first):', uk[:5] if len(uk) > 5 else uk)
    except Exception:
        pass
    return model, device


def preview_prokbert_embeddings(prokbert_path=PROKBERT_PATH, limit=10):
    with h5py.File(prokbert_path) as emb_file:
        embedding_group = emb_file['embeddings']
        example_ids = []
        for key in embedding_group.keys():
            example_ids.append(key)
            if len(example_ids) == limit:
                break
        print('total prokbert embeddings:', len(embedding_group))
        print('example embedding ids:', example_ids)
    return example_ids


def build_sample_embeddings(
    micro_to_otus,
    model,
    device,
    prokbert_path=PROKBERT_PATH,
    txt_emb=TXT_EMB,
    rename_map=None,
    resolver=None,
):
    sample_embeddings = {}
    missing_otus = 0

    with h5py.File(prokbert_path) as emb_file:
        embedding_group = emb_file['embeddings']
        for sample_key, otu_list in tqdm(micro_to_otus.items(), desc='Embedding samples'):
            otu_vectors = []
            for otu_id in otu_list:
                candidates = []
                # Prefer an exact resolver mapping if provided
                if resolver and otu_id in resolver:
                    candidates.append(resolver[otu_id])
                # Fallback to the raw id as a last resort
                candidates.append(otu_id)
                key_found = None
                for key in candidates:
                    if key in embedding_group:
                        key_found = key
                        break
                if key_found is not None:
                    vec = embedding_group[key_found][()]
                    otu_vectors.append(torch.tensor(vec, dtype=torch.float32, device=device))
                else:
                    missing_otus += 1
            if not otu_vectors:
                continue
            otu_tensor = torch.stack(otu_vectors, dim=0).unsqueeze(0)
            type2_tensor = torch.zeros((1, 0, txt_emb), dtype=torch.float32, device=device)
            mask = torch.ones((1, otu_tensor.shape[1]), dtype=torch.bool, device=device)
            with torch.no_grad():
                hidden_type1 = model.input_projection_type1(otu_tensor)
                hidden_type2 = model.input_projection_type2(type2_tensor)
                combined_hidden = torch.cat([hidden_type1, hidden_type2], dim=1)
                hidden = model.transformer(combined_hidden, src_key_padding_mask=~mask)
                sample_vec = hidden.mean(dim=1).squeeze(0).cpu()
            sample_embeddings[sample_key] = sample_vec

    print('sample embeddings ready:', len(sample_embeddings))
    print('missing otu embeddings:', missing_otus)
    if sample_embeddings:
        first_key = next(iter(sample_embeddings))
        print('example sample embedding', first_key, sample_embeddings[first_key][:5])
    return sample_embeddings, missing_otus


def load_otu_rename_map(path=RENAME_MAP_PATH, delimiter='\t'):
    """
    Load OTU rename pairs from mapping file.
    Returns a dict with both directions: {'old_to_new': {...}, 'new_to_old': {...}}
    """
    old_to_new = {}
    new_to_old = {}
    oldA97_to_new97 = {}
    new97_to_oldA97 = {}
    oldB97_to_new97 = {}
    new97_to_oldB97 = {}
    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Prefer tab-separated; fallback to any whitespace
            if delimiter and delimiter in line:
                parts = [p for p in line.split(delimiter) if p]
            else:
                parts = line.split()
            if len(parts) < 2:
                continue
            old_id, new_id = parts[0], parts[1]
            old_to_new[old_id] = new_id
            # Right side may contain semicolon-separated components like
            #  "90_x;96_y;97_z". Index each token to the same old_id so lookups
            #  by a single new token (e.g., '97_z') succeed.
            for token in new_id.split(';'):
                token = token.strip()
                if not token:
                    continue
                new_to_old[token] = old_id
            # Build 97-level A/B maps
            left_tokens = [t for t in old_id.split(';') if t]
            right_tokens = [t for t in new_id.split(';') if t]
            left_head = left_tokens[0] if left_tokens else ''
            left97 = next((t for t in left_tokens if t.startswith('97_')), None)
            right97 = next((t for t in right_tokens if t.startswith('97_')), None)
            if left97 and right97:
                num = left97.split('_', 1)[1]
                if left_head.startswith('A'):
                    oldA97_to_new97[f'A97_{num}'] = right97
                    new97_to_oldA97[right97] = f'A97_{num}'
                if left_head.startswith('B'):
                    oldB97_to_new97[f'B97_{num}'] = right97
                    new97_to_oldB97[right97] = f'B97_{num}'
    return {
        'old_to_new': old_to_new,
        'new_to_old': new_to_old,
        'oldA97_to_new97': oldA97_to_new97,
        'new97_to_oldA97': new97_to_oldA97,
        'oldB97_to_new97': oldB97_to_new97,
        'new97_to_oldB97': new97_to_oldB97,
    }


def build_otu_key_resolver(micro_to_otus, rename_map, prokbert_path=PROKBERT_PATH, prefer='B'):
    """
    Build an exact mapping from new 97_* ids to the actual embedding key present
    in the HDF5 (e.g., A97_1234 or B97_1234). If both exist, prefer the given
    domain ('A' or 'B').
    Returns: dict new97 -> resolved_key, and prints a small coverage summary.
    """
    if not rename_map:
        return {}
    # Collect union of 97_* OTUs across samples
    new97_ids = set()
    for lst in micro_to_otus.values():
        for oid in lst:
            if isinstance(oid, str) and oid.startswith('97_'):
                new97_ids.add(oid)
    resolver = {}
    with h5py.File(prokbert_path) as f:
        emb = f['embeddings']
        hits = misses = both = 0
        for oid in new97_ids:
            a = rename_map.get('new97_to_oldA97', {}).get(oid)
            b = rename_map.get('new97_to_oldB97', {}).get(oid)
            a_ok = a in emb if a else False
            b_ok = b in emb if b else False
            chosen = None
            if a_ok and b_ok:
                both += 1
                chosen = a if prefer.upper() == 'A' else b
            elif a_ok:
                chosen = a
            elif b_ok:
                chosen = b
            elif oid in emb:
                chosen = oid
            if chosen:
                resolver[oid] = chosen
                hits += 1
            else:
                misses += 1
    total = len(new97_ids)
    if total:
        rate = 100.0 * hits / total
        print(f'OTU key resolver: {hits}/{total} mapped ({rate:.1f}%), both={both}, misses={misses}, prefer={prefer}')
    return resolver


def translate_otu_ids(otu_ids, rename_map, direction='new_to_old'):
    """
    Translate a list of OTU ids using provided mapping.
    direction: 'new_to_old' (default) or 'old_to_new'. Unmapped IDs are returned unchanged.
    """
    if not rename_map:
        return list(otu_ids)
    mapper = rename_map.get(direction, {})
    return [mapper.get(oid, oid) for oid in otu_ids]


def score_otus_for_srs(srs, micro_to_otus, resolver, model, device, emb_group, txt_emb=TXT_EMB):
    """
    Compute raw per-OTU logits for a single SRS.
    - Looks up OTU vectors via resolver (exact HDF5 embedding keys) with fallback to raw ID
    - Runs input_projection_type1 → transformer → output_projection
    Returns dict {otu_id: logit}
    """
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
    x2 = torch.empty((1, 0, txt_emb), dtype=torch.float32, device=device)
    mask = torch.ones((1, x1.shape[1]), dtype=torch.bool, device=device)
    with torch.no_grad():
        h = model.input_projection_type1(x1)
        h = model.transformer(h, src_key_padding_mask=~mask)
        logits = model.output_projection(h).squeeze(-1).squeeze(0).cpu().numpy()
    return dict(zip(keep, logits))


def union_average_logits(per_sample_dicts):
    """
    Given a list of {otu_id: logit} dicts (e.g., per-SRS), return a single dict
    where each OTU's value is the mean of available logits across samples.
    """
    if not per_sample_dicts:
        return {}
    union = set().union(*[d.keys() for d in per_sample_dicts])
    out = {}
    for oid in union:
        vals = [d[oid] for d in per_sample_dicts if oid in d]
        if vals:
            out[oid] = float(sum(vals) / len(vals))
    return out
