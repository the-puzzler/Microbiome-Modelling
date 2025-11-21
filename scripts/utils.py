import csv
import os
import sys
import re

import h5py
import torch
import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path so we can import model.py from the repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import MicrobiomeTransformer


# Default paths shared across tasks
MICROBEATLAS_SAMPLES = 'data/diabimmune/microbeatlas_samples.tsv'
MAPPED_PATH = 'data/microbeatlas/samples-otus.97.mapped'
CHECKPOINT_PATH = 'data/model/checkpoint_epoch_0_final_newblack_2epoch.pt'
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
    # Prefer CUDA, else CPU (disable MPS due to performance)
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
    srs_to_terms=None,
    term_to_vec=None,
    include_text: bool = False,
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
            # Optional text (type2) tokens
            type2_tensor = torch.zeros((1, 0, txt_emb), dtype=torch.float32, device=device)
            if include_text and srs_to_terms and term_to_vec is not None:
                terms = [t for t in sorted(srs_to_terms.get(sample_key, set())) if t in term_to_vec]
                if terms:
                    type2_tensor = torch.stack([term_to_vec[t] for t in terms], dim=0).unsqueeze(0)
            mask = torch.ones((1, otu_tensor.shape[1] + type2_tensor.shape[1]), dtype=torch.bool, device=device)
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


def score_otus_for_srs_with_text(
    srs,
    micro_to_otus,
    resolver,
    model,
    device,
    emb_group,
    term_to_vec,
    srs_to_terms,
    txt_emb=TXT_EMB,
):
    """
    Compute per-OTU logits for a single SRS including text tokens (type2) if available.
    Concatenates type1 (OTU) and type2 (text) projections, runs transformer and slices
    logits for the type1 positions.
    Returns dict {otu_id: logit}.
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
    # Build type2 (text) tensor from provided mappings
    if srs_to_terms is not None and term_to_vec is not None:
        t_terms = [t for t in sorted(srs_to_terms.get(srs, set())) if t in term_to_vec]
        x2 = torch.stack([term_to_vec[t] for t in t_terms], dim=0).unsqueeze(0) if t_terms else torch.zeros((1, 0, txt_emb), dtype=torch.float32, device=device)
    else:
        x2 = torch.zeros((1, 0, txt_emb), dtype=torch.float32, device=device)
    n1 = x1.shape[1]
    mask = torch.ones((1, n1 + x2.shape[1]), dtype=torch.bool, device=device)
    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        h2 = model.input_projection_type2(x2)
        h = torch.cat([h1, h2], dim=1)
        h = model.transformer(h, src_key_padding_mask=~mask)
        logits = model.output_projection(h).squeeze(-1)
    logits_type1 = logits[:, :n1].squeeze(0).cpu().numpy()
    return dict(zip(keep, logits_type1))


def score_otu_list(
    otu_ids,
    resolver,
    model,
    device,
    emb_group,
    txt_emb=TXT_EMB,
):
    """
    Score an arbitrary list of OTU ids without text tokens.
    Returns dict {otu_id: logit}.
    """
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
    # Empty type2 to align with model forward expectations
    x2 = torch.zeros((1, 0, txt_emb), dtype=torch.float32, device=device)
    mask = torch.ones((1, x1.shape[1] + x2.shape[1]), dtype=torch.bool, device=device)
    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        h2 = model.input_projection_type2(x2)
        h = torch.cat([h1, h2], dim=1)
        h = model.transformer(h, src_key_padding_mask=~mask)
        logits = model.output_projection(h).squeeze(-1).squeeze(0).cpu().numpy()
    return dict(zip(keep, logits))

def score_otu_list_with_text(
    srs,
    otu_ids,
    resolver,
    model,
    device,
    emb_group,
    term_to_vec,
    srs_to_terms,
    txt_emb=TXT_EMB,
):
    """
    Score an arbitrary list of OTU ids for a given SRS with text tokens included.
    Returns dict {otu_id: logit} for the provided list (order preserved in output slice).

    This is like score_otus_for_srs_with_text, but takes an explicit otu_ids list
    (useful for colonisation experiments where we add a target OTU to T1).
    """
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
    # Build type2 (text) tensor
    t_terms = [t for t in sorted(srs_to_terms.get(srs, set())) if t in term_to_vec] if srs_to_terms is not None else []
    x2 = (torch.stack([term_to_vec[t] for t in t_terms], dim=0).unsqueeze(0)
          if t_terms else torch.zeros((1, 0, txt_emb), dtype=torch.float32, device=device))
    n1 = x1.shape[1]
    mask = torch.ones((1, n1 + x2.shape[1]), dtype=torch.bool, device=device)
    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        h2 = model.input_projection_type2(x2)
        h = torch.cat([h1, h2], dim=1)
        h = model.transformer(h, src_key_padding_mask=~mask)
        logits = model.output_projection(h).squeeze(-1)
    logits_type1 = logits[:, :n1].squeeze(0).cpu().numpy()
    return dict(zip(keep, logits_type1))

def load_term_embeddings(pkl_path= 'data/microbeatlas/word_embeddings_dany_biomes_combined_dany_og_biome_tech.pkl', device=None):
    """
    Load term -> embedding mapping from pickle. Lowercases terms and returns torch tensors on device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import pickle as _pkl
    with open(pkl_path, 'rb') as f:
        obj = _pkl.load(f)
    return {str(k).lower(): torch.tensor(np.asarray(v), dtype=torch.float32, device=device) for k, v in obj.items()}


def parse_run_terms(tsv_path='data/microbeatlas/sample_terms_mapping_combined_dany_og_biome_tech.txt'):
    """
    Parse a tab-separated RunID -> terms mapping file.
    Returns dict: run_id -> [term, ...]
    """
    run_to_terms = {}
    with open(tsv_path, 'r', errors='replace') as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts:
                continue
            rid = parts[0].strip()
            terms = [t.strip().lower() for t in parts[1:] if t and t.strip()]
            if rid and terms:
                run_to_terms[rid] = terms
    return run_to_terms


def build_srs_terms(run_to_srs, run_to_terms, mapped_path=MAPPED_PATH):
    """
    Build SRS -> union of terms by expanding all accessions that resolve to the same SRS
    using the mapped MicrobeAtlas headers.
    Args:
        run_to_srs: dict run/accession -> SRS
        run_to_terms: dict run/accession -> list[str]
    Returns dict srs -> set[str]
    """
    acc_to_srs = build_accession_to_srs_from_mapped(mapped_path)
    srs_to_accs = {}
    for acc, srs in acc_to_srs.items():
        srs_to_accs.setdefault(srs, []).append(acc)
    # ensure direct mapping runs are included
    for acc, srs in run_to_srs.items():
        srs_to_accs.setdefault(srs, []).append(acc)
    srs_to_terms = {}
    for srs, accs in srs_to_accs.items():
        terms = set()
        for acc in accs:
            if acc in run_to_terms:
                terms.update(run_to_terms[acc])
        if terms:
            srs_to_terms[srs] = terms
    return srs_to_terms


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


# ==== Shared helpers for baselines (presence matrices, multilabel pairs, evaluation) ====

def union_presence(srs_list, micro_to_otus):
    """
    Union OTUs across a list of SRS IDs.
    Returns a set of OTU ids.
    """
    seen = set()
    for srs in srs_list:
        for otu in micro_to_otus.get(srs, []):
            seen.add(otu)
    return seen


def build_presence_matrix(group_to_srs, micro_to_otus, min_prevalence=10):
    """
    Build a binary presence/absence matrix X over groups from SRS→OTUs mapping.

    Args:
        group_to_srs: dict key(any hashable)->list[SRS]
        micro_to_otus: dict SRS->list[OTU]
        min_prevalence: keep OTUs observed in at least this many groups

    Returns:
        X: np.ndarray [n_groups, n_otus]
        kept_otus: list[str]
        otu_index: dict OTU->col
        keys: list of group keys in row order
        presence_by_key: dict key->set[OTU]
        key_to_row: dict key->row index in X
    """
    from collections import Counter
    import numpy as _np

    presence_by_key = {k: union_presence(srs_list, micro_to_otus) for k, srs_list in group_to_srs.items()}
    prev = Counter()
    for pres in presence_by_key.values():
        prev.update(pres)
    kept_otus = sorted([otu for otu, c in prev.items() if c >= min_prevalence])
    otu_index = {otu: i for i, otu in enumerate(kept_otus)}

    keys = sorted(presence_by_key.keys())
    X = _np.zeros((len(keys), len(kept_otus)), dtype=_np.float32)
    for i, key in enumerate(keys):
        for otu in presence_by_key[key]:
            j = otu_index.get(otu)
            if j is not None:
                X[i, j] = 1.0
    key_to_row = {k: i for i, k in enumerate(keys)}
    return X, kept_otus, otu_index, keys, presence_by_key, key_to_row


def build_multilabel_pairs(keys, presence_by_key, kept_otus, key_to_row, mode, group_id_func):
    """
    Build masked multilabel targets for t1->t2 transitions within groups.

    Args:
        keys: list of group keys (rows correspond to these in X)
        presence_by_key: dict key->set[OTU]
        kept_otus: list of OTU column ids
        key_to_row: dict key->row index
        mode: 'dropout' or 'colonisation'
        group_id_func: callable(key)->group id (e.g., subject or (block,treatment))

    Returns:
        X_rows: list[int] row indices pointing into X
        Y_ml: np.ndarray [n_pairs, n_otus]
        M_ml: np.ndarray [n_pairs, n_otus] (bool mask of eligible positions)
        groups: np.ndarray [n_pairs] of group ids for CV grouping
    """
    import numpy as _np
    from collections import defaultdict as _dd

    assert mode in {"dropout", "colonisation"}
    by_group = _dd(list)
    for k in keys:
        by_group[group_id_func(k)].append(k)

    X_rows, Y_rows, M_rows, groups = [], [], [], []
    for gid, klist in by_group.items():
        for i in range(len(klist)):
            for j in range(len(klist)):
                if i == j:
                    continue
                k1, k2 = klist[i], klist[j]
                pres1 = presence_by_key.get(k1, set())
                pres2 = presence_by_key.get(k2, set())
                yrow = _np.zeros(len(kept_otus), dtype=_np.int64)
                mrow = _np.zeros(len(kept_otus), dtype=bool)
                for idx, otu in enumerate(kept_otus):
                    if mode == 'dropout':
                        if otu in pres1:
                            mrow[idx] = True
                            yrow[idx] = 1 if otu not in pres2 else 0
                    else:  # colonisation
                        if otu not in pres1:
                            mrow[idx] = True
                            yrow[idx] = 1 if otu in pres2 else 0
                X_rows.append(key_to_row[k1])
                Y_rows.append(yrow)
                M_rows.append(mrow)
                groups.append(gid)

    if not X_rows:
        return _np.array([], dtype=_np.int64), _np.zeros((0, len(kept_otus)), dtype=_np.int64), _np.zeros((0, len(kept_otus)), dtype=bool), _np.array([], dtype=object)

    X_rows = _np.array(X_rows, dtype=_np.int64)
    Y_ml = _np.stack(Y_rows)
    M_ml = _np.stack(M_rows)
    groups = _np.array(groups, dtype=object)
    return X_rows, Y_ml, M_ml, groups


def eval_masked_ovr(name, base_estimator, X_ml, Y_ml, M_ml, groups_ml, n_splits=5, seed=42):
    """
    Grouped K-fold evaluation with masked per-label AUC.
    Evaluates only eligible positions per label on test fold.
    Prints per-fold results and overall mean±std of valid folds.
    """
    import numpy as _np
    import math as _math
    from sklearn.model_selection import GroupKFold
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import roc_auc_score
    # Encode arbitrary group identifiers (e.g., tuples, strings, small arrays) into
    # integer labels to avoid numpy trying to sort heterogeneous Python objects
    # (which can raise TypeError inside GroupKFold).
    group_ids = []
    mapping = {}
    for g in groups_ml:
        key = g
        try:
            hash(key)
        except TypeError:
            # Convert unhashable objects (like numpy arrays) to a tuple of values
            key = tuple(_np.asarray(g).ravel().tolist())
        if key not in mapping:
            mapping[key] = len(mapping)
        group_ids.append(mapping[key])
    groups_arr = _np.asarray(group_ids, dtype=_np.int64)

    gkf = GroupKFold(n_splits=n_splits)
    fold_aucs = []
    total_labels = Y_ml.shape[1]
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_ml, groups=groups_arr), 1):
        tr_mask = _np.zeros(X_ml.shape[0], dtype=bool); tr_mask[tr_idx] = True
        te_mask = _np.zeros(X_ml.shape[0], dtype=bool); te_mask[te_idx] = True

        # Diagnostics (train)
        no_train_examples = 0
        single_class_train = 0
        for j in range(total_labels):
            elig = M_ml[tr_mask, j]
            if not _np.any(elig):
                no_train_examples += 1
                continue
            y_tr = Y_ml[tr_mask, j][elig]
            if _np.unique(y_tr).size < 2:
                single_class_train += 1

        clf = OneVsRestClassifier(base_estimator)
        clf.fit(X_ml[tr_mask], Y_ml[tr_mask])
        prob = clf.predict_proba(X_ml[te_mask])

        per_label = []
        for j in range(total_labels):
            mask = M_ml[te_mask, j]
            if not _np.any(mask):
                continue
            y_true = Y_ml[te_mask, j][mask]
            if _np.unique(y_true).size < 2:
                continue
            try:
                per_label.append(float(roc_auc_score(y_true, prob[:, j][mask])))
            except Exception:
                pass

        fold_auc = float(_np.mean(per_label)) if per_label else float("nan")
        fold_aucs.append(fold_auc)
        print(f"[{name}] Fold {fold}: macro AUC={fold_auc:.3f} over {len(per_label)} evaluable labels")
        print(f"[{name}] Label diagnostics (train): no-train-examples={no_train_examples}/{total_labels}, "
              f"single-class={single_class_train}/{total_labels}")

    valid = [a for a in fold_aucs if not (_math.isnan(a) or _math.isinf(a))]
    if valid:
        mean_auc = _np.mean(valid)
        std_auc = _np.std(valid)
        print(f"[{name}] Grouped {n_splits}-fold macro AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    else:
        print(f"[{name}] No evaluable labels across CV folds.")
