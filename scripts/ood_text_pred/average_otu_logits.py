#!/usr/bin/env python3
#%% Imports and setup
import os
import sys
import csv
from tqdm import tqdm
import random
import h5py

# Ensure project root on sys.path so `from scripts import utils` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


#%% Goal
# Stream the MicrobeAtlas mapped dataset and compute per-SRS average OTU logits
# on the fly using a Torch IterableDataset + DataLoader. We avoid pre-collecting
# all OTUs or writing temp files. For each header block (a run mapped to an SRS),
# we score its OTUs (no text) and incrementally aggregate sum/count per SRS.
# At the end, we write TSV: sampleid, average_logit (weighted across all runs).


#%% Configure paths
MAPPED_PATH = shared_utils.MAPPED_PATH
CHECKPOINT_PATH = shared_utils.CHECKPOINT_PATH
PROKBERT_PATH = shared_utils.PROKBERT_PATH
RENAME_MAP_PATH = shared_utils.RENAME_MAP_PATH
OUT_TSV = 'data/ood_text_pred/average_otu_logits_notextabl.tsv'
PREFER_DOMAIN = 'B'  # prefer B97_ when both A/B exist

# Optionally append zero-valued scratch tokens per SRS when scoring OTUs
USE_ZERO_SCRATCH_TOKENS = True
SCRATCH_TOKENS_PER_SRS = 16


#%% Dataset: stream mapped file and yield (srs, otu_ids) per header block
import torch
from torch.utils.data import IterableDataset, DataLoader


class MappedBlocksDataset(IterableDataset):
    def __init__(self, mapped_path):
        super().__init__()
        self.mapped_path = mapped_path

    def __iter__(self):
        current_srs = None
        current_otus = []
        with open(self.mapped_path, 'r', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                if line.startswith('>'):
                    if current_srs is not None and current_otus:
                        yield current_srs, current_otus
                    current_otus = []
                    header = line[1:].split()[0]
                    parts = header.split('.')
                    current_srs = parts[-1] if parts else header
                    continue
                if current_srs is None:
                    continue
                first_field = line.split()[0]
                toks = [t for t in first_field.split(';') if t]
                otu97 = next((t for t in toks if t.startswith('97_')), toks[-1] if toks else None)
                if otu97:
                    current_otus.append(otu97)
        if current_srs is not None and current_otus:
            yield current_srs, current_otus


#%% Quick prepass — count total header blocks and unique SRS for accurate progress
def scan_mapped_headers(mapped_path, return_set=False):
    total_headers = 0
    unique_srs = set()
    with open(mapped_path, 'r', errors='replace') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            total_headers += 1
            header = line[1:].split()[0]
            parts = header.split('.')
            srs = parts[-1] if parts else header
            if srs:
                unique_srs.add(srs)
    if return_set:
        return total_headers, unique_srs
    return total_headers, len(unique_srs)

def count_headers_for_subset(mapped_path, subset_srs):
    cnt = 0
    with open(mapped_path, 'r', errors='replace') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            header = line[1:].split()[0]
            parts = header.split('.')
            srs = parts[-1] if parts else header
            if srs in subset_srs:
                cnt += 1
    return cnt

# Choose a random subset of SRS to process
SUBSET_SRS_COUNT = 100_000
RANDOM_SEED = 42
total_headers, unique_srs_set = scan_mapped_headers(MAPPED_PATH, return_set=True)
total_srs = len(unique_srs_set)
sample_k = min(SUBSET_SRS_COUNT, total_srs)
random.seed(RANDOM_SEED)
subset_srs = set(random.sample(list(unique_srs_set), sample_k)) if sample_k < total_srs else set(unique_srs_set)
subset_headers = count_headers_for_subset(MAPPED_PATH, subset_srs)
print('Mapped headers:', total_headers, '| Unique SRS:', total_srs)
print('Subset SRS:', len(subset_srs), '| Subset headers:', subset_headers)


#%% Load model and prepare lazy resolver cache for 97_* → embedding key
model, device = shared_utils.load_microbiome_model(CHECKPOINT_PATH)
rename_map = shared_utils.load_otu_rename_map(RENAME_MAP_PATH) if os.path.exists(RENAME_MAP_PATH) else None

def make_resolver(emb_group):
    cache = {}
    prefer = str(PREFER_DOMAIN or 'B').upper()
    def resolve(new97):
        if new97 in cache:
            return cache[new97]
        candidates = []
        # Try mapping via rename_map if available
        if rename_map:
            a_old = rename_map.get('new97_to_oldA97', {}).get(new97)
            b_old = rename_map.get('new97_to_oldB97', {}).get(new97)
            if prefer == 'B' and b_old:
                candidates.append(b_old)
            if a_old:
                candidates.append(a_old)
            if prefer != 'B' and b_old:
                candidates.append(b_old)
        # Fallback heuristic: construct A97_/B97_ from 97_ suffix
        try:
            if str(new97).startswith('97_'):
                num = str(new97).split('_', 1)[1]
                if prefer == 'B':
                    candidates.extend([f'B97_{num}', f'A97_{num}'])
                else:
                    candidates.extend([f'A97_{num}', f'B97_{num}'])
        except Exception:
            pass
        # Last resort: try raw new97 id
        candidates.append(new97)
        chosen = None
        for key in candidates:
            if key in emb_group:
                chosen = key
                break
        cache[new97] = chosen
        return chosen
    return resolve


#%% Iterate the dataset with DataLoader and aggregate per-SRS average on the fly
os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
dataset = MappedBlocksDataset(MAPPED_PATH)
loader = DataLoader(dataset, batch_size=None, num_workers=0)  # HDF5 not multiprocess-safe

# Per-SRS accumulators: sum of logits and count of OTUs scored across all runs
srs_sum = {}
srs_cnt = {}
blocks_seen = 0
missing_blocks = 0
BATCH_BLOCKS = 16  # number of blocks (SRS chunks) to score together

with h5py.File(PROKBERT_PATH) as emb_file:
    emb_group = emb_file['embeddings']
    resolve = make_resolver(emb_group)
    seen_srs = set()
    pbar = tqdm(total=subset_headers, desc='Streaming subset blocks', mininterval=0.2)

    batch_buf = []  # list of tuples (srs, resolved_keys)

    def process_batch(buf):
        if not buf:
            return
        lengths = [len(keys) for (_, keys) in buf]
        if not any(lengths):
            return
        B = len(buf)
        L = max(lengths)
        # Build padded tensor on CPU then move to device
        x1_cpu = torch.zeros((B, L, shared_utils.OTU_EMB), dtype=torch.float32)
        mask = torch.zeros((B, L), dtype=torch.bool)
        for b, (_, keys) in enumerate(buf):
            i = 0
            for key in keys:
                vec = emb_group[key][()]  # numpy array
                x1_cpu[b, i, :].copy_(torch.from_numpy(vec).to(torch.float32))
                i += 1
            if i > 0:
                mask[b, :i] = True
        x1 = x1_cpu.to(device)
        mask_dev_otus = mask.to(device)
        with torch.no_grad():
            h1 = model.input_projection_type1(x1)
            if USE_ZERO_SCRATCH_TOKENS and SCRATCH_TOKENS_PER_SRS > 0:
                z = torch.zeros((B, SCRATCH_TOKENS_PER_SRS, shared_utils.D_MODEL), dtype=torch.float32, device=device)
                h = torch.cat([h1, z], dim=1)
                mask_dev = torch.ones((B, L + SCRATCH_TOKENS_PER_SRS), dtype=torch.bool, device=device)
            else:
                h = h1
                mask_dev = mask_dev_otus
            h = model.transformer(h, src_key_padding_mask=~mask_dev)
            logits = model.output_projection(h).squeeze(-1)  # (B, L or L+scratch)
        for b, (srs_b, _) in enumerate(buf):
            n = int(lengths[b])
            if n > 0:
                s = float(logits[b, :n].sum().item())
                srs_sum[srs_b] = srs_sum.get(srs_b, 0.0) + s
                srs_cnt[srs_b] = srs_cnt.get(srs_b, 0) + n

    for srs, otus in loader:
        # Skip blocks not in the sampled SRS subset
        if srs not in subset_srs:
            continue
        if srs not in seen_srs:
            seen_srs.add(srs)
            pbar.set_postfix(SRS=f"{len(seen_srs)}/{len(subset_srs)}")
        # Resolve keys; keep those present in HDF5
        keys = []
        for oid in set(otus):  # small dedup within block
            key = resolve(oid)
            if key is not None:
                keys.append(key)
        if not keys:
            missing_blocks += 1
            continue
        batch_buf.append((srs, keys))
        blocks_seen += 1
        pbar.update(1)
        if len(batch_buf) >= BATCH_BLOCKS:
            process_batch(batch_buf)
            batch_buf = []

    # process tail
    if batch_buf:
        process_batch(batch_buf)

# Write TSV once at the end based on accumulators
with open(OUT_TSV, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['sampleid', 'average_logit'])
    for srs, cnt in srs_cnt.items():
        if cnt > 0:
            avg = srs_sum.get(srs, 0.0) / float(cnt)
            w.writerow([srs, f'{avg:.6f}'])

print('Saved:', OUT_TSV, '| SRS:', len(srs_cnt), '| blocks:', blocks_seen, '| empty/missing blocks:', missing_blocks)


#%% Preview first few lines (optional)
try:
    with open(OUT_TSV, 'r') as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i >= 5:
                break
except Exception as e:
    print('Could not preview output:', e)

# %%
