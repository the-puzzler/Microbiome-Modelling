#%% Debug explorer for mappings, embeddings, and coverage
import os
import sys
from collections import Counter
from pathlib import Path

import h5py

# Ensure repo root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import load_gingivitis_run_data  # noqa: E402
from scripts.diabimmune.utils import load_run_data  # noqa: E402


def fmt(n):
    return f"{n:,}"


def print_file_status():
    print('Files present:')
    paths = {
        'mapped': shared_utils.MAPPED_PATH,
        'rename_map': shared_utils.RENAME_MAP_PATH,
        'prokbert_h5': shared_utils.PROKBERT_PATH,
        'diabimmune_samples_tsv': shared_utils.MICROBEATLAS_SAMPLES,
        'gingivitis_csv': 'data/gingivitis/gingiva.csv',
        'wgs_run_table': 'data/diabimmune/SraRunTable_wgs.csv',
        'extra_run_table': 'data/diabimmune/SraRunTable_extra.csv',
        'diabimmune_samples': 'data/diabimmune/samples.csv',
    }
    for name, p in paths.items():
        print(f"- {name}: {'OK' if os.path.exists(p) else 'MISSING'} ({p})")


def inspect_embeddings():
    if not os.path.exists(shared_utils.PROKBERT_PATH):
        print('Embeddings file missing; skip inspect_embeddings')
        return
    with h5py.File(shared_utils.PROKBERT_PATH) as f:
        emb = f['embeddings']
        keys = []
        for i, k in enumerate(emb.keys()):
            keys.append(k)
            if i >= 999:
                break
    cats = Counter()
    for k in keys:
        if k.startswith('A97_'):
            cats['A97_'] += 1
        elif k.startswith('B97_'):
            cats['B97_'] += 1
        elif k.startswith('97_'):
            cats['97_'] += 1
        else:
            cats['other'] += 1
    print('Embedding key prefix sample (first 1K):', dict(cats))
    prefer_A = cats['A97_'] >= cats['B97_']
    print('Preferred domain based on prefixes:', 'A97_' if prefer_A else 'B97_')


def inspect_rename_map():
    if not os.path.exists(shared_utils.RENAME_MAP_PATH):
        print('Rename map missing; skip inspect_rename_map')
        return None
    rm = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    print('Rename map sizes:', {
        'old_to_new': fmt(len(rm['old_to_new'])),
        'new_to_old': fmt(len(rm['new_to_old'])),
        'oldA97_to_new97': fmt(len(rm.get('oldA97_to_new97', {}))),
        'new97_to_oldA97': fmt(len(rm.get('new97_to_oldA97', {}))),
        'oldB97_to_new97': fmt(len(rm.get('oldB97_to_new97', {}))),
        'new97_to_oldB97': fmt(len(rm.get('new97_to_oldB97', {}))),
    })
    # Show some 97-level examples for both domains
    exA = list(rm.get('new97_to_oldA97', {}).items())[:5]
    exB = list(rm.get('new97_to_oldB97', {}).items())[:5]
    for k, v in exA:
        print('  new97->oldA97:', k, '=>', v)
    for k, v in exB:
        print('  new97->oldB97:', k, '=>', v)
    return rm


def build_acc_to_srs_from_mapped_sample(n=20):
    """Print a small sample of accession→SRS pairs from the mapped file headers."""
    if not os.path.exists(shared_utils.MAPPED_PATH):
        print('Mapped file missing; skip build_acc_to_srs_from_mapped_sample')
        return {}
    acc_to_srs = {}
    count = 0
    with open(shared_utils.MAPPED_PATH, 'r', errors='replace') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            header = line[1:].split()[0]
            parts = header.split('.')
            if len(parts) >= 2:
                acc = parts[0]
                srs = parts[-1]
                acc_to_srs[acc] = srs
                count += 1
                if count >= n:
                    break
    print('Mapped header sample (acc→SRS):', list(acc_to_srs.items())[:5])
    return acc_to_srs


def coverage_for_srs(srs_list, rename_map, limit=5):
    print(f'Checking coverage for {min(limit, len(srs_list))} SRS...')
    srs_list = list(srs_list)[:limit]
    micro_to_otus = shared_utils.collect_micro_to_otus_mapped(set(srs_list), shared_utils.MAPPED_PATH)
    with h5py.File(shared_utils.PROKBERT_PATH) as f:
        emb = f['embeddings']
        for srs in srs_list:
            otus = micro_to_otus.get(srs, [])
            n = len(otus)
            matched = 0
            examples = []
            for oid in otus[:50]:
                cands = [oid]
                if isinstance(oid, str) and oid.startswith('97_') and rename_map:
                    # Try both A and B candidates for visibility
                    old_a = rename_map.get('new97_to_oldA97', {}).get(oid)
                    old_b = rename_map.get('new97_to_oldB97', {}).get(oid)
                    if old_a:
                        cands.append(old_a)
                    if old_b:
                        cands.append(old_b)
                chosen = None
                for c in cands:
                    if c in emb:
                        chosen = c
                        break
                if chosen:
                    matched += 1
                    if len(examples) < 5:
                        examples.append((oid, chosen))
            rate = (100.0 * matched / n) if n else 0.0
            print(f'  SRS {srs}: OTUs={n}, matched={matched} ({rate:.1f}%), examples={examples}')


def gingivitis_section():
    print('\n=== Gingivitis ===')
    runs, sra_to_srs = load_gingivitis_run_data()
    print('Runs:', fmt(len(runs)), '| mapped to SRS:', fmt(len(sra_to_srs)))
    # Gather SRS
    srs = set(sra_to_srs.values())
    return srs


def diabimmune_section():
    print('\n=== DIABIMMUNE ===')
    try:
        run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
    except Exception as e:
        print('Error in load_run_data:', e)
        return set()
    print('SRA→SRS mappings:', fmt(len(SRA_to_micro)))
    srs = set(SRA_to_micro.values())
    print('Unique SRS (from SRA):', fmt(len(srs)))
    return srs


def model_check():
    print('\n=== Model load check ===')
    try:
        model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
        info = getattr(model, '_load_info', None)
        if info and getattr(info, 'missing_keys', None):
            print('Missing keys (first):', info.missing_keys[:10])
        if info and getattr(info, 'unexpected_keys', None):
            print('Unexpected keys (first):', info.unexpected_keys[:10])
        print('Device:', device)
    except Exception as e:
        print('Model load error:', e)


if __name__ == '__main__':
    print_file_status()
    inspect_embeddings()
    rm = inspect_rename_map()
    build_acc_to_srs_from_mapped_sample()
    g_srs = gingivitis_section()
    d_srs = diabimmune_section()
    if rm and g_srs:
        coverage_for_srs(list(g_srs), rm, limit=5)
    if rm and d_srs:
        coverage_for_srs(list(d_srs), rm, limit=5)
    model_check()
