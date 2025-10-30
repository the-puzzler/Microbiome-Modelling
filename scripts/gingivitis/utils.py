import csv
import os
import sys

# Ensure project root is on sys.path so absolute imports like `scripts.utils` work when running subfolder scripts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils

# Paths
GINGIVITIS_TABLE = 'data/gingivitis/gingiva.csv'


def load_gingivitis_run_data(
    gingivitis_path=GINGIVITIS_TABLE,
    microbeatlas_path=None,
):
    """
    Read gingivitis SRA run IDs and map them to MicrobeAtlas SRS IDs.

    Returns:
        run_ids: set of SRA Run IDs found in gingivitis table
        SRA_to_micro: dict mapping SRR/ERR run ID -> MicrobeAtlas SRS
    """
    run_ids = set()
    with open(gingivitis_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get('Run', '').strip()
            if run_id:
                run_ids.add(run_id)

    # Map using headers from the mapped file (handles SRR/ERR/ERS etc.)
    acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
    SRA_to_micro = {rid: acc_to_srs[rid] for rid in run_ids if rid in acc_to_srs}

    mapped = len(SRA_to_micro)
    print(f'gingivitis runs: {len(run_ids)}, mapped to microbeatlas (via mapped headers): {mapped}')
    return run_ids, SRA_to_micro


def collect_micro_to_otus(SRA_to_micro, mapped_path=shared_utils.MAPPED_PATH):
    """
    Build SRS -> OTUs using the mapped MicrobeAtlas file only (no BIOM fallback).
    """
    needed_srs = set(SRA_to_micro.values())
    return shared_utils.collect_micro_to_otus_mapped(needed_srs, mapped_path)


def load_microbiome_model(checkpoint_path=shared_utils.CHECKPOINT_PATH):
    return shared_utils.load_microbiome_model(checkpoint_path)


def preview_prokbert_embeddings(prokbert_path=shared_utils.PROKBERT_PATH, limit=10):
    return shared_utils.preview_prokbert_embeddings(prokbert_path, limit)


def build_sample_embeddings(
    micro_to_otus,
    model,
    device,
    prokbert_path=shared_utils.PROKBERT_PATH,
    txt_emb=shared_utils.TXT_EMB,
):
    rename_map = None
    try:
        if os.path.exists(shared_utils.RENAME_MAP_PATH):
            rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    except Exception:
        rename_map = None
    return shared_utils.build_sample_embeddings(
        micro_to_otus,
        model,
        device,
        prokbert_path=prokbert_path,
        txt_emb=txt_emb,
        rename_map=rename_map,
        map_direction='new_to_old',
    )
