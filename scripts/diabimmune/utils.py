import csv
import os
import sys

# Ensure project root is on sys.path so absolute imports like `scripts.utils` work when running subfolder scripts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils


WGS_RUN_TABLE = 'data/diabimmune/SraRunTable_wgs.csv'
EXTRA_RUN_TABLE = 'data/diabimmune/SraRunTable_extra.csv'
SAMPLES_TABLE = 'data/diabimmune/samples.csv'


def map_runs_to_microbeatlas(run_ids, microbeatlas_path=shared_utils.MICROBEATLAS_SAMPLES):
    """
    Given an iterable of SRA run IDs, map each to a MicrobeAtlas SRS using the
    MicrobeAtlas samples table (TSV). Returns {run_id -> SRS}.
    """
    run_ids = set(run_ids)
    SRA_to_micro = {}
    with open(microbeatlas_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            srs = row['#sid']
            rids = row['rids']
            if not rids:
                continue
            for rid in rids.replace(';', ',').split(','):
                rid = rid.strip()
                if rid and rid in run_ids:
                    SRA_to_micro[rid] = srs
    return SRA_to_micro


def load_run_data(
    wgs_path=WGS_RUN_TABLE,
    extra_path=EXTRA_RUN_TABLE,
    samples_path=SAMPLES_TABLE,
    microbeatlas_path=shared_utils.MICROBEATLAS_SAMPLES,
):
    run_rows = {}
    SRA_to_micro = {}
    gid_to_sample = {}
    micro_to_subject = {}
    micro_to_sample = {}

    for path in (wgs_path, extra_path):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = row['Run'].strip()
                library_name = row.get('Library Name', '').strip()
                subject_id = row.get('host_subject_id', row.get('host_subject_id (run)', '')).strip()
                run_rows[run_id] = {'library': library_name, 'subject': subject_id}

    with open(microbeatlas_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            srs = row['#sid']
            rids = row['rids']
            if not rids:
                continue
            for rid in rids.replace(';', ',').split(','):
                rid = rid.strip()
                if rid:
                    SRA_to_micro[rid] = srs

    with open(samples_path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            pieces = line.split(',')
            if header is None:
                header = pieces
                header[0] = header[0].lstrip('\ufeff')
                continue
            row = dict(zip(header, pieces))
            for key in ('gid_wgs', 'gid_16s'):
                gid = row.get(key, '').strip()
                if gid:
                    gid_to_sample[gid] = {'subject': row['subjectID'], 'sample': row['sampleID']}

    for run_id, srs in SRA_to_micro.items():
        run_info = run_rows.get(run_id, {})
        library_name = run_info.get('library', '')
        if library_name and library_name in gid_to_sample:
            micro_to_sample[srs] = gid_to_sample[library_name]
            micro_to_subject[srs] = gid_to_sample[library_name]['subject']
        elif run_info.get('subject'):
            micro_to_subject[srs] = run_info['subject']

    return run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample


def collect_micro_to_otus(
    SRA_to_micro,
    micro_to_subject,
    mapped_path=shared_utils.MAPPED_PATH,
):
    """
    Build SRS -> OTUs using the mapped MicrobeAtlas file only (no BIOM fallback).
    """
    needed_srs = set(SRA_to_micro.values()) | set(micro_to_subject.keys())
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
    srs_to_terms=None,
    term_to_vec=None,
    include_text: bool = False,
):
    rename_map = None
    try:
        if os.path.exists(shared_utils.RENAME_MAP_PATH):
            rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    except Exception:
        rename_map = None
    resolver = None
    try:
        if rename_map:
            # Prefer bacteria for DIABIMMUNE
            resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B')
    except Exception:
        resolver = None
    return shared_utils.build_sample_embeddings(
        micro_to_otus,
        model,
        device,
        prokbert_path=prokbert_path,
        txt_emb=txt_emb,
        rename_map=rename_map,
        resolver=resolver,
        srs_to_terms=srs_to_terms,
        term_to_vec=term_to_vec,
        include_text=include_text,
    )
