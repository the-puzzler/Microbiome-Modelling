import csv
import os
import sys

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils


META_PATH = 'data/infants/meta_withbirth.csv'
INFANTS_OTUS_PATH = 'data/infants/infants_otus.tsv'


def load_infants_meta(meta_path=META_PATH):
    """
    Load infants metadata mapping SampleID -> class label (Env).
    Returns: dict sample_id -> label
    """
    meta = {}
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row['SampleID'].strip()
            label = row['Env'].strip()
            if sample_id and label:
                meta[sample_id] = label
    return meta


def collect_micro_to_otus_for_samples(sample_ids, acc_to_srs):
    """
    Given iterable of ERR/SRR sample IDs and accession->SRS mapping, collect
    SRS -> 97_* OTU lists via the mapped file. Returns (srs_for_sample, micro_to_otus)
    where srs_for_sample maps SampleID -> SRS (only those that mapped).
    """
    srs_for_sample = {}
    needed_srs = set()
    for sid in sample_ids:
        srs = acc_to_srs.get(sid)
        if srs:
            srs_for_sample[sid] = srs
            needed_srs.add(srs)
    micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
    return srs_for_sample, micro_to_otus


def load_infants_otus_tsv(path=INFANTS_OTUS_PATH):
    """
    Parse infants_otus.tsv with columns: sample_id, otus, abundances (tab-separated).
    Returns dict: sample_id -> list of OTU ids (strings), ignoring empty tokens.
    """
    sample_to_otus = {}
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            sid = parts[0].strip()
            otus_field = parts[1].strip()
            # Strip surrounding brackets if present and split by comma
            if otus_field.startswith('[') and otus_field.endswith(']'):
                otus_field = otus_field[1:-1]
            otus = [tok.strip() for tok in otus_field.split(',') if tok.strip()]
            sample_to_otus[sid] = otus
    return sample_to_otus
