#!/usr/bin/env python3

import csv
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402
from scripts.diabimmune import plot_diabimmune_trajectory_overlay as base  # noqa: E402
from scripts.rollout_metropolis.core import EmbeddingCache, build_otu_index, compute_embedding_from_otus  # noqa: E402


ROLL_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob_oneoutoneinanchored.tsv")
OUT_DIR = os.path.join("data", "rollout_metropolis")
OUT_PNG = os.path.join(OUT_DIR, "diabimmune_rollout_trajectory_overlay_oneoutoneinanchored.png")

REAL_NPZ = os.path.join("data", "diabimmune", "diabimmune_real_embeddings_cache_oneoutoneinanchored.npz")
ENDPOINTS_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoints_cache_oneoutoneinanchored.npz")
CACHE_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_trajectory_overlay_cache_oneoutoneinanchored.npz")

SAMPLES_CSV = os.path.join("data", "diabimmune", "samples.csv")


def _parse_index_list(raw):
    if not raw:
        return []
    return [int(tok) for tok in str(raw).split(";") if str(tok).strip()]


def _load_rollout_rows(tsv_path):
    by_start = defaultdict(list)
    with open(tsv_path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            subject = row.get("subject", "").strip()
            t_start = row.get("t_start", "").strip()
            step = row.get("step", "").strip()
            if not subject or not t_start or not step:
                continue
            by_start[(subject, t_start)].append(row)
    return by_start


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def age_bin_days(age_value):
    return str(int(round(float(age_value))))


def load_samples_table(samples_csv):
    table = {}
    with open(samples_csv) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                header[0] = header[0].lstrip("\ufeff")
                continue
            row = dict(zip(header, parts))
            sid = row.get("sampleID", "").strip()
            if sid:
                table[sid] = row
    return table


def build_subject_age_to_otus(micro_to_sample, micro_to_otus, samples_table):
    subject_age_to_otus = defaultdict(set)
    for srs, info in micro_to_sample.items():
        subj = str(info.get("subject", "")).strip()
        sample_id = str(info.get("sample", "")).strip()
        if not subj or not sample_id:
            continue
        age = safe_float(samples_table.get(sample_id, {}).get("age_at_collection", ""))
        if age is None:
            continue
        age_key = age_bin_days(age)
        otus = micro_to_otus.get(srs, [])
        if otus:
            subject_age_to_otus[(subj, age_key)].update(otus)
    return subject_age_to_otus


def _needs_rebuild(roll_tsv, out_npz):
    if not os.path.exists(out_npz):
        return True
    try:
        return os.path.getmtime(out_npz) < os.path.getmtime(roll_tsv)
    except Exception:
        return True


def ensure_oneoutoneinanchored_caches():
    if not os.path.exists(ROLL_TSV):
        raise SystemExit("Missing rollout TSV: %s" % ROLL_TSV)

    if not (_needs_rebuild(ROLL_TSV, REAL_NPZ) or _needs_rebuild(ROLL_TSV, ENDPOINTS_NPZ)):
        return

    os.makedirs(os.path.dirname(REAL_NPZ), exist_ok=True)
    os.makedirs(os.path.dirname(ENDPOINTS_NPZ), exist_ok=True)

    by_start = _load_rollout_rows(ROLL_TSV)
    start_keys = sorted(by_start.keys())
    if not start_keys:
        raise SystemExit("No usable (subject,t_start) rows found in TSV: %s" % ROLL_TSV)

    _run_rows, sra_to_micro, _gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
    micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)
    samples_table = load_samples_table(SAMPLES_CSV)
    subject_age_to_otus = build_subject_age_to_otus(micro_to_sample, micro_to_otus, samples_table)

    all_otus, _otu_to_idx = build_otu_index(micro_to_otus)
    if not all_otus:
        raise SystemExit("No OTUs available for index mapping.")

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = (
        shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")
        if rename_map
        else {}
    )

    emb_cache = EmbeddingCache()
    real_keys = []
    real_embs = []
    endpoints = []
    endpoints_subject = []
    endpoints_tstart = []

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]

        for subj, t_start in start_keys:
            otus = sorted(subject_age_to_otus.get((subj, t_start), set()))
            if not otus:
                continue
            e = compute_embedding_from_otus(
                otus,
                model=model,
                device=device,
                emb_group=emb_group,
                resolver=resolver,
                scratch_tokens=0,
                d_model=shared_utils.D_MODEL,
                emb_cache=emb_cache,
            )
            if e is None:
                continue
            real_keys.append((str(subj), str(t_start)))
            real_embs.append(e.astype(np.float32))

        for subj, t_start in start_keys:
            rows = by_start[(subj, t_start)]
            rows_sorted = sorted(
                rows,
                key=lambda r: int(r.get("step")) if str(r.get("step", "")).isdigit() else -1,
            )
            if not rows_sorted:
                continue
            last = rows_sorted[-1]
            idx_list = _parse_index_list(last.get("current_otu_indices", ""))
            otus_final = [all_otus[i] for i in idx_list if 0 <= i < len(all_otus)]
            e = compute_embedding_from_otus(
                otus_final,
                model=model,
                device=device,
                emb_group=emb_group,
                resolver=resolver,
                scratch_tokens=0,
                d_model=shared_utils.D_MODEL,
                emb_cache=emb_cache,
            )
            if e is None:
                continue
            endpoints.append(e.astype(np.float32))
            endpoints_subject.append(str(subj))
            endpoints_tstart.append(str(t_start))

    if not real_embs:
        raise SystemExit("No real embeddings computed (check DIABIMMUNE mappings and TSV).")
    if not endpoints:
        raise SystemExit("No endpoints computed (check TSV).")

    np.savez(
        REAL_NPZ,
        keys=np.asarray(real_keys, dtype=object),
        emb=np.stack(real_embs, axis=0),
    )
    np.savez(
        ENDPOINTS_NPZ,
        endpoints=np.stack(endpoints, axis=0),
        subject=np.asarray(endpoints_subject, dtype=object),
        t1=np.asarray(endpoints_tstart, dtype=object),
        t1_labels=np.asarray(endpoints_tstart, dtype=object),
        endpoints_per_start=np.asarray(1, dtype=int),
    )

    print("Saved:", REAL_NPZ)
    print("Saved:", ENDPOINTS_NPZ)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_oneoutoneinanchored_caches()

    base.ROLL_TSV = ROLL_TSV
    base.ENDPOINTS_NPZ = ENDPOINTS_NPZ
    base.REAL_NPZ = REAL_NPZ
    base.OUT_PNG = OUT_PNG
    base.CACHE_NPZ = CACHE_NPZ
    base.SAMPLES_CSV = SAMPLES_CSV
    base.FONT_SIZE = 14
    base.HIDE_XY_TICK_LABELS = True
    base.main()


if __name__ == "__main__":
    main()
