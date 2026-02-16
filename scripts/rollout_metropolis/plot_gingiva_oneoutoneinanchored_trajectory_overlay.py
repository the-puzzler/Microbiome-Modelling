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
from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402
from scripts.gingivitis import plot_gingivitis_trajectory_overlay as base  # noqa: E402
from scripts.rollout_metropolis.core import EmbeddingCache, build_otu_index, compute_embedding_from_otus  # noqa: E402


ROLL_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob_oneoutoneinanchored.tsv")
OUT_DIR = os.path.join("data", "rollout_metropolis")
OUT_PNG = os.path.join(OUT_DIR, "gingivitis_rollout_trajectory_overlay_oneoutoneinanchored.png")

REAL_NPZ = os.path.join("data", "gingivitis", "visionary_rollout_direction_cache_oneoutoneinanchored.npz")
ENDPOINTS_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_endpoints_cache_oneoutoneinanchored.npz")
CACHE_NPZ = os.path.join("data", "gingivitis", "gingivitis_rollout_trajectory_overlay_cache_oneoutoneinanchored.npz")

GINGIVA_CSV = os.path.join("data", "gingivitis", "gingiva.csv")


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


def _build_subject_time_to_otus(gingiva_csv, sra_to_micro, micro_to_otus):
    subject_time_to_otus = defaultdict(set)
    with open(gingiva_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("Run", "").strip()
            subj = row.get("subject_code", "").strip()
            tcode = row.get("time_code", "").strip()
            if not run or not subj or not tcode:
                continue
            srs = sra_to_micro.get(run)
            if not srs:
                continue
            otus = micro_to_otus.get(srs, [])
            if otus:
                subject_time_to_otus[(subj, tcode)].update(otus)
    return subject_time_to_otus


def _needs_rebuild(roll_tsv, out_npz):
    if not os.path.exists(out_npz):
        return True
    try:
        return os.path.getmtime(out_npz) < os.path.getmtime(roll_tsv)
    except Exception:
        return True


def _best_row_for_endpoint(rows_sorted, *, window=10):
    """
    Select the endpoint row for a truncated "optimum search":
    stop after `window` consecutive steps without a new best anchor_mean_logit,
    and return the best-scoring row seen up to that stopping point.
    """
    if not rows_sorted:
        return None
    if "anchor_mean_logit" not in rows_sorted[0]:
        return rows_sorted[-1]

    best = float("-inf")
    best_row = rows_sorted[0]
    since_best = 0
    for row in rows_sorted:
        try:
            v = float(row.get("anchor_mean_logit", "nan"))
        except Exception:
            v = float("nan")
        if np.isfinite(v) and v > best:
            best = float(v)
            best_row = row
            since_best = 0
        else:
            since_best += 1
            if since_best >= int(window):
                break
    return best_row


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

    _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=GINGIVA_CSV)
    micro_to_otus = collect_micro_to_otus(sra_to_micro)
    subject_time_to_otus = _build_subject_time_to_otus(GINGIVA_CSV, sra_to_micro, micro_to_otus)

    all_otus, _otu_to_idx = build_otu_index(micro_to_otus)
    if not all_otus:
        raise SystemExit("No OTUs available for index mapping.")

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    emb_cache = EmbeddingCache()
    real_keys = []
    real_embs = []
    endpoints = []
    endpoints_subject = []
    endpoints_tstart = []

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]

        for subj, t_start in start_keys:
            otus = sorted(subject_time_to_otus.get((subj, t_start), set()))
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
            endpoint_row = _best_row_for_endpoint(rows_sorted, window=10)
            if endpoint_row is None:
                continue
            idx_list = _parse_index_list(endpoint_row.get("current_otu_indices", ""))
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
        raise SystemExit("No real embeddings computed (check gingiva mappings and TSV).")
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
    base.main()


if __name__ == "__main__":
    main()
