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
from scripts.rollout.core import build_otu_index, compute_embedding_from_otus  # noqa: E402


def _parse_index_list(raw):
    if not raw:
        return []
    return [int(tok) for tok in str(raw).split(";") if str(tok).strip()]


def load_rollout_rows(tsv_path):
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


def _gingiva_config():
    from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402

    rollout_tsv = os.path.join("data", "gingivitis", "visionary_rollout_prob.tsv")
    gingiva_csv = os.path.join("data", "gingivitis", "gingiva.csv")
    out_real_npz = os.path.join("data", "gingivitis", "visionary_rollout_direction_cache.npz")
    out_end_npz = os.path.join("data", "gingivitis", "gingivitis_rollout_endpoints_cache.npz")

    _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=gingiva_csv)
    micro_to_otus = collect_micro_to_otus(sra_to_micro)  # SRS -> OTUs

    # Build (subject,time) -> OTUs (union across runs).
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

    return {
        "name": "gingiva",
        "rollout_tsv": rollout_tsv,
        "out_real_npz": out_real_npz,
        "out_end_npz": out_end_npz,
        "micro_to_otus": micro_to_otus,
        "get_start_otus": lambda subj, t: subject_time_to_otus.get((subj, t), set()),
    }


def _diabimmune_config():
    from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402

    rollout_tsv = os.path.join("data", "diabimmune", "visionary_rollout_prob.tsv")
    samples_csv = os.path.join("data", "diabimmune", "samples.csv")
    out_real_npz = os.path.join("data", "diabimmune", "diabimmune_real_embeddings_cache.npz")
    out_end_npz = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoints_cache.npz")

    run_rows, sra_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
    micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)  # SRS -> OTUs

    # Load sample ages and build (subject,age) -> OTUs.
    samples_table = {}
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
                samples_table[sid] = row

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def age_bin_days(age_value):
        return str(int(round(float(age_value))))

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

    return {
        "name": "diabimmune",
        "rollout_tsv": rollout_tsv,
        "out_real_npz": out_real_npz,
        "out_end_npz": out_end_npz,
        "micro_to_otus": micro_to_otus,
        "get_start_otus": lambda subj, t: subject_age_to_otus.get((subj, t), set()),
    }


def _needs_cache(cfg):
    return not (os.path.exists(cfg["out_real_npz"]) and os.path.exists(cfg["out_end_npz"]))


def _build_caches(cfg):
    rollout_tsv = cfg["rollout_tsv"]
    if not os.path.exists(rollout_tsv):
        raise SystemExit(f"Missing rollout TSV: {rollout_tsv}")

    by_start = load_rollout_rows(rollout_tsv)
    start_keys = sorted(by_start.keys())
    if not start_keys:
        raise SystemExit(f"No usable (subject,t_start) rows found in TSV: {rollout_tsv}")

    micro_to_otus = cfg["micro_to_otus"]
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

    os.makedirs(os.path.dirname(cfg["out_real_npz"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["out_end_npz"]), exist_ok=True)

    real_keys = []
    real_embs = []
    endpoints = []
    endpoints_subject = []
    endpoints_tstart = []

    get_start_otus = cfg["get_start_otus"]
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]

        for subj, t_start in start_keys:
            otus = sorted(get_start_otus(subj, t_start))
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
            )
            if e is None:
                continue
            real_keys.append((str(subj), str(t_start)))
            real_embs.append(e.astype(np.float32))

        for subj, t_start in start_keys:
            rows = by_start[(subj, t_start)]
            rows_sorted = sorted(rows, key=lambda r: int(r.get("step")) if str(r.get("step", "")).isdigit() else -1)
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
            )
            if e is None:
                continue
            endpoints.append(e.astype(np.float32))
            endpoints_subject.append(str(subj))
            endpoints_tstart.append(str(t_start))

    if not real_embs:
        raise SystemExit("No real embeddings computed (check data mappings and TSV).")
    if not endpoints:
        raise SystemExit("No endpoints computed (check TSV).")

    np.savez(
        cfg["out_real_npz"],
        keys=np.asarray(real_keys, dtype=object),
        emb=np.stack(real_embs, axis=0),
    )
    np.savez(
        cfg["out_end_npz"],
        endpoints=np.stack(endpoints, axis=0),
        subject=np.asarray(endpoints_subject, dtype=object),
        t1=np.asarray(endpoints_tstart, dtype=object),
        t1_labels=np.asarray(endpoints_tstart, dtype=object),
        endpoints_per_start=np.asarray(1, dtype=int),
    )

    print(f"Saved: {cfg['out_real_npz']}")
    print(f"Saved: {cfg['out_end_npz']}")
    print(f"{cfg['name']} real points:", len(real_keys))
    print(f"{cfg['name']} endpoints:", len(endpoints))


def main():
    configs = {"gingiva": _gingiva_config, "diabimmune": _diabimmune_config}
    cfgs = [configs[k]() for k in ("gingiva", "diabimmune")]
    to_run = [c for c in cfgs if _needs_cache(c) and os.path.exists(c["rollout_tsv"])]

    if not to_run:
        print("No caches to build (either already exist or rollout TSV is missing).")
        return

    for cfg in to_run:
        _build_caches(cfg)


if __name__ == "__main__":
    main()
