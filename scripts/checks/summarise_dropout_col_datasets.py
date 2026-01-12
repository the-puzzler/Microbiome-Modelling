#!/usr/bin/env python3
"""
Quick summary of how many samples / groups are used for the
dropout and colonisation tasks for:
- DIABIMMUNE
- Gingivitis
- Snowmelt

This mirrors the grouping logic in the corresponding task scripts
but does not load models or embeddings – it only inspects metadata.
"""

import os
import sys
from collections import defaultdict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.diabimmune.utils import load_run_data  # noqa: E402
from scripts.gingivitis.utils import load_gingivitis_run_data  # noqa: E402
from scripts.snowmelt.utils import load_snowmelt_metadata  # noqa: E402


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def summarise_diabimmune():
    print("=== DIABIMMUNE (dropout / colonisation) ===")
    run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
    print(f"raw runs in WGS/extra tables           : {len(run_rows)}")
    print(f"runs mapped to MicrobeAtlas SRS        : {len(SRA_to_micro)}")
    print(f"SRS with subject/sample annotation     : {len(micro_to_sample)}")

    samples_csv = os.path.join("data", "diabimmune", "samples.csv")
    samples_table = {}
    with open(samples_csv) as f:
        header = None
        for line in f:
            parts = line.strip().split(",")
            if not parts:
                continue
            if header is None:
                header = parts
                header[0] = header[0].lstrip("\ufeff")
                continue
            row = dict(zip(header, parts))
            samples_table[row["sampleID"]] = row

    subject_time_to_srs = defaultdict(list)
    subjects = set()
    for srs, info in micro_to_sample.items():
        subject = info.get("subject")
        sample_id = info.get("sample")
        if not subject or not sample_id:
            continue
        rec = samples_table.get(sample_id, {})
        age = _safe_float(rec.get("age_at_collection", ""))
        if age is None:
            continue
        subject_time_to_srs[(subject, age)].append(srs)
        subjects.add(subject)

    unique_srs_used = {s for srs_list in subject_time_to_srs.values() for s in srs_list}
    print(f"subjects with age-resolved samples     : {len(subjects)}")
    print(f"subject–age groups (used as samples)   : {len(subject_time_to_srs)}")
    print(f"unique SRS in those groups             : {len(unique_srs_used)}")

    # subjects contributing to longitudinal tasks (≥2 timepoints)
    subjects_with_pairs = 0
    for subj in subjects:
        times = sorted({t for (s, t) in subject_time_to_srs.keys() if s == subj})
        if len(times) >= 2:
            subjects_with_pairs += 1
    print(f"subjects with ≥2 timepoints (task-usable): {subjects_with_pairs}")
    print()


def summarise_gingivitis():
    print("=== Gingivitis (dropout / colonisation) ===")
    gingivitis_csv = os.path.join("data", "gingivitis", "gingiva.csv")
    run_ids, SRA_to_micro = load_gingivitis_run_data(gingivitis_path=gingivitis_csv)
    print(f"raw runs in gingiva.csv                : {len(run_ids)}")
    print(f"runs mapped to MicrobeAtlas SRS        : {len(SRA_to_micro)}")

    records = []
    with open(gingivitis_csv) as f:
        import csv

        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("Run", "").strip()
            subj = row.get("subject_code", "").strip()
            tcode = row.get("time_code", "").strip()
            if not run or not subj or not tcode:
                continue
            srs = SRA_to_micro.get(run)
            if not srs:
                continue
            records.append({"run": run, "srs": srs, "subject": subj, "time": tcode})

    subject_time_to_srs = defaultdict(list)
    for r in records:
        subject_time_to_srs[(r["subject"], r["time"])].append(r["srs"])

    subjects = sorted({r["subject"] for r in records})
    unique_srs_used = {s for srs_list in subject_time_to_srs.values() for s in srs_list}

    print(f"subjects with mapped runs              : {len(subjects)}")
    print(f"subject–time groups (used as samples)  : {len(subject_time_to_srs)}")
    print(f"unique SRS in those groups             : {len(unique_srs_used)}")

    subjects_with_pairs = 0
    for subj in subjects:
        times = {t for (s, t) in subject_time_to_srs.keys() if s == subj}
        if len(times) >= 2:
            subjects_with_pairs += 1
    print(f"subjects with ≥2 timepoints (task-usable): {subjects_with_pairs}")
    print()


def summarise_snowmelt():
    print("=== Snowmelt (dropout / colonisation) ===")
    snow_csv = os.path.join("data", "snowmelt", "snowmelt.csv")
    run_meta, run_to_srs = load_snowmelt_metadata(snow_csv)
    print(f"runs with block/treatment/time metadata: {len(run_meta)}")
    print(f"runs mapped to SRS                     : {len(run_to_srs)}")

    from collections import defaultdict as dd

    bt_time_to_srs = dd(list)
    for run, meta in run_meta.items():
        srs = run_to_srs.get(run)
        if not srs:
            continue
        key = (meta["block"], meta["treatment"], meta["time"])
        bt_time_to_srs[key].append(srs)

    unique_srs_used = {s for srs_list in bt_time_to_srs.values() for s in srs_list}
    print(f"(block,treatment,time) groups (samples): {len(bt_time_to_srs)}")
    print(f"unique SRS in those groups             : {len(unique_srs_used)}")

    # (block,treatment) combinations with ≥2 timepoints
    bt_to_times = {}
    for (block, trt, time) in bt_time_to_srs.keys():
        bt_to_times.setdefault((block, trt), set()).add(time)
    bt_with_pairs = sum(1 for times in bt_to_times.values() if len(times) >= 2)
    print(f"(block,treatment) combos with ≥2 times : {bt_with_pairs}")
    print()


def main():
    summarise_diabimmune()
    summarise_gingivitis()
    summarise_snowmelt()


if __name__ == "__main__":
    main()

