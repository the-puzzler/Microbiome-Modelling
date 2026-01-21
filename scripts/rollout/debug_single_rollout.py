#!/usr/bin/env python3

import argparse
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
from scripts.rollout.core import build_otu_index, rollout_steps  # noqa: E402


GINGIVITIS_CSV = os.path.join("data", "gingivitis", "gingiva.csv")
OUT_TSV = os.path.join("data", "gingivitis", "single_rollout_debug.tsv")


def safe_float_time(val):
    s = str(val).strip()
    if s.upper() == "B":
        return -1e9
    try:
        return float(s)
    except Exception:
        return 1e9


def build_subject_time_otus(gingivitis_csv, micro_to_otus, sra_to_micro):
    records = []
    with open(gingivitis_csv) as f:
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
            records.append({"srs": srs, "subject": subj, "time": tcode})

    subject_time_to_otus = defaultdict(set)
    subject_to_times = defaultdict(list)
    for r in records:
        otus = micro_to_otus.get(r["srs"], [])
        if otus:
            subject_time_to_otus[(r["subject"], r["time"])].update(otus)
            subject_to_times[r["subject"]].append(r["time"])
    return subject_time_to_otus, subject_to_times


def parse_args():
    p = argparse.ArgumentParser(description="Run a single rollout (gingiva or diabimmune) and log intermediate dynamics.")
    p.add_argument("--dataset", choices=["gingiva", "diabimmune"], default="gingiva")
    p.add_argument("--subject", default="", help="Subject id (default: pick first available).")
    p.add_argument("--t1", default="", help="Start timepoint (default: pick first available).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.out is None:
        args.out = (
            os.path.join("data", "gingivitis", "single_rollout_debug.tsv")
            if args.dataset == "gingiva"
            else os.path.join("data", "diabimmune", "single_rollout_debug.tsv")
        )

    if args.dataset == "gingiva":
        _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=GINGIVITIS_CSV)
        micro_to_otus = collect_micro_to_otus(sra_to_micro)
        subject_time_to_otus, subject_to_times = build_subject_time_otus(GINGIVITIS_CSV, micro_to_otus, sra_to_micro)
        subject = args.subject.strip()
        t1 = args.t1.strip()
        if not (subject and t1):
            picked = None
            for subj in sorted(subject_to_times.keys()):
                times = sorted(set(subject_to_times[subj]), key=lambda t: (safe_float_time(t), str(t)))
                if len(times) >= 1:
                    picked = (subj, str(times[0]))
                    break
            if picked is None:
                raise SystemExit("Could not find any subject with >=1 timepoint.")
            subject, t1 = picked
        otus_t1 = sorted(subject_time_to_otus.get((subject, t1), set()))
    else:
        from scripts.diabimmune.utils import load_run_data as load_diabimmune_run_data  # noqa: E402
        from scripts.diabimmune.utils import collect_micro_to_otus as collect_diabimmune_micro_to_otus  # noqa: E402

        run_rows, sra_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_diabimmune_run_data()
        micro_to_otus = collect_diabimmune_micro_to_otus(sra_to_micro, micro_to_subject)

        samples_csv = os.path.join("data", "diabimmune", "samples.csv")
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

        def age_bin_days(age_value):
            return str(int(round(float(age_value))))

        subject_time_to_otus = defaultdict(set)
        subject_to_times = defaultdict(list)
        for srs, info in micro_to_sample.items():
            subj = str(info.get("subject", "")).strip()
            sample_id = str(info.get("sample", "")).strip()
            if not subj or not sample_id:
                continue
            age_str = samples_table.get(sample_id, {}).get("age_at_collection", "")
            try:
                age = float(age_str)
            except Exception:
                continue
            t = age_bin_days(age)
            otus = micro_to_otus.get(srs, [])
            if otus:
                subject_time_to_otus[(subj, t)].update(otus)
                subject_to_times[subj].append(t)

        subject = args.subject.strip()
        t1 = args.t1.strip()
        if not (subject and t1):
            if not subject_to_times:
                raise SystemExit("No DIABIMMUNE subject-time groups found.")
            subj = sorted(subject_to_times.keys())[0]
            t = sorted(set(subject_to_times[subj]), key=lambda x: float(x))[0]
            subject, t1 = subj, t
        otus_t1 = sorted(subject_time_to_otus.get((subject, t1), set()))

    all_otus, otu_to_idx = build_otu_index(micro_to_otus)
    if not all_otus:
        raise SystemExit("No OTUs available for candidate pool.")
    if not otus_t1:
        raise SystemExit(f"No OTUs for (subject,t_start)=({subject},{t1}).")

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    absent = list(set(all_otus) - set(otus_t1))
    if len(absent) > args.max_candidates:
        absent = rng.choice(np.asarray(absent, dtype=object), size=args.max_candidates, replace=False).tolist()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "subject",
        "t_start",
        "step",
        "n_current",
        "n_same",
        "n_added",
        "n_removed",
        "current_otu_indices",
    ]

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file, open(args.out, "w", newline="") as out_f:
        emb_group = emb_file["embeddings"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in rollout_steps(
            start_otus=otus_t1,
            absent_pool=absent,
            otu_to_idx=otu_to_idx,
            model=model,
            device=device,
            emb_group=emb_group,
            resolver=resolver,
            rng=rng,
            steps=args.steps,
            temperature=args.temperature,
        ):
            writer.writerow({"subject": subject, "t_start": t1, **row})

    print(f"Start: ({subject}, {t1})")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
