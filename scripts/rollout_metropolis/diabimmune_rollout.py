#!/usr/bin/env python3

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402
from scripts.rollout_metropolis.core import write_rollout_tsv  # noqa: E402


OUT_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob_metropolis.tsv")
SAMPLES_CSV = os.path.join("data", "diabimmune", "samples.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Metropolis rollouts from DIABIMMUNE (subject, age_at_collection_in_days) starts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-starts", type=int, default=1000, help="Randomly sample this many real start points (0 = all).")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--p-anchor", type=float, default=0.95)
    p.add_argument("--p-add", type=float, default=0.34)
    p.add_argument("--p-drop", type=float, default=0.33)
    p.add_argument("--p-swap", type=float, default=0.33)
    p.add_argument("--n-proposals", type=int, default=10, help="Number of candidate moves scored per step (batched).")
    p.add_argument("--out", default=OUT_TSV)
    p.add_argument("--samples-csv", default=SAMPLES_CSV)
    return p.parse_args()


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


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    _run_rows, sra_to_micro, _gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
    micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)
    samples_table = load_samples_table(args.samples_csv)
    subject_age_to_otus = build_subject_age_to_otus(micro_to_sample, micro_to_otus, samples_table)

    starts_all = [(subj, str(age), sorted(otus)) for (subj, age), otus in subject_age_to_otus.items() if otus]
    if not starts_all:
        raise SystemExit("No (subject,age) start points available for DIABIMMUNE.")

    rng = np.random.default_rng(args.seed)
    if args.n_starts and args.n_starts > 0 and len(starts_all) > args.n_starts:
        idx = rng.choice(len(starts_all), size=args.n_starts, replace=False)
        starts = [starts_all[i] for i in idx.tolist()]
    else:
        starts = starts_all

    write_rollout_tsv(
        out_tsv=args.out,
        starts=starts,
        micro_to_otus=micro_to_otus,
        seed=args.seed,
        steps_per_rollout=args.steps,
        temperature=args.temperature,
        checkpoint_path=shared_utils.CHECKPOINT_PATH,
        prokbert_path=shared_utils.PROKBERT_PATH,
        rename_map_path=shared_utils.RENAME_MAP_PATH,
        prefer_resolver="B",
        p_anchor=args.p_anchor,
        p_add=args.p_add,
        p_drop=args.p_drop,
        p_swap=args.p_swap,
        n_proposals=args.n_proposals,
    )
    print(f"Saved: {args.out}")
    print("start points:", len(starts))


if __name__ == "__main__":
    main()
