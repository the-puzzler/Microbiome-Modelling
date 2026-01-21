#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402
from scripts.rollout_metropolis.core import write_rollout_tsv  # noqa: E402


OUT_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob_metropolis.tsv")
GINGIVA_CSV = os.path.join("data", "gingivitis", "gingiva.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Metropolis rollouts from gingivitis (subject,time_code) starts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--p-anchor", type=float, default=0.95)
    p.add_argument("--p-add", type=float, default=0.34)
    p.add_argument("--p-drop", type=float, default=0.33)
    p.add_argument("--p-swap", type=float, default=0.33)
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--out", default=OUT_TSV)
    p.add_argument("--gingiva-csv", default=GINGIVA_CSV)
    return p.parse_args()


def build_subject_time_to_otus(gingiva_csv, sra_to_micro, micro_to_otus):
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


def main():
    args = parse_args()
    np.random.seed(args.seed)

    _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=args.gingiva_csv)
    micro_to_otus = collect_micro_to_otus(sra_to_micro)
    subject_time_to_otus = build_subject_time_to_otus(args.gingiva_csv, sra_to_micro, micro_to_otus)

    starts = [(subj, t, sorted(otus)) for (subj, t), otus in subject_time_to_otus.items() if otus]
    if not starts:
        raise SystemExit("No gingivitis start points found.")

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
        max_candidates=args.max_candidates,
    )
    print(f"Saved: {args.out}")
    print("start points:", len(starts))


if __name__ == "__main__":
    main()

