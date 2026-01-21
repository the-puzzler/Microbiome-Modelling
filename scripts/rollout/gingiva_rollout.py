#!/usr/bin/env python3

import csv
import argparse
import os
import random
import sys
from collections import defaultdict

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402
from scripts.rollout.core import write_rollout_tsv  # noqa: E402


OUT_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob.tsv")
GINGIVITIS_CSV = os.path.join("data", "gingivitis", "gingiva.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Run rollouts from each gingiva (subject,timepoint) start.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-subjects", type=int, default=0, help="0 means all subjects.")
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", default=OUT_TSV)
    p.add_argument("--gingiva-csv", default=GINGIVITIS_CSV)
    return p.parse_args()


def build_subject_time_to_otus(gingivitis_csv, micro_to_otus, sra_to_micro):
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
            records.append((subj, tcode, srs))

    subject_time_to_otus = defaultdict(set)
    for subj, tcode, srs in records:
        otus = micro_to_otus.get(srs, [])
        if otus:
            subject_time_to_otus[(subj, tcode)].update(otus)
    return subject_time_to_otus


def main():
    args = parse_args()
    random.seed(args.seed)
    _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=args.gingiva_csv)
    micro_to_otus = collect_micro_to_otus(sra_to_micro)

    # Build starts: one rollout from each (subject,time_code) real point.
    subject_time_to_otus = build_subject_time_to_otus(args.gingiva_csv, micro_to_otus, sra_to_micro)
    starts = []
    for (subj, t), otus in sorted(subject_time_to_otus.items()):
        starts.append((str(subj), str(t), sorted(otus)))

    if args.max_subjects and args.max_subjects > 0:
        subjects = sorted({s for (s, _t, _o) in starts})
        if len(subjects) > args.max_subjects:
            keep = set(random.sample(subjects, args.max_subjects))
            starts = [x for x in starts if x[0] in keep]

    write_rollout_tsv(
        out_tsv=args.out,
        starts=starts,
        micro_to_otus=micro_to_otus,
        seed=args.seed,
        max_candidates=args.max_candidates,
        steps_per_rollout=args.steps,
        temperature=args.temperature,
        checkpoint_path=shared_utils.CHECKPOINT_PATH,
        prokbert_path=shared_utils.PROKBERT_PATH,
        rename_map_path=shared_utils.RENAME_MAP_PATH,
        prefer_resolver="B",
        scratch_tokens=0,
        d_model=shared_utils.D_MODEL,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
