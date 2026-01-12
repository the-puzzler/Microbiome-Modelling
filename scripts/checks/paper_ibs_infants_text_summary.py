#!/usr/bin/env python3
"""
Summarise text annotations associated with IBS and infants samples.

For each dataset:
  - Collect the set of runs/samples used.
  - Look up their associated text terms via shared_utils.parse_run_terms.
  - Count, for each term, in how many samples it appears.
  - Write a tab-separated summary file listing term, count, and percentage of
    samples carrying that term.

Outputs:
  - data/IBS/text_annotation_summary.txt
  - data/infants/text_annotation_summary.txt
"""

import os
import sys
from collections import defaultdict


# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.IBS.predict_ibs import (  # noqa: E402
    load_ibs_metadata,
)
from scripts.infants.utils import (  # noqa: E402
    load_infants_meta,
)


def summarise_terms_for_ids(sample_ids, run_to_terms, out_path, label):
    """
    Generic helper: given iterable of sample_ids and a run->terms mapping,
    write a summary file with term counts and coverage percentages.
    """
    sample_ids = list(sample_ids)
    total = len(sample_ids)
    term_counts = defaultdict(int)
    samples_with_any_terms = 0

    for sid in sample_ids:
        terms = run_to_terms.get(sid, [])
        if not terms:
            continue
        samples_with_any_terms += 1
        for t in set(terms):
            term_counts[t] += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Text annotation summary for {label}\n")
        f.write(f"# total_samples: {total}\n")
        f.write(f"# samples_with_any_terms: {samples_with_any_terms}\n")
        f.write("# term\tcount\ttotal_pct\n")
        for term, cnt in sorted(term_counts.items(), key=lambda kv: kv[1], reverse=True):
            pct = 100.0 * cnt / total if total > 0 else 0.0
            f.write(f"{term}\t{cnt}\t{pct:.2f}\n")

    print(f"Wrote {len(term_counts)} terms for {label} to {out_path}")


def summarise_ibs():
    """Summarise text annotations for IBS runs."""
    records = load_ibs_metadata()
    if not records:
        print("No IBS metadata records found; skipping IBS summary.")
        return
    run_ids = [r[0] for r in records]
    run_to_terms = shared_utils.parse_run_terms()
    out_path = os.path.join("data", "IBS", "text_annotation_summary.txt")
    summarise_terms_for_ids(run_ids, run_to_terms, out_path, label="IBS")


def summarise_infants():
    """Summarise text annotations for infants samples."""
    meta = load_infants_meta()
    if not meta:
        print("No infants metadata found; skipping infants summary.")
        return
    sample_ids = list(meta.keys())
    run_to_terms = shared_utils.parse_run_terms()
    out_path = os.path.join("data", "infants", "text_annotation_summary.txt")
    summarise_terms_for_ids(sample_ids, run_to_terms, out_path, label="infants")


def main():
    summarise_ibs()
    summarise_infants()


if __name__ == "__main__":
    main()

