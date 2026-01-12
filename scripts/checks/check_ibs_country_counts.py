#!/usr/bin/env python3
"""
Quick sanity-check script for IBS sample counts per country.

It mirrors the logic in scripts/IBS/predict_ibs.py:
  - Load IBS metadata (filtering out self-diagnosed etc.).
  - Map runs to MicrobeAtlas SRS.
  - Build sample embeddings for those SRS.
  - Assemble per-country datasets (only samples that have embeddings).

Then it prints, for each country used in the task:
  - Number of samples actually entering the IBS prediction experiment.
With special emphasis on USA, United Kingdom, and Australia.
"""

import os
import sys

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.IBS.predict_ibs import (  # noqa: E402
    load_ibs_metadata,
    map_runs_to_srs,
    build_sample_embeddings_for_runs,
    assemble_country_datasets,
)


def main():
    records = load_ibs_metadata()
    if not records:
        print("No IBS records after filtering; nothing to count.")
        return

    run_ids = [r[0] for r in records]
    run_to_srs = map_runs_to_srs(run_ids)
    if not run_to_srs:
        print("No IBS runs could be mapped to SRS; nothing to count.")
        return

    sample_embeddings = build_sample_embeddings_for_runs(run_to_srs)
    if not sample_embeddings:
        print("No embeddings built for IBS SRS; nothing to count.")
        return

    datasets = assemble_country_datasets(records, run_to_srs, sample_embeddings)
    if not datasets:
        print("No per-country datasets assembled; nothing to count.")
        return

    print("\n=== IBS sample counts per country (used in task) ===")
    for country, (X, y) in sorted(datasets.items()):
        n = int(X.shape[0])
        pos = int(np.asarray(y, dtype=int).sum())
        neg = int((np.asarray(y, dtype=int) == 0).sum())
        print(f"{country:15s}  total={n:4d}  IBS={pos:3d}  Control={neg:3d}")

    # Highlight the main three explicitly
    print("\n=== Focus countries ===")
    for country in ["USA", "United Kingdom", "Australia"]:
        if country in datasets:
            X, y = datasets[country]
            n = int(X.shape[0])
            pos = int(np.asarray(y, dtype=int).sum())
            neg = int((np.asarray(y, dtype=int) == 0).sum())
            print(f"{country:15s}  total={n:4d}  IBS={pos:3d}  Control={neg:3d}")
        else:
            print(f"{country:15s}  (no samples used)")


if __name__ == "__main__":
    main()

