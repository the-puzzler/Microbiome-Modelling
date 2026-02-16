#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402
from scripts.rollout_metropolis.core import EmbeddingCache, build_otu_index, score_logits_for_sets, sigmoid  # noqa: E402


DEFAULT_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob_oneoutoneinanchored.tsv")


def parse_args():
    p = argparse.ArgumentParser(
        description="DIABIMMUNE: sweep anchor probability threshold vs anchored fraction on step-0 states."
    )
    p.add_argument("--tsv", default=DEFAULT_TSV, help="Rollout TSV (uses step==0 rows).")
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature used in sigmoid(logit/T).")
    p.add_argument(
        "--thresholds",
        default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99",
        help="Comma-separated p_anchor thresholds to evaluate.",
    )
    p.add_argument("--max-starts", type=int, default=0, help="Limit number of step-0 starts (0 = all in TSV).")
    p.add_argument("--batch", type=int, default=8, help="Sets per model forward pass.")
    return p.parse_args()


def _parse_index_list(raw):
    if not raw:
        return []
    return [int(tok) for tok in str(raw).split(";") if str(tok).strip()]


def load_step0_sets(tsv_path, *, max_starts=0):
    keys = []
    idx_lists = []
    sizes = []
    with open(tsv_path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row.get("step", "").strip() != "0":
                continue
            subj = row.get("subject", "").strip()
            t_start = row.get("t_start", "").strip()
            if not subj or not t_start:
                continue
            idx = _parse_index_list(row.get("current_otu_indices", ""))
            keys.append((subj, t_start))
            idx_lists.append(idx)
            sizes.append(len(idx))
            if max_starts and len(keys) >= int(max_starts):
                break
    return keys, idx_lists, np.asarray(sizes, dtype=int)


def main():
    args = parse_args()
    if not os.path.exists(args.tsv):
        raise SystemExit(f"Missing TSV: {args.tsv}")

    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if str(x).strip()]
    thresholds = sorted(set(thresholds))
    if not thresholds:
        raise SystemExit("No thresholds provided.")

    keys, idx_lists, sizes = load_step0_sets(args.tsv, max_starts=args.max_starts)
    if not keys:
        raise SystemExit("No step-0 rows found in TSV.")

    _run_rows, sra_to_micro, _gid_to_sample, micro_to_subject, _micro_to_sample = load_run_data()
    micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)
    all_otus, _otu_to_idx = build_otu_index(micro_to_otus)
    if not all_otus:
        raise SystemExit("No OTUs available for index mapping.")

    # Convert indices -> OTU ids; filter out invalid indices.
    sets = []
    for idx in idx_lists:
        otus = [all_otus[i] for i in idx if 0 <= int(i) < len(all_otus)]
        sets.append(sorted(set(otus)))

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = (
        shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")
        if rename_map
        else {}
    )

    anchor_counts = np.zeros((len(sets), len(thresholds)), dtype=np.int32)
    droppable_counts = np.zeros((len(sets), len(thresholds)), dtype=np.int32)
    valid_sizes = np.asarray([len(s) for s in sets], dtype=np.int32)

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        emb_cache = EmbeddingCache()

        bsz = max(1, int(args.batch))
        for i0 in tqdm(range(0, len(sets), bsz), desc="Scoring step-0 sets", unit="batch", dynamic_ncols=True):
            chunk = sets[i0 : i0 + bsz]
            scored = score_logits_for_sets(chunk, model, device, emb_group, resolver, emb_cache=emb_cache)
            for j, logits in enumerate(scored):
                i = i0 + j
                # probs for OTUs actually scored (should match input set filtered by embeddings)
                probs = []
                for _otu, logit in logits.items():
                    probs.append(float(sigmoid(float(logit) / float(args.temperature))))
                probs = np.asarray(probs, dtype=np.float32)
                # If some OTUs are missing embeddings, probs array is shorter; treat missing as p=0.
                # (This matches pick_anchor_set behavior: only logits-present OTUs can become anchors.)
                for t_i, thr in enumerate(thresholds):
                    na = int(np.sum(probs >= float(thr)))
                    anchor_counts[i, t_i] = na
                    droppable_counts[i, t_i] = max(0, int(valid_sizes[i]) - na)

    # Summarize
    def q(arr, p):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.quantile(arr, p))

    print("tsv\tstarts\tthreshold\tpct_fully_anchored\tmedian_anchor_frac\tp90_anchor_frac\tmedian_droppable")
    for t_i, thr in enumerate(thresholds):
        nc = np.maximum(valid_sizes, 1)
        frac = anchor_counts[:, t_i].astype(float) / nc.astype(float)
        fully = float(np.mean(anchor_counts[:, t_i] >= valid_sizes)) * 100.0
        med_drop = float(np.median(droppable_counts[:, t_i]))
        print(
            f"{args.tsv}\t{len(sets)}\t{thr:.5f}\t{fully:.1f}\t{q(frac,0.5):.3f}\t{q(frac,0.9):.3f}\t{med_drop:.1f}"
        )


if __name__ == "__main__":
    main()
