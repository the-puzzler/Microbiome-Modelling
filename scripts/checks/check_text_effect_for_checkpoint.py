#!/usr/bin/env python3
"""
Quick sanity check: does a given checkpoint "use" text tokens in a non-trivial way?

Heuristic approach:
  - Load a checkpoint (default: data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt)
  - Pick a subset of SRS that have both OTUs and text terms.
  - For each SRS, compute OTU logits under three conditions:
        (1) no text  -> score_otus_for_srs
        (2) real text -> score_otus_for_srs_with_text
        (3) shuffled text -> score_otus_for_srs_with_text with permuted term-sets
  - Compare how much logits change:
        delta_real   = mean |logits_real - logits_no_text|
        delta_shuf   = mean |logits_shuf - logits_no_text|

If the model genuinely learned to use text tokens, you'd expect:
  - The real-text perturbation to produce consistent, structured changes
    that differ from what you get with shuffled/random text.
  - At minimum, delta_real should not look indistinguishable from delta_shuf
    across many SRS.

This is not a proof, but it helps detect obviously random / untrained text paths.
"""

import os
import sys
import random

import numpy as np
import h5py


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


CKPT_PATH = "data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt" #2epoch_notextabl
MAPPED_PATH = shared_utils.MAPPED_PATH


def main(n_srs=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("Loading checkpoint:", CKPT_PATH)
    model, device = shared_utils.load_microbiome_model(CKPT_PATH)

    print("Building accession->SRS mapping from mapped file...")
    acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(MAPPED_PATH)
    srs_ids = sorted(set(acc_to_srs.values()))
    print("Total SRS in mapped file:", len(srs_ids))

    print("Loading term mappings...")
    run_to_terms = shared_utils.parse_run_terms()
    srs_to_terms = shared_utils.build_srs_terms(acc_to_srs, run_to_terms, mapped_path=MAPPED_PATH)
    print("SRS with any terms:", len(srs_to_terms))

    # Restrict to SRS that have terms
    candidate_srs = [srs for srs in srs_ids if srs in srs_to_terms]
    if not candidate_srs:
        print("No SRS with associated terms; nothing to test.")
        return

    # Limit to a subset
    if len(candidate_srs) > n_srs:
        candidate_srs = random.sample(candidate_srs, n_srs)
    print("Using", len(candidate_srs), "SRS for the check.")

    print("Collecting OTUs for selected SRS...")
    micro_to_otus = shared_utils.collect_micro_to_otus_mapped(set(candidate_srs), mapped_path=MAPPED_PATH)
    print("SRS with OTUs among selected:", len(micro_to_otus))

    # Build resolver for OTU keys (prefer B) if rename map exists
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    # Build a shuffled SRS->terms mapping
    srs_list_for_shuffle = list(srs_to_terms.keys())
    shuffled_srs_to_terms = {}
    perm = list(srs_list_for_shuffle)
    random.shuffle(perm)
    for srs, shuffled_src in zip(srs_list_for_shuffle, perm):
        shuffled_srs_to_terms[srs] = srs_to_terms[shuffled_src]

    print("Scoring SRS with and without text...")
    deltas_real = []
    deltas_shuf = []

    term_to_vec = shared_utils.load_term_embeddings()

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for srs in candidate_srs:
            if srs not in micro_to_otus:
                continue

            # Baseline: no text
            logits_no = shared_utils.score_otus_for_srs(
                srs,
                micro_to_otus=micro_to_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
            )
            if not logits_no:
                continue

            # With real text
            logits_real = shared_utils.score_otus_for_srs_with_text(
                srs,
                micro_to_otus=micro_to_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
                term_to_vec=term_to_vec,
                srs_to_terms=srs_to_terms,
            )

            # With shuffled text
            logits_shuf = shared_utils.score_otus_for_srs_with_text(
                srs,
                micro_to_otus=micro_to_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
                term_to_vec=term_to_vec,
                srs_to_terms=shuffled_srs_to_terms,
            )

            # Restrict comparisons to OTUs present in baseline
            keys = list(logits_no.keys())
            base_vals = np.array([logits_no[k] for k in keys], dtype=float)

            if logits_real:
                vals_real = np.array([logits_real.get(k, np.nan) for k in keys], dtype=float)
                mask = np.isfinite(vals_real)
                if mask.any():
                    deltas_real.append(float(np.nanmean(np.abs(vals_real[mask] - base_vals[mask]))))

            if logits_shuf:
                vals_shuf = np.array([logits_shuf.get(k, np.nan) for k in keys], dtype=float)
                mask = np.isfinite(vals_shuf)
                if mask.any():
                    deltas_shuf.append(float(np.nanmean(np.abs(vals_shuf[mask] - base_vals[mask]))))

    if deltas_real and deltas_shuf:
        print("\nMean |logit_real - logit_no_text| across SRS:", float(np.mean(deltas_real)))
        print("Mean |logit_shuffled - logit_no_text| across SRS:", float(np.mean(deltas_shuf)))
        print("Std real:", float(np.std(deltas_real)), "| Std shuffled:", float(np.std(deltas_shuf)))
        print("\nIf the model truly learned to use text, you might expect the real-text deltas")
        print("to behave differently from shuffled-text deltas (e.g., smaller, more structured,")
        print("or more strongly aligned with known term semantics). If they look similar, it")
        print("suggests the text pathway is effectively untrained/random for this checkpoint.")
    else:
        print("Not enough valid SRS to compute deltas; check OTU and term coverage.")


if __name__ == "__main__":
    main()

