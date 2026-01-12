#!/usr/bin/env python3
"""
Check how a checkpoint reacts to RANDOM text embeddings vs real text.

For a given checkpoint:
  - Select a subset of SRS that have both OTUs and text terms.
  - For each SRS, compute OTU logits under:
        (1) no text              -> score_otus_for_srs
        (2) real text embeddings -> score_otus_for_srs_with_text (term_to_vec_real)
        (3) random text embeddings -> score_otus_for_srs_with_text (term_to_vec_rand)
  - For each SRS, compute:
        delta_real = mean |logit_real - logit_no|
        delta_rand = mean |logit_rand - logit_no|

If the model truly learned to use the *content* of text embeddings (not just the
presence of extra tokens), you'd expect the behaviour under real vs random text
to look qualitatively different.
"""

import os
import sys
import random

import numpy as np
import h5py
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402


# Default checkpoint path: edit this to test a different model
CKPT_PATH = 'data/model/checkpoint_epoch_0_final_newblack_2epoch.pt' #_notextabl
MAPPED_PATH = shared_utils.MAPPED_PATH


def main(n_srs=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    if len(candidate_srs) > n_srs:
        candidate_srs = random.sample(candidate_srs, n_srs)
    print("Using", len(candidate_srs), "SRS for the check.")

    print("Collecting OTUs for selected SRS...")
    micro_to_otus = shared_utils.collect_micro_to_otus_mapped(set(candidate_srs), mapped_path=MAPPED_PATH)
    print("SRS with OTUs among selected:", len(micro_to_otus))

    # Resolver for OTU keys
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    # Real term embeddings
    term_to_vec_real = shared_utils.load_term_embeddings(device=device)

    # Build random term embedding dictionary with same keys / dimensionality
    print("Building random term embeddings with same keys/dimension as real ones...")
    # Use TXT_EMB as dimension; real vectors should be that size
    dim = shared_utils.TXT_EMB
    term_to_vec_rand = {}
    for term in term_to_vec_real.keys():
        term_to_vec_rand[term] = torch.randn(dim, device=device, dtype=torch.float32)

    print("Scoring SRS under no-text, real-text, and random-text conditions...")
    deltas_real = []
    deltas_rand = []

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
                term_to_vec=term_to_vec_real,
                srs_to_terms=srs_to_terms,
            )

            # With random text (same term structure, random vectors)
            logits_rand = shared_utils.score_otus_for_srs_with_text(
                srs,
                micro_to_otus=micro_to_otus,
                resolver=resolver,
                model=model,
                device=device,
                emb_group=emb_group,
                term_to_vec=term_to_vec_rand,
                srs_to_terms=srs_to_terms,
            )

            keys = list(logits_no.keys())
            base_vals = np.array([logits_no[k] for k in keys], dtype=float)

            if logits_real:
                vals_real = np.array([logits_real.get(k, np.nan) for k in keys], dtype=float)
                mask = np.isfinite(vals_real)
                if mask.any():
                    deltas_real.append(float(np.nanmean(np.abs(vals_real[mask] - base_vals[mask]))))

            if logits_rand:
                vals_rand = np.array([logits_rand.get(k, np.nan) for k in keys], dtype=float)
                mask = np.isfinite(vals_rand)
                if mask.any():
                    deltas_rand.append(float(np.nanmean(np.abs(vals_rand[mask] - base_vals[mask]))))

    if deltas_real and deltas_rand:
        print("\nMean |logit_real - logit_no_text| across SRS:", float(np.mean(deltas_real)))
        print("Mean |logit_random - logit_no_text| across SRS:", float(np.mean(deltas_rand)))
        print("Std real:", float(np.std(deltas_real)), "| Std random:", float(np.std(deltas_rand)))
        print("\nIf real-text and random-text deltas look similar, the model is likely not")
        print("using the specific LM embeddings in a meaningful way. If they differ")
        print("systematically, that suggests the model is sensitive to actual term content.")
    else:
        print("Not enough valid SRS to compute deltas; check OTU and term coverage.")


if __name__ == "__main__":
    main()

