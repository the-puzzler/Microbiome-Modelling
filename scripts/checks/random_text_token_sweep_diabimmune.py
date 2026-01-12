#!/usr/bin/env python3
"""
DIABIMMUNE dropout: zero tokens vs real text (no-text checkpoint only).

We focus on the OTU-only checkpoint:
  data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt

Using the DIABIMMUNE dropout setup (subject–age groups, all ordered pairs),
we evaluate four conditions on the *same examples*:

  1) baseline:   OTUs only (no text, no extra tokens)
  2) +4 zeros:   OTUs + 4 zero-valued "scratch" tokens
  3) +text:      OTUs + real LM text tokens for that SRS
  4) +text+4z:   OTUs + real text tokens + 4 zero-valued tokens

In all cases we:
  - run the transformer on the combined sequence,
  - read out logits only for the OTU positions,
  - use union-average aggregation across SRS per (subject, age) group,
  - build dropout labels from (t1, t2) ordered pairs,
  - compute ROC AUC / AP for predicting dropout from logits.
"""

import os
import sys
import csv
from collections import defaultdict

import h5py
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import load_run_data as load_diabimmune_run_data  # noqa: E402


CKPT_NO_TEXT = "data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt"
ZERO_TOKENS = 16


def build_diabimmune_grouping():
    """
    Build DIABIMMUNE subject–age grouping for dropout, mirroring the main
    dropout_test / compare_text_effect_dropout logic.

    Returns:
        micro_to_otus: dict SRS -> list[otu_id]
        subject_time_to_srs: dict (subject, age) -> list[SRS]
        pairs_by_subject: dict subject -> list[(age1, age2)] for all ordered pairs
        SRA_to_micro: mapping RunID -> SRS
    """
    run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_diabimmune_run_data()
    print(f"[diabimmune] runs: {len(run_rows)} | mapped to SRS: {len(SRA_to_micro)}")

    samples_csv = "data/diabimmune/samples.csv"
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

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    subject_time_to_srs = defaultdict(list)
    subjects = set()
    for srs, info in micro_to_sample.items():
        subject = info.get("subject")
        sample_id = info.get("sample")
        if not subject or not sample_id:
            continue
        rec = samples_table.get(sample_id, {})
        age = safe_float(rec.get("age_at_collection", ""))
        if age is None:
            continue
        subject_time_to_srs[(subject, age)].append(srs)
        subjects.add(subject)

    print("[diabimmune] subjects with age-resolved samples:", len(subjects))
    print("[diabimmune] subject-time groups:", len(subject_time_to_srs))

    needed_srs = {s for srs_list in subject_time_to_srs.values() for s in srs_list}
    micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
    print("[diabimmune] SRS with OTUs:", len(micro_to_otus))

    # Build ordered timepoint pairs per subject
    def safe_float_time(t):
        try:
            return float(t)
        except Exception:
            return None

    pairs_by_subject = {}
    total_pairs = 0
    for subject in subjects:
        times = sorted(
            {age for (subj, age) in subject_time_to_srs.keys() if subj == subject},
            key=lambda x: (safe_float_time(x) is None, safe_float_time(x), x),
        )
        if len(times) < 2:
            continue
        pairs = [(t1, t2) for t1 in times for t2 in times if t1 != t2]
        if pairs:
            pairs_by_subject[subject] = pairs
            total_pairs += len(pairs)

    print("[diabimmune] subjects with ≥2 timepoints:", len(pairs_by_subject), "| total pairs:", total_pairs)
    return micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro


def compute_dropout_metrics(subject_time_scores, subject_time_presence, pairs_by_subject):
    """
    Convert group-level OTU scores into dropout labels/scores and compute AUC/AP.
    """
    y_true = []
    y_score = []
    for subj, pairs in pairs_by_subject.items():
        for t1, t2 in pairs:
            s1 = subject_time_scores.get((subj, t1), {})
            p2 = subject_time_presence.get((subj, t2), set())
            for oid, sc in s1.items():
                # Dropout label: 1 if present at t1 but absent at t2
                y_true.append(1 if oid not in p2 else 0)
                y_score.append(sc)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_drop = y_true
    y_score = np.asarray(y_score, dtype=np.float32)

    if y_true.size and np.unique(y_true).size > 1:
        auc = roc_auc_score(y_drop, -y_score)
        probs_drop = 1 - 1 / (1 + np.exp(-y_score))
        ap = average_precision_score(y_drop, probs_drop)
    else:
        auc = float("nan")
        ap = float("nan")
    return auc, ap


def main():
    micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro = build_diabimmune_grouping()

    print(f"\n=== DIABIMMUNE dropout — OTU-only checkpoint: {CKPT_NO_TEXT} ===")
    model, device = shared_utils.load_microbiome_model(CKPT_NO_TEXT)

    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    # Real text mappings
    term_to_vec = shared_utils.load_term_embeddings(device=device)
    run_to_terms = shared_utils.parse_run_terms()
    srs_to_terms = shared_utils.build_srs_terms(SRA_to_micro, run_to_terms, shared_utils.MAPPED_PATH)

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]

        # --- Condition 1: baseline (no text, no zeros) ---
        print("\n[diabimmune] Condition: baseline (no text, no zeros)")
        st_scores_base = {}
        st_presence_base = {}
        for (subject, age), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc="[diabimmune] groups (baseline)",
        ):
            sample_dicts = []
            for srs in srs_list:
                sdict = shared_utils.score_otus_for_srs(
                    srs,
                    micro_to_otus=micro_to_otus,
                    resolver=resolver,
                    model=model,
                    device=device,
                    emb_group=emb_group,
                )
                if sdict:
                    sample_dicts.append(sdict)
            if not sample_dicts:
                st_scores_base[(subject, age)] = {}
                st_presence_base[(subject, age)] = set()
                continue
            avg = shared_utils.union_average_logits(sample_dicts)
            st_scores_base[(subject, age)] = avg
            st_presence_base[(subject, age)] = set(avg.keys())

        auc_base, ap_base = compute_dropout_metrics(st_scores_base, st_presence_base, pairs_by_subject)
        print(f"[diabimmune] baseline — AUC: {auc_base:.4f} | AP: {ap_base:.4f}")

        # --- Condition 2: +4 zero tokens (no text) ---
        print("\n[diabimmune] Condition: +4 zero tokens (no text)")
        st_scores_zero = {}
        st_presence_zero = {}
        for (subject, age), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc="[diabimmune] groups (+4 zeros)",
        ):
            sample_dicts = []
            for srs in srs_list:
                otu_ids = micro_to_otus.get(srs, [])
                if not otu_ids:
                    continue
                vecs = []
                keep = []
                for oid in otu_ids:
                    key = resolver.get(oid, oid) if resolver else oid
                    if key in emb_group:
                        vecs.append(torch.tensor(emb_group[key][()], dtype=torch.float32, device=device))
                        keep.append(oid)
                if not vecs:
                    continue
                x1 = torch.stack(vecs, dim=0).unsqueeze(0)  # (1, n1, OTU_EMB)
                n1 = x1.shape[1]
                with torch.no_grad():
                    h1 = model.input_projection_type1(x1)
                    if ZERO_TOKENS > 0:
                        z = torch.zeros((1, ZERO_TOKENS, shared_utils.D_MODEL), dtype=torch.float32, device=device)
                        h = torch.cat([h1, z], dim=1)
                    else:
                        h = h1
                    mask = torch.ones((1, h.shape[1]), dtype=torch.bool, device=device)
                    h_out = model.transformer(h, src_key_padding_mask=~mask)
                    logits_all = model.output_projection(h_out).squeeze(-1).squeeze(0).cpu().numpy()
                logits_map = dict(zip(keep, logits_all[: len(keep)]))
                if logits_map:
                    sample_dicts.append(logits_map)
            if not sample_dicts:
                st_scores_zero[(subject, age)] = {}
                st_presence_zero[(subject, age)] = set()
                continue
            avg = shared_utils.union_average_logits(sample_dicts)
            st_scores_zero[(subject, age)] = avg
            st_presence_zero[(subject, age)] = set(avg.keys())

        auc_zero, ap_zero = compute_dropout_metrics(st_scores_zero, st_presence_zero, pairs_by_subject)
        print(f"[diabimmune] +4 zeros — AUC: {auc_zero:.4f} | AP: {ap_zero:.4f} | ΔAUC vs baseline: {auc_zero - auc_base:+.4f}")

        # --- Condition 3: real text only (no zeros) ---
        print("\n[diabimmune] Condition: real text (no zeros)")
        st_scores_text = {}
        st_presence_text = {}
        for (subject, age), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc="[diabimmune] groups (+text)",
        ):
            sample_dicts = []
            for srs in srs_list:
                sdict = shared_utils.score_otus_for_srs_with_text(
                    srs,
                    micro_to_otus=micro_to_otus,
                    resolver=resolver,
                    model=model,
                    device=device,
                    emb_group=emb_group,
                    term_to_vec=term_to_vec,
                    srs_to_terms=srs_to_terms,
                )
                if sdict:
                    sample_dicts.append(sdict)
            if not sample_dicts:
                st_scores_text[(subject, age)] = {}
                st_presence_text[(subject, age)] = set()
                continue
            avg = shared_utils.union_average_logits(sample_dicts)
            st_scores_text[(subject, age)] = avg
            st_presence_text[(subject, age)] = set(avg.keys())

        auc_text, ap_text = compute_dropout_metrics(st_scores_text, st_presence_text, pairs_by_subject)
        print(f"[diabimmune] +text — AUC: {auc_text:.4f} | AP: {ap_text:.4f} | ΔAUC vs baseline: {auc_text - auc_base:+.4f}")

        # --- Condition 4: real text + 4 zero tokens ---
        print("\n[diabimmune] Condition: real text + 4 zero tokens")
        st_scores_text_zero = {}
        st_presence_text_zero = {}
        for (subject, age), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc="[diabimmune] groups (+text +4 zeros)",
        ):
            sample_dicts = []
            for srs in srs_list:
                otu_ids = micro_to_otus.get(srs, [])
                if not otu_ids:
                    continue
                vecs = []
                keep = []
                for oid in otu_ids:
                    key_emb = resolver.get(oid, oid) if resolver else oid
                    if key_emb in emb_group:
                        vecs.append(torch.tensor(emb_group[key_emb][()], dtype=torch.float32, device=device))
                        keep.append(oid)
                if not vecs:
                    continue

                # Build text token vectors for this SRS
                terms = [t for t in sorted(srs_to_terms.get(srs, set())) if t in term_to_vec]
                if terms:
                    x2_raw = torch.stack([term_to_vec[t] for t in terms], dim=0).unsqueeze(0)
                else:
                    x2_raw = torch.zeros((1, 0, shared_utils.TXT_EMB), dtype=torch.float32, device=device)

                x1 = torch.stack(vecs, dim=0).unsqueeze(0)
                n1 = x1.shape[1]
                with torch.no_grad():
                    h1 = model.input_projection_type1(x1)
                    h2 = model.input_projection_type2(x2_raw)
                    h = torch.cat([h1, h2], dim=1)
                    if ZERO_TOKENS > 0:
                        z = torch.zeros((1, ZERO_TOKENS, shared_utils.D_MODEL), dtype=torch.float32, device=device)
                        h = torch.cat([h, z], dim=1)
                    mask = torch.ones((1, h.shape[1]), dtype=torch.bool, device=device)
                    h_out = model.transformer(h, src_key_padding_mask=~mask)
                    logits_all = model.output_projection(h_out).squeeze(-1).squeeze(0).cpu().numpy()

                logits_map = dict(zip(keep, logits_all[: len(keep)]))
                if logits_map:
                    sample_dicts.append(logits_map)

            if not sample_dicts:
                st_scores_text_zero[(subject, age)] = {}
                st_presence_text_zero[(subject, age)] = set()
                continue
            avg = shared_utils.union_average_logits(sample_dicts)
            st_scores_text_zero[(subject, age)] = avg
            st_presence_text_zero[(subject, age)] = set(avg.keys())

        auc_text_zero, ap_text_zero = compute_dropout_metrics(st_scores_text_zero, st_presence_text_zero, pairs_by_subject)
        print(
            f"[diabimmune] +text+4z — AUC: {auc_text_zero:.4f} | AP: {ap_text_zero:.4f} | "
            f"ΔAUC vs baseline: {auc_text_zero - auc_base:+.4f}"
        )

    print("\n=== Summary (DIABIMMUNE dropout, no-text checkpoint) ===")
    print("Condition\tAUC\tAP")
    print(f"baseline\t{auc_base:.4f}\t{ap_base:.4f}")
    print(f"+16zeros \t{auc_zero:.4f}\t{ap_zero:.4f}")
    print(f"+text    \t{auc_text:.4f}\t{ap_text:.4f}")
    print(f"+text+16z \t{auc_text_zero:.4f}\t{ap_text_zero:.4f}")


if __name__ == "__main__":
    main()

