#!/usr/bin/env python3
"""
Surgical tests for how text tokens affect dropout performance (gingivitis).

For each checkpoint:
  - OTU-only:  data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt
  - Text-trained: data/model/checkpoint_epoch_0_final_newblack_2epoch.pt

We run the gingivitis dropout task under several conditions:
  1) no-text        : OTUs only (baseline, uses shared_utils.score_otus_for_srs)
  2) real-text      : OTUs + real LM term embeddings (replicates existing code)
  3) blocked-text   : OTUs + text embeddings are built, but transformer sees OTUs only
  4) zero-text      : OTUs + extra zero-valued text tokens (same positions, no content)
  5) shuffled-text  : OTUs + real LM embeddings but with SRS->terms mapping shuffled

All conditions use the same SRS, same (subject,time) groups, and same T1->T2
pairs; only the way we build the transformer inputs is changed.
"""

import os
import sys
import csv
from collections import defaultdict
import random

import h5py
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.gingivitis.utils import (  # noqa: E402
    load_gingivitis_run_data,
    collect_micro_to_otus as collect_ging_micro_to_otus,
)


CKPT_NO_TEXT = "data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt"
CKPT_WITH_TEXT = "data/model/checkpoint_epoch_0_final_newblack_2epoch.pt"


def build_gingivitis_grouping():
    gingivitis_csv = "data/gingivitis/gingiva.csv"
    run_ids, SRA_to_micro = load_gingivitis_run_data(
        gingivitis_path=gingivitis_csv,
        microbeatlas_path=shared_utils.MICROBEATLAS_SAMPLES,
    )
    print(f"[setup] gingivitis runs: {len(run_ids)} | mapped to SRS: {len(SRA_to_micro)}")

    micro_to_otus = collect_ging_micro_to_otus(SRA_to_micro)
    print("[setup] gingivitis SRS with OTUs:", len(micro_to_otus))

    records = []
    with open(gingivitis_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("Run", "").strip()
            subj = row.get("subject_code", "").strip()
            tcode = row.get("time_code", "").strip()
            if not run or not subj or not tcode:
                continue
            srs = SRA_to_micro.get(run)
            if not srs:
                continue
            records.append({"run": run, "srs": srs, "subject": subj, "time": tcode})

    subject_time_to_srs = defaultdict(list)
    for r in records:
        subject_time_to_srs[(r["subject"], r["time"])].append(r["srs"])

    subjects = sorted({r["subject"] for r in records})
    print("[setup] subjects with mapped runs:", len(subjects))
    print("[setup] subject-time groups:", len(subject_time_to_srs))

    def safe_float_time(t):
        try:
            return float(t)
        except Exception:
            return None

    pairs_by_subject = {}
    total_pairs = 0
    for subject in subjects:
        times = sorted(
            {t for (s, t) in subject_time_to_srs.keys() if s == subject},
            key=lambda x: (safe_float_time(x) is None, safe_float_time(x), x),
        )
        if len(times) < 2:
            continue
        pairs = [(t1, t2) for t1 in times for t2 in times if t1 != t2]
        if pairs:
            pairs_by_subject[subject] = pairs
            total_pairs += len(pairs)

    print("[setup] subjects with ≥2 timepoints:", len(pairs_by_subject), "| total pairs:", total_pairs)
    return micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro


def score_with_mode(
    srs,
    mode,
    micro_to_otus,
    resolver,
    model,
    device,
    emb_group,
    term_to_vec=None,
    srs_to_terms=None,
    srs_to_terms_shuffled=None,
):
    """
    mode:
      - 'real'     : OTUs + real text, normal mixing
      - 'blocked'  : OTUs only; text embeddings built but not passed to transformer
      - 'zero'     : OTUs + zero text tokens
      - 'shuffled' : OTUs + real text, but SRS->terms mapping permuted
    """
    otu_ids = micro_to_otus.get(srs, [])
    if not otu_ids:
        return {}
    vecs = []
    keep = []
    for oid in otu_ids:
        key = resolver.get(oid, oid) if resolver else oid
        if key in emb_group:
            vecs.append(torch.tensor(emb_group[key][()], dtype=torch.float32, device=device))
            keep.append(oid)
    if not vecs:
        return {}
    x1 = torch.stack(vecs, dim=0).unsqueeze(0)  # (1, n1, OTU_EMB)
    n1 = x1.shape[1]

    # Build text embeddings
    if mode in ("real", "zero", "shuffled"):
        if srs_to_terms is None or term_to_vec is None:
            x2 = torch.zeros((1, 0, shared_utils.TXT_EMB), dtype=torch.float32, device=device)
        else:
            if mode == "shuffled" and srs_to_terms_shuffled is not None:
                terms_src = srs_to_terms_shuffled
            else:
                terms_src = srs_to_terms
            t_terms = [t for t in sorted(terms_src.get(srs, set())) if t in term_to_vec]
            if t_terms:
                t_vecs = [term_to_vec[t] for t in t_terms]
                x2 = torch.stack(t_vecs, dim=0).unsqueeze(0)
            else:
                x2 = torch.zeros((1, 0, shared_utils.TXT_EMB), dtype=torch.float32, device=device)
    else:
        # Should not happen
        x2 = torch.zeros((1, 0, shared_utils.TXT_EMB), dtype=torch.float32, device=device)

    n2 = x2.shape[1]

    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        if mode == "blocked":
            # Completely ignore text: no change from OTU-only
            h = h1
            mask = torch.ones((1, n1), dtype=torch.bool, device=device)
        elif mode == "zero":
            # Include extra positions, but zero content
            z2 = torch.zeros((1, n2, shared_utils.D_MODEL), dtype=torch.float32, device=device)
            h = torch.cat([h1, z2], dim=1)
            mask = torch.ones((1, n1 + n2), dtype=torch.bool, device=device)
        else:
            # 'real' or 'shuffled': standard text path
            h2 = model.input_projection_type2(x2)
            h = torch.cat([h1, h2], dim=1)
            mask = torch.ones((1, n1 + n2), dtype=torch.bool, device=device)

        h = model.transformer(h, src_key_padding_mask=~mask)
        logits = model.output_projection(h).squeeze(-1)  # (1, n1(+n2))

    logits_type1 = logits[:, :n1].squeeze(0).cpu().numpy()
    return dict(zip(keep, logits_type1))


def eval_checkpoint(checkpoint_path, micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro, label):
    print(f"\n=== Gingivitis dropout — checkpoint: {checkpoint_path} ({label}) ===")
    model, device = shared_utils.load_microbiome_model(checkpoint_path)

    # Resolver
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = None
    if rename_map:
        resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    # Baseline: no text (shared_utils.score_otus_for_srs)
    subject_time_otu_scores_base = {}
    subject_time_otu_presence_base = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for (subject, time_code), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc=f"[{label}] groups (no-text)",
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
                subject_time_otu_scores_base[(subject, time_code)] = {}
                subject_time_otu_presence_base[(subject, time_code)] = set()
                continue
            avg_scores = shared_utils.union_average_logits(sample_dicts)
            subject_time_otu_scores_base[(subject, time_code)] = avg_scores
            subject_time_otu_presence_base[(subject, time_code)] = set(avg_scores.keys())

    y_true_base = []
    y_score_base = []
    for subject, pairs in pairs_by_subject.items():
        for t1, t2 in pairs:
            scores_t1 = subject_time_otu_scores_base.get((subject, t1), {})
            present_t2 = subject_time_otu_presence_base.get((subject, t2), set())
            for otu, score in scores_t1.items():
                y_true_base.append(1 if otu in present_t2 else 0)
                y_score_base.append(score)

    y_true_base = np.asarray(y_true_base, dtype=np.int64)
    y_drop_base = 1 - y_true_base
    y_score_base = np.asarray(y_score_base, dtype=np.float32)
    auc_base = roc_auc_score(y_drop_base, -y_score_base) if y_true_base.size else float("nan")
    ap_base = average_precision_score(y_drop_base, 1 - 1 / (1 + np.exp(-y_score_base))) if y_true_base.size else float("nan")
    print(f"[{label}] no-text (baseline) — AUC: {auc_base:.4f} | AP: {ap_base:.4f}")

    # Shared term mappings
    term_to_vec_real = shared_utils.load_term_embeddings(device=device)
    run_to_terms = shared_utils.parse_run_terms()
    run_to_srs = SRA_to_micro
    srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)

    # Shuffled SRS->terms mapping
    srs_keys = list(srs_to_terms.keys())
    shuffled_keys = list(srs_keys)
    random.shuffle(shuffled_keys)
    srs_to_terms_shuffled = {srs: srs_to_terms[shuffled_keys[i]] for i, srs in enumerate(srs_keys)}

    # Evaluate text-based modes
    results = {
        "no-text": (auc_base, ap_base),
    }

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]

        for mode in ["real", "blocked", "zero", "shuffled"]:
            label_mode = {
                "real": "real-text",
                "blocked": "blocked-text",
                "zero": "zero-text",
                "shuffled": "shuffled-text",
            }[mode]
            print(f"[{label}] scoring groups ({label_mode})...")

            subject_time_scores = {}
            subject_time_presence = {}
            for (subject, time_code), srs_list in tqdm(
                list(subject_time_to_srs.items()),
                desc=f"[{label}] groups ({label_mode})",
            ):
                sample_dicts = []
                for srs in srs_list:
                    sdict = score_with_mode(
                        srs,
                        mode=mode,
                        micro_to_otus=micro_to_otus,
                        resolver=resolver,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        term_to_vec=term_to_vec_real,
                        srs_to_terms=srs_to_terms,
                        srs_to_terms_shuffled=srs_to_terms_shuffled,
                    )
                    if sdict:
                        sample_dicts.append(sdict)
                if not sample_dicts:
                    subject_time_scores[(subject, time_code)] = {}
                    subject_time_presence[(subject, time_code)] = set()
                    continue
                avg_scores = shared_utils.union_average_logits(sample_dicts)
                subject_time_scores[(subject, time_code)] = avg_scores
                subject_time_presence[(subject, time_code)] = set(avg_scores.keys())

            y_true = []
            y_score = []
            for subject, pairs in pairs_by_subject.items():
                for t1, t2 in pairs:
                    scores_t1 = subject_time_scores.get((subject, t1), {})
                    present_t2 = subject_time_presence.get((subject, t2), set())
                    for otu, score in scores_t1.items():
                        y_true.append(1 if otu in present_t2 else 0)
                        y_score.append(score)

            y_true = np.asarray(y_true, dtype=np.int64)
            y_drop = 1 - y_true
            y_score = np.asarray(y_score, dtype=np.float32)
            if y_true.size and np.unique(y_true).size > 1:
                auc = roc_auc_score(y_drop, -y_score)
                ap = average_precision_score(y_drop, 1 - 1 / (1 + np.exp(-y_score)))
            else:
                auc = float("nan")
                ap = float("nan")
            print(f"[{label}] {label_mode} — AUC: {auc:.4f} | AP: {ap:.4f} | ΔAUC vs baseline: {auc - auc_base:+.4f}")
            results[label_mode] = (auc, ap)

    return results


def main():
    micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro = build_gingivitis_grouping()

    res_no_text_ckpt = eval_checkpoint(
        CKPT_NO_TEXT,
        micro_to_otus,
        subject_time_to_srs,
        pairs_by_subject,
        SRA_to_micro,
        label="no-text checkpoint",
    )
    res_text_ckpt = eval_checkpoint(
        CKPT_WITH_TEXT,
        micro_to_otus,
        subject_time_to_srs,
        pairs_by_subject,
        SRA_to_micro,
        label="text-trained checkpoint",
    )

    print("\n=== Summary (AUC, gingivitis dropout) ===")
    print("Checkpoint\tCondition\tAUC")
    for ckpt_label, res in [
        ("no-text", res_no_text_ckpt),
        ("text-trained", res_text_ckpt),
    ]:
        for cond in ["no-text", "real-text", "blocked-text", "zero-text", "shuffled-text"]:
            auc = res.get(cond, (float("nan"),))[0]
            print(f"{ckpt_label}\t{cond}\t{auc:.4f}")


if __name__ == "__main__":
    main()
