#!/usr/bin/env python3
"""
Sweep the number of additional zero-valued "scratchpad" tokens when REAL text
is present for gingivitis dropout, to see how much extra capacity helps,
especially for the text-trained checkpoint.

Conditions:
  - Always use OTUs + real LM term embeddings for each SRS.
  - Additionally append N zero-valued tokens (in hidden space) per SRS.

We evaluate:
  - OTU-only checkpoint
  - Text-trained checkpoint

using the same gingivitis dropout grouping as the other scripts.
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


def score_with_real_text_and_zero_tokens(
    srs,
    micro_to_otus,
    resolver,
    model,
    device,
    emb_group,
    term_to_vec,
    srs_to_terms,
    n_zero_tokens,
):
    """
    Score OTUs for a single SRS with:
      - OTUs + real LM term embeddings
      - plus n_zero_tokens zero-valued hidden slots appended at the end.
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

    # Build real text embeddings for this SRS
    t_terms = [t for t in sorted(srs_to_terms.get(srs, set())) if t in term_to_vec]
    if t_terms:
        t_vecs = [term_to_vec[t] for t in t_terms]
        x2 = torch.stack(t_vecs, dim=0).unsqueeze(0)  # (1, n2, TXT_EMB)
    else:
        x2 = torch.zeros((1, 0, shared_utils.TXT_EMB), dtype=torch.float32, device=device)
    n2 = x2.shape[1]

    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        h2 = model.input_projection_type2(x2)
        h = torch.cat([h1, h2], dim=1)  # (1, n1+n2, d_model)

        if n_zero_tokens > 0:
            z3 = torch.zeros((1, n_zero_tokens, shared_utils.D_MODEL), dtype=torch.float32, device=device)
            h = torch.cat([h, z3], dim=1)
            mask = torch.ones((1, n1 + n2 + n_zero_tokens), dtype=torch.bool, device=device)
        else:
            mask = torch.ones((1, n1 + n2), dtype=torch.bool, device=device)

        h = model.transformer(h, src_key_padding_mask=~mask)
        logits = model.output_projection(h).squeeze(-1)  # (1, n1+n2(+n_zero))

    logits_type1 = logits[:, :n1].squeeze(0).cpu().numpy()
    return dict(zip(keep, logits_type1))


def eval_checkpoint_for_sweep(
    checkpoint_path,
    micro_to_otus,
    subject_time_to_srs,
    pairs_by_subject,
    SRA_to_micro,
    n_zero_list,
    label,
):
    print(f"\n=== Gingivitis dropout (text + zero) — checkpoint: {checkpoint_path} ({label}) ===")
    model, device = shared_utils.load_microbiome_model(checkpoint_path)

    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = None
    if rename_map:
        resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    # Shared term mappings
    term_to_vec_real = shared_utils.load_term_embeddings(device=device)
    run_ids, SRA_to_micro2 = load_gingivitis_run_data()
    run_to_srs = SRA_to_micro2
    run_to_terms = shared_utils.parse_run_terms()
    srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)

    results = {}

    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]

        for n_zero in n_zero_list:
            print(f"[{label}] scoring with real text + {n_zero} zero tokens per SRS...")
            subject_time_scores = {}
            subject_time_presence = {}

            for (subject, time_code), srs_list in tqdm(
                list(subject_time_to_srs.items()),
                desc=f"[{label}] groups (n_zero={n_zero})",
            ):
                sample_dicts = []
                for srs in srs_list:
                    sdict = score_with_real_text_and_zero_tokens(
                        srs,
                        micro_to_otus=micro_to_otus,
                        resolver=resolver,
                        model=model,
                        device=device,
                        emb_group=emb_group,
                        term_to_vec=term_to_vec_real,
                        srs_to_terms=srs_to_terms,
                        n_zero_tokens=n_zero,
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
            print(f"[{label}] n_zero={n_zero} — AUC: {auc:.4f} | AP: {ap:.4f}")
            results[n_zero] = (auc, ap)

    return results


def main():
    micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro = build_gingivitis_grouping()

    # Sweep over different numbers of zero scratch tokens appended after real text
    n_zero_list = [0, 2, 4, 8, 16, 32]

    res_no_text_ckpt = eval_checkpoint_for_sweep(
        CKPT_NO_TEXT,
        micro_to_otus,
        subject_time_to_srs,
        pairs_by_subject,
        SRA_to_micro,
        n_zero_list,
        label="no-text checkpoint",
    )
    res_text_ckpt = eval_checkpoint_for_sweep(
        CKPT_WITH_TEXT,
        micro_to_otus,
        subject_time_to_srs,
        pairs_by_subject,
        SRA_to_micro,
        n_zero_list,
        label="text-trained checkpoint",
    )

    print("\n=== Sweep summary (AUC, gingivitis dropout) — real text + zero tokens ===")
    print("Checkpoint\t#zero_tokens\tAUC")
    for ckpt_label, res in [
        ("no-text", res_no_text_ckpt),
        ("text-trained", res_text_ckpt),
    ]:
        for n_zero in n_zero_list:
            auc = res.get(n_zero, (float("nan"),))[0]
            print(f"{ckpt_label}\t{n_zero}\t{auc:.4f}")


if __name__ == "__main__":
    main()

