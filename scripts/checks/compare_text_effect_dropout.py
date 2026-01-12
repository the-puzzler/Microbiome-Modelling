#!/usr/bin/env python3
"""
Compare text effects on dropout prediction for three datasets:
    - Gingivitis
    - Snowmelt
    - DIABIMMUNE

For each dataset, we evaluate two checkpoints:
  1) OTU-only model (never trained with text):
       data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt
  2) Text-trained model:
       data/model/checkpoint_epoch_0_final_newblack_2epoch.pt

For each checkpoint and dataset we run three conditions on the same examples:
  - No text
  - Real text (LM term embeddings)
  - Random text (same term structure, random vectors)

We report ROC AUC (and AP) so you can see, side by side, how text and random
text affect each model on each dropout task.
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
from scripts.snowmelt.utils import load_snowmelt_metadata  # noqa: E402
from scripts.diabimmune.utils import load_run_data as load_diabimmune_run_data  # noqa: E402


CKPT_NO_TEXT = "data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt"
CKPT_WITH_TEXT = "data/model/checkpoint_epoch_0_final_newblack_2epoch.pt"


# === GINGIVITIS HELPERS ===

def build_gingivitis_grouping():
    gingivitis_csv = "data/gingivitis/gingiva.csv"
    run_ids, SRA_to_micro = load_gingivitis_run_data(
        gingivitis_path=gingivitis_csv,
        microbeatlas_path=shared_utils.MICROBEATLAS_SAMPLES,
    )
    print(f"[gingivitis] runs: {len(run_ids)} | mapped to SRS: {len(SRA_to_micro)}")

    micro_to_otus = collect_ging_micro_to_otus(SRA_to_micro)
    print("[gingivitis] SRS with OTUs:", len(micro_to_otus))

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
    print("[gingivitis] subjects with mapped runs:", len(subjects))
    print("[gingivitis] subject-time groups:", len(subject_time_to_srs))

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

    print("[gingivitis] subjects with ≥2 timepoints:", len(pairs_by_subject), "| total pairs:", total_pairs)
    return micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro


def eval_gingivitis_checkpoint(checkpoint_path, micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro, label):
    print(f"\n=== Gingivitis — checkpoint: {checkpoint_path} ({label}) ===")
    model, device = shared_utils.load_microbiome_model(checkpoint_path)

    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = None
    if rename_map:
        resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    # No text
    subject_time_otu_scores = {}
    subject_time_otu_presence = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for (subject, time_code), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc=f"[gingivitis/{label}] groups (no text)",
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
                subject_time_otu_scores[(subject, time_code)] = {}
                subject_time_otu_presence[(subject, time_code)] = set()
                continue
            avg_scores = shared_utils.union_average_logits(sample_dicts)
            subject_time_otu_scores[(subject, time_code)] = avg_scores
            subject_time_otu_presence[(subject, time_code)] = set(avg_scores.keys())

    y_true_base = []
    y_score_base = []
    for subject, pairs in pairs_by_subject.items():
        for t1, t2 in pairs:
            scores_t1 = subject_time_otu_scores.get((subject, t1), {})
            present_t2 = subject_time_otu_presence.get((subject, t2), set())
            for otu, score in scores_t1.items():
                y_true_base.append(1 if otu in present_t2 else 0)
                y_score_base.append(score)

    y_true_base = np.asarray(y_true_base, dtype=np.int64)
    y_drop_base = 1 - y_true_base
    y_score_base = np.asarray(y_score_base, dtype=np.float32)
    auc_base = roc_auc_score(y_drop_base, -y_score_base) if y_true_base.size else float("nan")
    ap_base = average_precision_score(y_drop_base, 1 - 1 / (1 + np.exp(-y_score_base))) if y_true_base.size else float("nan")
    print(f"[gingivitis/{label}] no text     — AUC: {auc_base:.4f} | AP: {ap_base:.4f}")

    # Real text
    term_to_vec_real = shared_utils.load_term_embeddings(device=device)
    run_to_srs = SRA_to_micro
    run_to_terms = shared_utils.parse_run_terms()
    srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)

    subject_time_otu_scores_real = {}
    subject_time_otu_presence_real = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for (subject, time_code), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc=f"[gingivitis/{label}] groups (real text)",
        ):
            sdicts = []
            for srs in srs_list:
                sdict = shared_utils.score_otus_for_srs_with_text(
                    srs,
                    micro_to_otus=micro_to_otus,
                    resolver=resolver,
                    model=model,
                    device=device,
                    emb_group=emb_group,
                    term_to_vec=term_to_vec_real,
                    srs_to_terms=srs_to_terms,
                )
                if sdict:
                    sdicts.append(sdict)
            avg = shared_utils.union_average_logits(sdicts) if sdicts else {}
            subject_time_otu_scores_real[(subject, time_code)] = avg
            subject_time_otu_presence_real[(subject, time_code)] = set(avg.keys())

    y_true_real = []
    y_score_real = []
    for subject, pairs in pairs_by_subject.items():
        for t1, t2 in pairs:
            s1 = subject_time_otu_scores_real.get((subject, t1), {})
            p2 = subject_time_otu_presence_real.get((subject, t2), set())
            for otu, sc in s1.items():
                y_true_real.append(1 if otu in p2 else 0)
                y_score_real.append(sc)

    y_true_real = np.asarray(y_true_real, dtype=np.int64)
    y_drop_real = 1 - y_true_real
    y_score_real = np.asarray(y_score_real, dtype=np.float64)
    auc_real = roc_auc_score(y_drop_real, -y_score_real) if y_true_real.size else float("nan")
    ap_real = average_precision_score(y_drop_real, 1 - 1 / (1 + np.exp(-y_score_real))) if y_true_real.size else float("nan")
    print(f"[gingivitis/{label}] real text   — AUC: {auc_real:.4f} | AP: {ap_real:.4f} | ΔAUC: {auc_real - auc_base:+.4f}")

    # Random text
    dim = shared_utils.TXT_EMB
    term_to_vec_rand = {t: torch.randn(dim, device=device, dtype=torch.float32) for t in term_to_vec_real.keys()}

    subject_time_otu_scores_rand = {}
    subject_time_otu_presence_rand = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for (subject, time_code), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc=f"[gingivitis/{label}] groups (random text)",
        ):
            sdicts = []
            for srs in srs_list:
                sdict = shared_utils.score_otus_for_srs_with_text(
                    srs,
                    micro_to_otus=micro_to_otus,
                    resolver=resolver,
                    model=model,
                    device=device,
                    emb_group=emb_group,
                    term_to_vec=term_to_vec_rand,
                    srs_to_terms=srs_to_terms,
                )
                if sdict:
                    sdicts.append(sdict)
            avg = shared_utils.union_average_logits(sdicts) if sdicts else {}
            subject_time_otu_scores_rand[(subject, time_code)] = avg
            subject_time_otu_presence_rand[(subject, time_code)] = set(avg.keys())

    y_true_rand = []
    y_score_rand = []
    for subject, pairs in pairs_by_subject.items():
        for t1, t2 in pairs:
            s1 = subject_time_otu_scores_rand.get((subject, t1), {})
            p2 = subject_time_otu_presence_rand.get((subject, t2), set())
            for otu, sc in s1.items():
                y_true_rand.append(1 if otu in p2 else 0)
                y_score_rand.append(sc)

    y_true_rand = np.asarray(y_true_rand, dtype=np.int64)
    y_drop_rand = 1 - y_true_rand
    y_score_rand = np.asarray(y_score_rand, dtype=np.float64)
    auc_rand = roc_auc_score(y_drop_rand, -y_score_rand) if y_true_rand.size else float("nan")
    ap_rand = average_precision_score(y_drop_rand, 1 - 1 / (1 + np.exp(-y_score_rand))) if y_true_rand.size else float("nan")
    print(f"[gingivitis/{label}] random text — AUC: {auc_rand:.4f} | AP: {ap_rand:.4f} | ΔAUC: {auc_rand - auc_base:+.4f}")

    return {
        "auc_base": auc_base,
        "ap_base": ap_base,
        "auc_real": auc_real,
        "ap_real": ap_real,
        "auc_rand": auc_rand,
        "ap_rand": ap_rand,
    }


# === SNOWMELT HELPERS ===

def build_snowmelt_grouping():
    snow_csv = "data/snowmelt/snowmelt.csv"
    run_meta, run_to_srs = load_snowmelt_metadata(snow_csv)
    print(f"[snowmelt] runs with metadata: {len(run_meta)} | mapped to SRS: {len(run_to_srs)}")

    bt_time_to_srs = defaultdict(list)
    for run, meta in run_meta.items():
        srs = run_to_srs.get(run)
        if not srs:
            continue
        key = (meta["block"], meta["treatment"], meta["time"])
        bt_time_to_srs[key].append(srs)

    print("[snowmelt] (block,treatment,time) groups:", len(bt_time_to_srs))

    needed_srs = set()
    for srs_list in bt_time_to_srs.values():
        needed_srs.update(srs_list)
    micro_to_otus = shared_utils.collect_micro_to_otus_mapped(needed_srs, shared_utils.MAPPED_PATH)
    print("[snowmelt] SRS with OTUs:", len(micro_to_otus))

    bt_to_times = defaultdict(list)
    for (block, trt, time) in bt_time_to_srs.keys():
        bt_to_times[(block, trt)].append(time)

    return micro_to_otus, bt_time_to_srs, bt_to_times, run_to_srs


def eval_snowmelt_checkpoint(checkpoint_path, micro_to_otus, bt_time_to_srs, bt_to_times, run_to_srs, label):
    print(f"\n=== Snowmelt — checkpoint: {checkpoint_path} ({label}) ===")
    model, device = shared_utils.load_microbiome_model(checkpoint_path)

    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    def time_order_key(t):
        order = {"A": 0, "B": 1, "C": 2}
        return order.get(t, 99)

    # No text
    bt_time_logits = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for key, srs_list in tqdm(
            bt_time_to_srs.items(),
            desc=f"[snowmelt/{label}] groups (no text)",
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
                continue
            bt_time_logits[key] = shared_utils.union_average_logits(sample_dicts)

    y_true_base = []
    y_score_base = []
    for bt, times_list in bt_to_times.items():
        times_sorted = sorted(set(times_list), key=time_order_key)
        if len(times_sorted) < 2:
            continue
        for t1 in times_sorted:
            for t2 in times_sorted:
                if t1 == t2:
                    continue
                key1 = (bt[0], bt[1], t1)
                key2 = (bt[0], bt[1], t2)
                if key1 not in bt_time_logits or key2 not in bt_time_logits:
                    continue
                logits_t1 = bt_time_logits[key1]
                present_t2 = set(bt_time_logits[key2].keys())
                for oid, score in logits_t1.items():
                    y_true_base.append(1 if oid not in present_t2 else 0)  # dropout=1
                    y_score_base.append(score)

    y_true_base = np.asarray(y_true_base, dtype=np.int64)
    y_score_base = np.asarray(y_score_base, dtype=np.float32)
    auc_base = roc_auc_score(y_true_base, -y_score_base) if y_true_base.size else float("nan")
    ap_base = average_precision_score(y_true_base, 1 / (1 + np.exp(-y_score_base))) if y_true_base.size else float("nan")
    print(f"[snowmelt/{label}] no text     — AUC: {auc_base:.4f} | AP: {ap_base:.4f}")

    # Real text
    term_to_vec_real = shared_utils.load_term_embeddings(device=device)
    run_to_terms = shared_utils.parse_run_terms()
    srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)

    bt_time_logits_real = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for key, srs_list in tqdm(
            bt_time_to_srs.items(),
            desc=f"[snowmelt/{label}] groups (real text)",
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
                    term_to_vec=term_to_vec_real,
                    srs_to_terms=srs_to_terms,
                )
                if sdict:
                    sample_dicts.append(sdict)
            if not sample_dicts:
                continue
            bt_time_logits_real[key] = shared_utils.union_average_logits(sample_dicts)

    y_true_real = []
    y_score_real = []
    for bt, times_list in bt_to_times.items():
        times_sorted = sorted(set(times_list), key=time_order_key)
        if len(times_sorted) < 2:
            continue
        for t1 in times_sorted:
            for t2 in times_sorted:
                if t1 == t2:
                    continue
                key1 = (bt[0], bt[1], t1)
                key2 = (bt[0], bt[1], t2)
                if key1 not in bt_time_logits_real or key2 not in bt_time_logits_real:
                    continue
                logits_t1 = bt_time_logits_real[key1]
                present_t2 = set(bt_time_logits_real[key2].keys())
                for oid, score in logits_t1.items():
                    y_true_real.append(1 if oid not in present_t2 else 0)
                    y_score_real.append(score)

    y_true_real = np.asarray(y_true_real, dtype=np.int64)
    y_score_real = np.asarray(y_score_real, dtype=np.float32)
    auc_real = roc_auc_score(y_true_real, -y_score_real) if y_true_real.size else float("nan")
    ap_real = average_precision_score(y_true_real, 1 / (1 + np.exp(-y_score_real))) if y_true_real.size else float("nan")
    print(f"[snowmelt/{label}] real text   — AUC: {auc_real:.4f} | AP: {ap_real:.4f} | ΔAUC: {auc_real - auc_base:+.4f}")

    # Random text
    dim = shared_utils.TXT_EMB
    term_to_vec_rand = {t: torch.randn(dim, device=device, dtype=torch.float32) for t in shared_utils.load_term_embeddings(device=device).keys()}
    bt_time_logits_rand = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for key, srs_list in tqdm(
            bt_time_to_srs.items(),
            desc=f"[snowmelt/{label}] groups (random text)",
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
                    term_to_vec=term_to_vec_rand,
                    srs_to_terms=srs_to_terms,
                )
                if sdict:
                    sample_dicts.append(sdict)
            if not sample_dicts:
                continue
            bt_time_logits_rand[key] = shared_utils.union_average_logits(sample_dicts)

    y_true_rand = []
    y_score_rand = []
    for bt, times_list in bt_to_times.items():
        times_sorted = sorted(set(times_list), key=time_order_key)
        if len(times_sorted) < 2:
            continue
        for t1 in times_sorted:
            for t2 in times_sorted:
                if t1 == t2:
                    continue
                key1 = (bt[0], bt[1], t1)
                key2 = (bt[0], bt[1], t2)
                if key1 not in bt_time_logits_rand or key2 not in bt_time_logits_rand:
                    continue
                logits_t1 = bt_time_logits_rand[key1]
                present_t2 = set(bt_time_logits_rand[key2].keys())
                for oid, score in logits_t1.items():
                    y_true_rand.append(1 if oid not in present_t2 else 0)
                    y_score_rand.append(score)

    y_true_rand = np.asarray(y_true_rand, dtype=np.int64)
    y_score_rand = np.asarray(y_score_rand, dtype=np.float32)
    auc_rand = roc_auc_score(y_true_rand, -y_score_rand) if y_true_rand.size else float("nan")
    ap_rand = average_precision_score(y_true_rand, 1 / (1 + np.exp(-y_score_rand))) if y_true_rand.size else float("nan")
    print(f"[snowmelt/{label}] random text — AUC: {auc_rand:.4f} | AP: {ap_rand:.4f} | ΔAUC: {auc_rand - auc_base:+.4f}")

    return {
        "auc_base": auc_base,
        "ap_base": ap_base,
        "auc_real": auc_real,
        "ap_real": ap_real,
        "auc_rand": auc_rand,
        "ap_rand": ap_rand,
    }


# === DIABIMMUNE HELPERS ===

def build_diabimmune_grouping():
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

    def sort_times(times):
        return sorted([t for t in times if t is not None])

    pairs_by_subject = {}
    total_pairs = 0
    for subj in sorted(subjects):
        times = sort_times([t for (s, t) in subject_time_to_srs.keys() if s == subj])
        if len(times) < 2:
            continue
        pairs = [(t1, t2) for t1 in times for t2 in times if t1 != t2]
        if pairs:
            pairs_by_subject[subj] = pairs
            total_pairs += len(pairs)

    print("[diabimmune] subjects with ≥2 timepoints:", len(pairs_by_subject), "| total pairs:", total_pairs)
    return micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro


def eval_diabimmune_checkpoint(checkpoint_path, micro_to_otus, subject_time_to_srs, pairs_by_subject, SRA_to_micro, label):
    print(f"\n=== DIABIMMUNE — checkpoint: {checkpoint_path} ({label}) ===")
    model, device = shared_utils.load_microbiome_model(checkpoint_path)

    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    # No text
    st_otu_scores = {}
    st_otu_presence = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for (subject, age), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc=f"[diabimmune/{label}] groups (no text)",
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
                st_otu_scores[(subject, age)] = {}
                st_otu_presence[(subject, age)] = set()
                continue
            avg = shared_utils.union_average_logits(sample_dicts)
            st_otu_scores[(subject, age)] = avg
            st_otu_presence[(subject, age)] = set(avg.keys())

    y_drop_base = []
    y_score_base = []
    for subj, pairs in pairs_by_subject.items():
        for t1, t2 in pairs:
            s1 = st_otu_scores.get((subj, t1), {})
            p2 = st_otu_presence.get((subj, t2), set())
            for oid, sc in s1.items():
                y_drop_base.append(1 if oid not in p2 else 0)
                y_score_base.append(sc)

    y_drop_base = np.asarray(y_drop_base, dtype=np.int64)
    y_score_base = np.asarray(y_score_base, dtype=np.float64)
    auc_base = roc_auc_score(y_drop_base, -y_score_base) if y_drop_base.size else float("nan")
    ap_base = average_precision_score(y_drop_base, 1 - 1 / (1 + np.exp(-y_score_base))) if y_drop_base.size else float("nan")
    print(f"[diabimmune/{label}] no text     — AUC: {auc_base:.4f} | AP: {ap_base:.4f}")

    # Real text
    term_to_vec_real = shared_utils.load_term_embeddings(device=device)
    run_to_terms = shared_utils.parse_run_terms()
    srs_to_terms = shared_utils.build_srs_terms(SRA_to_micro, run_to_terms, shared_utils.MAPPED_PATH)

    st_otu_scores_real = {}
    st_otu_presence_real = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for (subject, age), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc=f"[diabimmune/{label}] groups (real text)",
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
                    term_to_vec=term_to_vec_real,
                    srs_to_terms=srs_to_terms,
                )
                if sdict:
                    sample_dicts.append(sdict)
            if not sample_dicts:
                st_otu_scores_real[(subject, age)] = {}
                st_otu_presence_real[(subject, age)] = set()
                continue
            avg = shared_utils.union_average_logits(sample_dicts)
            st_otu_scores_real[(subject, age)] = avg
            st_otu_presence_real[(subject, age)] = set(avg.keys())

    y_drop_real = []
    y_score_real = []
    for subj, pairs in pairs_by_subject.items():
        for t1, t2 in pairs:
            s1 = st_otu_scores_real.get((subj, t1), {})
            p2 = st_otu_presence_real.get((subj, t2), set())
            for oid, sc in s1.items():
                y_drop_real.append(1 if oid not in p2 else 0)
                y_score_real.append(sc)

    y_drop_real = np.asarray(y_drop_real, dtype=np.int64)
    y_score_real = np.asarray(y_score_real, dtype=np.float64)
    auc_real = roc_auc_score(y_drop_real, -y_score_real) if y_drop_real.size else float("nan")
    ap_real = average_precision_score(y_drop_real, 1 - 1 / (1 + np.exp(-y_score_real))) if y_drop_real.size else float("nan")
    print(f"[diabimmune/{label}] real text   — AUC: {auc_real:.4f} | AP: {ap_real:.4f} | ΔAUC: {auc_real - auc_base:+.4f}")

    # Random text
    dim = shared_utils.TXT_EMB
    term_to_vec_rand = {t: torch.randn(dim, device=device, dtype=torch.float32) for t in shared_utils.load_term_embeddings(device=device).keys()}
    st_otu_scores_rand = {}
    st_otu_presence_rand = {}
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for (subject, age), srs_list in tqdm(
            list(subject_time_to_srs.items()),
            desc=f"[diabimmune/{label}] groups (random text)",
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
                    term_to_vec=term_to_vec_rand,
                    srs_to_terms=srs_to_terms,
                )
                if sdict:
                    sample_dicts.append(sdict)
            if not sample_dicts:
                st_otu_scores_rand[(subject, age)] = {}
                st_otu_presence_rand[(subject, age)] = set()
                continue
            avg = shared_utils.union_average_logits(sample_dicts)
            st_otu_scores_rand[(subject, age)] = avg
            st_otu_presence_rand[(subject, age)] = set(avg.keys())

    y_drop_rand = []
    y_score_rand = []
    for subj, pairs in pairs_by_subject.items():
        for t1, t2 in pairs:
            s1 = st_otu_scores_rand.get((subj, t1), {})
            p2 = st_otu_presence_rand.get((subj, t2), set())
            for oid, sc in s1.items():
                y_drop_rand.append(1 if oid not in p2 else 0)
                y_score_rand.append(sc)

    y_drop_rand = np.asarray(y_drop_rand, dtype=np.int64)
    y_score_rand = np.asarray(y_score_rand, dtype=np.float64)
    auc_rand = roc_auc_score(y_drop_rand, -y_score_rand) if y_drop_rand.size else float("nan")
    ap_rand = average_precision_score(y_drop_rand, 1 - 1 / (1 + np.exp(-y_score_rand))) if y_drop_rand.size else float("nan")
    print(f"[diabimmune/{label}] random text — AUC: {auc_rand:.4f} | AP: {ap_rand:.4f} | ΔAUC: {auc_rand - auc_base:+.4f}")

    return {
        "auc_base": auc_base,
        "ap_base": ap_base,
        "auc_real": auc_real,
        "ap_real": ap_real,
        "auc_rand": auc_rand,
        "ap_rand": ap_rand,
    }


def main():
    # Gingivitis
    ging_micro, ging_grouping, ging_pairs, ging_SRA_to_micro = build_gingivitis_grouping()
    ging_no_text = eval_gingivitis_checkpoint(
        CKPT_NO_TEXT, ging_micro, ging_grouping, ging_pairs, ging_SRA_to_micro, label="no-text checkpoint"
    )
    ging_with_text = eval_gingivitis_checkpoint(
        CKPT_WITH_TEXT, ging_micro, ging_grouping, ging_pairs, ging_SRA_to_micro, label="text-trained checkpoint"
    )

    # Snowmelt
    snow_micro, snow_bt_time_to_srs, snow_bt_to_times, snow_run_to_srs = build_snowmelt_grouping()
    snow_no_text = eval_snowmelt_checkpoint(
        CKPT_NO_TEXT, snow_micro, snow_bt_time_to_srs, snow_bt_to_times, snow_run_to_srs, label="no-text checkpoint"
    )
    snow_with_text = eval_snowmelt_checkpoint(
        CKPT_WITH_TEXT, snow_micro, snow_bt_time_to_srs, snow_bt_to_times, snow_run_to_srs, label="text-trained checkpoint"
    )

    # DIABIMMUNE
    diab_micro, diab_grouping, diab_pairs, diab_SRA_to_micro = build_diabimmune_grouping()
    diab_no_text = eval_diabimmune_checkpoint(
        CKPT_NO_TEXT, diab_micro, diab_grouping, diab_pairs, diab_SRA_to_micro, label="no-text checkpoint"
    )
    diab_with_text = eval_diabimmune_checkpoint(
        CKPT_WITH_TEXT, diab_micro, diab_grouping, diab_pairs, diab_SRA_to_micro, label="text-trained checkpoint"
    )

    print("\n=== Summary (AUC) ===")
    print("Dataset\tCheckpoint\tCondition\tAUC")
    print(f"Gingivitis\tno-text\tno-text\t{ging_no_text['auc_base']:.4f}")
    print(f"Gingivitis\tno-text\treal-text\t{ging_no_text['auc_real']:.4f}")
    print(f"Gingivitis\tno-text\trandom-text\t{ging_no_text['auc_rand']:.4f}")
    print(f"Gingivitis\ttext-trained\tno-text\t{ging_with_text['auc_base']:.4f}")
    print(f"Gingivitis\ttext-trained\treal-text\t{ging_with_text['auc_real']:.4f}")
    print(f"Gingivitis\ttext-trained\trandom-text\t{ging_with_text['auc_rand']:.4f}")

    print(f"Snowmelt\tno-text\tno-text\t{snow_no_text['auc_base']:.4f}")
    print(f"Snowmelt\tno-text\treal-text\t{snow_no_text['auc_real']:.4f}")
    print(f"Snowmelt\tno-text\trandom-text\t{snow_no_text['auc_rand']:.4f}")
    print(f"Snowmelt\ttext-trained\tno-text\t{snow_with_text['auc_base']:.4f}")
    print(f"Snowmelt\ttext-trained\treal-text\t{snow_with_text['auc_real']:.4f}")
    print(f"Snowmelt\ttext-trained\trandom-text\t{snow_with_text['auc_rand']:.4f}")

    print(f"DIABIMMUNE\tno-text\tno-text\t{diab_no_text['auc_base']:.4f}")
    print(f"DIABIMMUNE\tno-text\treal-text\t{diab_no_text['auc_real']:.4f}")
    print(f"DIABIMMUNE\tno-text\trandom-text\t{diab_no_text['auc_rand']:.4f}")
    print(f"DIABIMMUNE\ttext-trained\tno-text\t{diab_with_text['auc_base']:.4f}")
    print(f"DIABIMMUNE\ttext-trained\treal-text\t{diab_with_text['auc_real']:.4f}")
    print(f"DIABIMMUNE\ttext-trained\trandom-text\t{diab_with_text['auc_rand']:.4f}")


if __name__ == "__main__":
    main()

