#!/usr/bin/env python3
"""
Compare text effects on colonisation prediction for three datasets:
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
  - Random text (same term IDs, random vectors)

We report ROC AUC (and AP) so you can see how text and random text affect
each model on each colonisation task.
"""

import os
import sys
import csv
import random
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# === GINGIVITIS COLONISATION ===

def build_gingivitis_colonisation_examples(n_colonizer_samples=10000, n_non_colonizer_samples=10000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    gingivitis_csv = "data/gingivitis/gingiva.csv"
    run_ids, SRA_to_micro = load_gingivitis_run_data(
        gingivitis_path=gingivitis_csv,
    )
    print(f"[gingivitis] runs: {len(run_ids)} | mapped to SRS: {len(SRA_to_micro)}")

    micro_to_otus = collect_ging_micro_to_otus(SRA_to_micro)
    print("[gingivitis] SRS with OTUs:", len(micro_to_otus))

    # Group by patient/timepoint as in colonisation_test.py
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

    patient_timepoint_samples = defaultdict(lambda: defaultdict(list))
    for r in records:
        patient_timepoint_samples[r["subject"]][r["time"]].append(r["srs"])

    multi_timepoint_patients = {p: tp for p, tp in patient_timepoint_samples.items() if len(tp) > 1}
    print(f"[gingivitis] patients with multiple timepoints: {len(multi_timepoint_patients)}")

    # Collect colonizer examples
    print("[gingivitis] Finding colonizer examples...")
    all_colonizer_examples = []
    for patient, timepoints in tqdm(multi_timepoint_patients.items(), desc="[gingivitis] patients"):
        timepoint_list = list(timepoints.keys())
        for i in range(len(timepoint_list)):
            for j in range(len(timepoint_list)):
                if i == j:
                    continue
                t1 = timepoint_list[i]
                t2 = timepoint_list[j]
                srs_t1 = timepoints[t1]
                srs_t2 = timepoints[t2]
                otus_t1 = set()
                for srs in srs_t1:
                    otus_t1.update(micro_to_otus.get(srs, []))
                otus_t2 = set()
                for srs in srs_t2:
                    otus_t2.update(micro_to_otus.get(srs, []))
                colonizers = otus_t2 - otus_t1
                for target in colonizers:
                    all_colonizer_examples.append(
                        {
                            "patient": patient,
                            "t1": t1,
                            "t2": t2,
                            "srs_t1": list(srs_t1),
                            "target_otu": target,
                            "is_colonizer": True,
                        }
                    )

    print(f"[gingivitis] total colonizer examples: {len(all_colonizer_examples)}")

    if len(all_colonizer_examples) > n_colonizer_samples:
        sampled_colonizers = random.sample(all_colonizer_examples, n_colonizer_samples)
    else:
        sampled_colonizers = all_colonizer_examples
        print(f"[gingivitis] using all {len(sampled_colonizers)} colonizer examples")

    # Non-colonizers
    print("[gingivitis] Preparing non-colonizer examples...")
    all_possible_otus = set()
    for srs, otus in micro_to_otus.items():
        all_possible_otus.update(otus)

    sampled_non_colonizers = []
    for ex in tqdm(sampled_colonizers, desc="[gingivitis] sampling non-colonizers"):
        if len(sampled_non_colonizers) >= n_non_colonizer_samples:
            break
        patient = ex["patient"]
        t1 = ex["t1"]
        t2 = ex["t2"]
        srs_t1 = patient_timepoint_samples[patient][t1]
        srs_t2 = patient_timepoint_samples[patient][t2]
        otus_t1 = set()
        for srs in srs_t1:
            otus_t1.update(micro_to_otus.get(srs, []))
        otus_t2 = set()
        for srs in srs_t2:
            otus_t2.update(micro_to_otus.get(srs, []))
        absent_both = list(all_possible_otus - otus_t1 - otus_t2)
        if not absent_both:
            continue
        target = random.choice(absent_both)
        sampled_non_colonizers.append(
            {
                "patient": patient,
                "t1": t1,
                "t2": t2,
                "srs_t1": list(srs_t1),
                "target_otu": target,
                "is_colonizer": False,
            }
        )

    examples = sampled_colonizers + sampled_non_colonizers
    print(f"[gingivitis] total examples: {len(examples)}")
    return micro_to_otus, SRA_to_micro, examples


def eval_gingivitis_colonisation_checkpoint(checkpoint_path, micro_to_otus, SRA_to_micro, examples, label):
    print(f"\n=== Gingivitis colonisation — checkpoint: {checkpoint_path} ({label}) ===")
    model, device = shared_utils.load_microbiome_model(checkpoint_path)

    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    # Shared term mappings
    term_to_vec_real = shared_utils.load_term_embeddings(device=device)
    run_to_terms = shared_utils.parse_run_terms()
    run_to_srs = SRA_to_micro
    srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)

    # Random term embeddings
    dim = shared_utils.TXT_EMB
    term_to_vec_rand = {t: torch.randn(dim, device=device, dtype=torch.float32) for t in term_to_vec_real.keys()}

    results = {}
    for mode, term_to_vec in [
        ("no-text", None),
        ("real-text", term_to_vec_real),
        ("random-text", term_to_vec_rand),
    ]:
        target_scores = []
        target_labels = []

        with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
            emb_group = emb_file["embeddings"]
            for ex in tqdm(examples, desc=f"[gingivitis/{label}] scoring ({mode})"):
                srs_t1 = ex["srs_t1"]
                target = ex["target_otu"]
                per_sample_scores = []
                for srs in srs_t1:
                    base_otus = micro_to_otus.get(srs, [])
                    aug_otus = list(base_otus)
                    if target not in aug_otus:
                        aug_otus.append(target)
                    if mode == "no-text":
                        logits_map = shared_utils.score_otu_list(
                            aug_otus,
                            resolver=resolver,
                            model=model,
                            device=device,
                            emb_group=emb_group,
                        )
                    else:
                        logits_map = shared_utils.score_otu_list_with_text(
                            srs,
                            aug_otus,
                            resolver=resolver,
                            model=model,
                            device=device,
                            emb_group=emb_group,
                            term_to_vec=term_to_vec,
                            srs_to_terms=srs_to_terms,
                        )
                    if target in logits_map:
                        per_sample_scores.append(logits_map[target])
                if per_sample_scores:
                    target_scores.append(float(np.mean(per_sample_scores)))
                    target_labels.append(1 if ex["is_colonizer"] else 0)

        if not target_scores:
            print(f"[gingivitis/{label}] {mode}: no scored examples")
            results[mode] = (float("nan"), float("nan"))
            continue

        y = np.array(target_labels, dtype=np.int64)
        logits = np.array(target_scores, dtype=np.float32)
        probs = sigmoid(logits)

        if y.size and (y.min() != y.max()):
            auc = roc_auc_score(y, probs)
            ap = average_precision_score(y, probs)
        else:
            auc = float("nan")
            ap = float("nan")
        print(f"[gingivitis/{label}] {mode} — AUC: {auc:.4f} | AP: {ap:.4f}")
        results[mode] = (auc, ap)

    return results


# === SNOWMELT COLONISATION ===

def build_snowmelt_colonisation_examples(n_colonizer_samples=50000, n_non_colonizer_samples=50000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

    def time_order_key(t):
        order = {"A": 0, "B": 1, "C": 2}
        return order.get(t, 99)

    bt_to_times = defaultdict(set)
    for (block, trt, time) in bt_time_to_srs.keys():
        bt_to_times[(block, trt)].add(time)

    # Colonizers
    print("[snowmelt] Finding colonizer examples...")
    all_colonizer_examples = []
    for (block, trt), times_set in tqdm(bt_to_times.items(), desc="[snowmelt] (block,treatment)"):
        times_sorted = sorted(times_set, key=time_order_key)
        if len(times_sorted) < 2:
            continue
        for i in range(len(times_sorted)):
            for j in range(len(times_sorted)):
                if i == j:
                    continue
                t1 = times_sorted[i]
                t2 = times_sorted[j]
                srs_t1 = bt_time_to_srs.get((block, trt, t1), [])
                srs_t2 = bt_time_to_srs.get((block, trt, t2), [])
                otus_t1 = set()
                for srs in srs_t1:
                    otus_t1.update(micro_to_otus.get(srs, []))
                otus_t2 = set()
                for srs in srs_t2:
                    otus_t2.update(micro_to_otus.get(srs, []))
                colonizers = otus_t2 - otus_t1
                for target in colonizers:
                    all_colonizer_examples.append(
                        {
                            "block": block,
                            "treatment": trt,
                            "t1": t1,
                            "t2": t2,
                            "srs_t1": list(srs_t1),
                            "target_otu": target,
                            "is_colonizer": True,
                        }
                    )

    print(f"[snowmelt] total colonizer examples: {len(all_colonizer_examples)}")

    if len(all_colonizer_examples) > n_colonizer_samples:
        sampled_colonizers = random.sample(all_colonizer_examples, n_colonizer_samples)
    else:
        sampled_colonizers = all_colonizer_examples
        print(f"[snowmelt] using all {len(sampled_colonizers)} colonizer examples")

    # Non-colonizers
    print("[snowmelt] Preparing non-colonizer examples...")
    all_possible_otus = set()
    for srs, otus in micro_to_otus.items():
        all_possible_otus.update(otus)

    sampled_non_colonizers = []
    for ex in tqdm(sampled_colonizers, desc="[snowmelt] sampling non-colonizers"):
        if len(sampled_non_colonizers) >= n_non_colonizer_samples:
            break
        block, trt, t1, t2 = ex["block"], ex["treatment"], ex["t1"], ex["t2"]
        srs_t1 = bt_time_to_srs.get((block, trt, t1), [])
        srs_t2 = bt_time_to_srs.get((block, trt, t2), [])
        otus_t1 = set()
        for srs in srs_t1:
            otus_t1.update(micro_to_otus.get(srs, []))
        otus_t2 = set()
        for srs in srs_t2:
            otus_t2.update(micro_to_otus.get(srs, []))
        absent_both = list(all_possible_otus - otus_t1 - otus_t2)
        if not absent_both:
            continue
        target = random.choice(absent_both)
        sampled_non_colonizers.append(
            {
                "block": block,
                "treatment": trt,
                "t1": t1,
                "t2": t2,
                "srs_t1": list(srs_t1),
                "target_otu": target,
                "is_colonizer": False,
            }
        )

    examples = sampled_colonizers + sampled_non_colonizers
    print(f"[snowmelt] total examples: {len(examples)}")
    return micro_to_otus, run_to_srs, examples


def eval_snowmelt_colonisation_checkpoint(checkpoint_path, micro_to_otus, run_to_srs, examples, label):
    print(f"\n=== Snowmelt colonisation — checkpoint: {checkpoint_path} ({label}) ===")
    model, device = shared_utils.load_microbiome_model(checkpoint_path)

    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    term_to_vec_real = shared_utils.load_term_embeddings(device=device)
    run_to_terms = shared_utils.parse_run_terms()
    srs_to_terms = shared_utils.build_srs_terms(run_to_srs, run_to_terms, shared_utils.MAPPED_PATH)

    dim = shared_utils.TXT_EMB
    term_to_vec_rand = {t: torch.randn(dim, device=device, dtype=torch.float32) for t in term_to_vec_real.keys()}

    results = {}
    for mode, term_to_vec in [
        ("no-text", None),
        ("real-text", term_to_vec_real),
        ("random-text", term_to_vec_rand),
    ]:
        target_scores = []
        target_labels = []

        with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
            emb_group = emb_file["embeddings"]
            for ex in tqdm(examples, desc=f"[snowmelt/{label}] scoring ({mode})"):
                srs_t1 = ex["srs_t1"]
                target = ex["target_otu"]
                per_sample_scores = []
                for srs in srs_t1:
                    base_otus = micro_to_otus.get(srs, [])
                    aug_otus = list(base_otus)
                    if target not in aug_otus:
                        aug_otus.append(target)
                    if mode == "no-text":
                        logits_map = shared_utils.score_otu_list(
                            aug_otus,
                            resolver=resolver,
                            model=model,
                            device=device,
                            emb_group=emb_group,
                        )
                    else:
                        logits_map = shared_utils.score_otu_list_with_text(
                            srs,
                            aug_otus,
                            resolver=resolver,
                            model=model,
                            device=device,
                            emb_group=emb_group,
                            term_to_vec=term_to_vec,
                            srs_to_terms=srs_to_terms,
                        )
                    if target in logits_map:
                        per_sample_scores.append(logits_map[target])
                if per_sample_scores:
                    target_scores.append(float(np.mean(per_sample_scores)))
                    target_labels.append(1 if ex["is_colonizer"] else 0)

        if not target_scores:
            print(f"[snowmelt/{label}] {mode}: no scored examples")
            results[mode] = (float("nan"), float("nan"))
            continue

        y = np.array(target_labels, dtype=np.int64)
        logits = np.array(target_scores, dtype=np.float32)
        probs = sigmoid(logits)

        if y.size and (y.min() != y.max()):
            auc = roc_auc_score(y, probs)
            ap = average_precision_score(y, probs)
        else:
            auc = float("nan")
            ap = float("nan")
        print(f"[snowmelt/{label}] {mode} — AUC: {auc:.4f} | AP: {ap:.4f}")
        results[mode] = (auc, ap)

    return results


# === DIABIMMUNE COLONISATION ===

def build_diabimmune_colonisation_examples(n_colonizer_samples=20000, n_non_colonizer_samples=20000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

    # Colonizers
    print("[diabimmune] Finding colonizer examples...")
    all_colonizer_examples = []
    for subject, timepoints in tqdm(
        defaultdict(dict, {subj: {age: subject_time_to_srs[(subj, age)] for (s, age) in subject_time_to_srs if s == subj} for subj in subjects}).items(),
        desc="[diabimmune] subjects",
    ):
        ages = list(timepoints.keys())
        if len(ages) < 2:
            continue
        for i in range(len(ages)):
            for j in range(len(ages)):
                if i == j:
                    continue
                t1 = ages[i]
                t2 = ages[j]
                srs_t1 = timepoints[t1]
                srs_t2 = timepoints[t2]
                otus_t1 = set()
                for srs in srs_t1:
                    otus_t1.update(micro_to_otus.get(srs, []))
                otus_t2 = set()
                for srs in srs_t2:
                    otus_t2.update(micro_to_otus.get(srs, []))
                colonizers = otus_t2 - otus_t1
                for target in colonizers:
                    all_colonizer_examples.append(
                        {
                            "subject": subject,
                            "t1": t1,
                            "t2": t2,
                            "srs_t1": list(srs_t1),
                            "target_otu": target,
                            "is_colonizer": True,
                        }
                    )

    print(f"[diabimmune] total colonizer examples: {len(all_colonizer_examples)}")

    if len(all_colonizer_examples) > n_colonizer_samples:
        sampled_colonizers = random.sample(all_colonizer_examples, n_colonizer_samples)
    else:
        sampled_colonizers = all_colonizer_examples
        print(f"[diabimmune] using all {len(sampled_colonizers)} colonizer examples")

    # Non-colonizers
    print("[diabimmune] Preparing non-colonizer examples...")
    all_possible_otus = set()
    for srs, otus in micro_to_otus.items():
        all_possible_otus.update(otus)

    sampled_non_colonizers = []
    # rebuild subject→timepoints mapping in a simpler form
    subject_to_timepoints = defaultdict(dict)
    for (subj, age), srs_list in subject_time_to_srs.items():
        subject_to_timepoints[subj][age] = srs_list

    for ex in tqdm(sampled_colonizers, desc="[diabimmune] sampling non-colonizers"):
        if len(sampled_non_colonizers) >= n_non_colonizer_samples:
            break
        subject = ex["subject"]
        t1 = ex["t1"]
        t2 = ex["t2"]
        srs_t1 = subject_to_timepoints[subject][t1]
        srs_t2 = subject_to_timepoints[subject][t2]
        otus_t1 = set()
        for srs in srs_t1:
            otus_t1.update(micro_to_otus.get(srs, []))
        otus_t2 = set()
        for srs in srs_t2:
            otus_t2.update(micro_to_otus.get(srs, []))
        absent_both = list(all_possible_otus - otus_t1 - otus_t2)
        if not absent_both:
            continue
        target = random.choice(absent_both)
        sampled_non_colonizers.append(
            {
                "subject": subject,
                "t1": t1,
                "t2": t2,
                "srs_t1": list(srs_t1),
                "target_otu": target,
                "is_colonizer": False,
            }
        )

    examples = sampled_colonizers + sampled_non_colonizers
    print(f"[diabimmune] total examples: {len(examples)}")
    return micro_to_otus, SRA_to_micro, examples


def eval_diabimmune_colonisation_checkpoint(checkpoint_path, micro_to_otus, SRA_to_micro, examples, label):
    print(f"\n=== DIABIMMUNE colonisation — checkpoint: {checkpoint_path} ({label}) ===")
    model, device = shared_utils.load_microbiome_model(checkpoint_path)

    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH) if os.path.exists(shared_utils.RENAME_MAP_PATH) else None
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B") if rename_map else {}

    term_to_vec_real = shared_utils.load_term_embeddings(device=device)
    run_to_terms = shared_utils.parse_run_terms()
    srs_to_terms = shared_utils.build_srs_terms(SRA_to_micro, run_to_terms, shared_utils.MAPPED_PATH)

    dim = shared_utils.TXT_EMB
    term_to_vec_rand = {t: torch.randn(dim, device=device, dtype=torch.float32) for t in term_to_vec_real.keys()}

    results = {}
    for mode, term_to_vec in [
        ("no-text", None),
        ("real-text", term_to_vec_real),
        ("random-text", term_to_vec_rand),
    ]:
        target_scores = []
        target_labels = []

        with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
            emb_group = emb_file["embeddings"]
            for ex in tqdm(examples, desc=f"[diabimmune/{label}] scoring ({mode})"):
                srs_t1 = ex["srs_t1"]
                target = ex["target_otu"]
                per_sample_scores = []
                for srs in srs_t1:
                    base_otus = micro_to_otus.get(srs, [])
                    aug_otus = list(base_otus)
                    if target not in aug_otus:
                        aug_otus.append(target)
                    if mode == "no-text":
                        logits_map = shared_utils.score_otu_list(
                            aug_otus,
                            resolver=resolver,
                            model=model,
                            device=device,
                            emb_group=emb_group,
                        )
                    else:
                        logits_map = shared_utils.score_otu_list_with_text(
                            srs,
                            aug_otus,
                            resolver=resolver,
                            model=model,
                            device=device,
                            emb_group=emb_group,
                            term_to_vec=term_to_vec,
                            srs_to_terms=srs_to_terms,
                        )
                    if target in logits_map:
                        per_sample_scores.append(logits_map[target])
                if per_sample_scores:
                    target_scores.append(float(np.mean(per_sample_scores)))
                    target_labels.append(1 if ex["is_colonizer"] else 0)

        if not target_scores:
            print(f"[diabimmune/{label}] {mode}: no scored examples")
            results[mode] = (float("nan"), float("nan"))
            continue

        y = np.array(target_labels, dtype=np.int64)
        logits = np.array(target_scores, dtype=np.float32)
        probs = sigmoid(logits)

        if y.size and (y.min() != y.max()):
            auc = roc_auc_score(y, probs)
            ap = average_precision_score(y, probs)
        else:
            auc = float("nan")
            ap = float("nan")
        print(f"[diabimmune/{label}] {mode} — AUC: {auc:.4f} | AP: {ap:.4f}")
        results[mode] = (auc, ap)

    return results


def main():
    # Gingivitis
    ging_micro, ging_SRA_to_micro, ging_examples = build_gingivitis_colonisation_examples()
    ging_no_text = eval_gingivitis_colonisation_checkpoint(
        CKPT_NO_TEXT, ging_micro, ging_SRA_to_micro, ging_examples, label="no-text checkpoint"
    )
    ging_with_text = eval_gingivitis_colonisation_checkpoint(
        CKPT_WITH_TEXT, ging_micro, ging_SRA_to_micro, ging_examples, label="text-trained checkpoint"
    )

    # Snowmelt
    snow_micro, snow_run_to_srs, snow_examples = build_snowmelt_colonisation_examples()
    snow_no_text = eval_snowmelt_colonisation_checkpoint(
        CKPT_NO_TEXT, snow_micro, snow_run_to_srs, snow_examples, label="no-text checkpoint"
    )
    snow_with_text = eval_snowmelt_colonisation_checkpoint(
        CKPT_WITH_TEXT, snow_micro, snow_run_to_srs, snow_examples, label="text-trained checkpoint"
    )

    # DIABIMMUNE
    diab_micro, diab_SRA_to_micro, diab_examples = build_diabimmune_colonisation_examples()
    diab_no_text = eval_diabimmune_colonisation_checkpoint(
        CKPT_NO_TEXT, diab_micro, diab_SRA_to_micro, diab_examples, label="no-text checkpoint"
    )
    diab_with_text = eval_diabimmune_colonisation_checkpoint(
        CKPT_WITH_TEXT, diab_micro, diab_SRA_to_micro, diab_examples, label="text-trained checkpoint"
    )

    print("\n=== Summary (Colonisation AUC) ===")
    print("Dataset\tCheckpoint\tCondition\tAUC")
    print(f"Gingivitis\tno-text\tno-text\t{ging_no_text['no-text'][0]:.4f}")
    print(f"Gingivitis\tno-text\treal-text\t{ging_no_text['real-text'][0]:.4f}")
    print(f"Gingivitis\tno-text\trandom-text\t{ging_no_text['random-text'][0]:.4f}")
    print(f"Gingivitis\ttext-trained\tno-text\t{ging_with_text['no-text'][0]:.4f}")
    print(f"Gingivitis\ttext-trained\treal-text\t{ging_with_text['real-text'][0]:.4f}")
    print(f"Gingivitis\ttext-trained\trandom-text\t{ging_with_text['random-text'][0]:.4f}")

    print(f"Snowmelt\tno-text\tno-text\t{snow_no_text['no-text'][0]:.4f}")
    print(f"Snowmelt\tno-text\treal-text\t{snow_no_text['real-text'][0]:.4f}")
    print(f"Snowmelt\tno-text\trandom-text\t{snow_no_text['random-text'][0]:.4f}")
    print(f"Snowmelt\ttext-trained\tno-text\t{snow_with_text['no-text'][0]:.4f}")
    print(f"Snowmelt\ttext-trained\treal-text\t{snow_with_text['real-text'][0]:.4f}")
    print(f"Snowmelt\ttext-trained\trandom-text\t{snow_with_text['random-text'][0]:.4f}")

    print(f"DIABIMMUNE\tno-text\tno-text\t{diab_no_text['no-text'][0]:.4f}")
    print(f"DIABIMMUNE\tno-text\treal-text\t{diab_no_text['real-text'][0]:.4f}")
    print(f"DIABIMMUNE\tno-text\trandom-text\t{diab_no_text['random-text'][0]:.4f}")
    print(f"DIABIMMUNE\ttext-trained\tno-text\t{diab_with_text['no-text'][0]:.4f}")
    print(f"DIABIMMUNE\ttext-trained\treal-text\t{diab_with_text['real-text'][0]:.4f}")
    print(f"DIABIMMUNE\ttext-trained\trandom-text\t{diab_with_text['random-text'][0]:.4f}")


if __name__ == "__main__":
    main()

