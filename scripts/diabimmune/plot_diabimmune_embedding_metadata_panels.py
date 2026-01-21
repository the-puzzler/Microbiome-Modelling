#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402


REAL_NPZ = os.path.join("data", "diabimmune", "diabimmune_real_embeddings_cache.npz")
SAMPLES_CSV = os.path.join("data", "diabimmune", "samples.csv")
PREGNANCY_BIRTH_CSV = os.path.join("data", "diabimmune", "pregnancy_birth.csv")
OUT_PNG = os.path.join("data", "diabimmune", "diabimmune_embedding_metadata_panels.png")
STABILITY_NPZ = os.path.join("data", "diabimmune", "diabimmune_stability_scores_cache.npz")


def parse_args():
    p = argparse.ArgumentParser(description="DIABIMMUNE: PCA of real embeddings coloured by metadata.")
    p.add_argument("--real-npz", default=REAL_NPZ)
    p.add_argument("--samples-csv", default=SAMPLES_CSV)
    p.add_argument("--pregnancy-birth-csv", default=PREGNANCY_BIRTH_CSV)
    p.add_argument("--stability-cache", default=STABILITY_NPZ)
    p.add_argument("--out", default=OUT_PNG)
    p.add_argument("--max-samples", type=int, default=0, help="Optional cap for faster debugging (0 = all).")
    return p.parse_args()


def pca2_fit(X):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, vt[:2]


def pca2_transform(X, mu, comps):
    return (np.asarray(X, dtype=float) - mu) @ comps.T


def age_bin_label(age):
    a = float(age)
    if a < 216.2:
        return "27-216.2"
    if a < 405.3:
        return "216.2-405.3"
    if a < 594.5:
        return "405.3-594.5"
    if a < 783.7:
        return "594.5-783.7"
    if a < 972.8:
        return "783.7-972.8"
    return "972.8-"


def load_csv_table(path, key_field):
    table = {}
    with open(path) as f:
        r = csv.DictReader(f)
        r.fieldnames[0] = r.fieldnames[0].lstrip("\ufeff")
        for row in r:
            key = row.get(key_field, "").strip()
            if key:
                table[key] = row
    return table


def load_samples_rows(samples_csv):
    rows = []
    with open(samples_csv) as f:
        r = csv.DictReader(f)
        r.fieldnames[0] = r.fieldnames[0].lstrip("\ufeff")
        for row in r:
            rows.append(row)
    return rows


def build_subject_age_to_row(samples_rows):
    out = {}
    for row in samples_rows:
        subj = str(row.get("subjectID", "")).strip()
        age = row.get("age_at_collection", "")
        if not subj or age == "":
            continue
        age_key = str(int(round(float(age))))
        out[(subj, age_key)] = row
    return out


def build_subject_age_to_otus(samples_csv):
    samples_rows = load_samples_rows(samples_csv)
    subj_age_to_row = build_subject_age_to_row(samples_rows)
    sample_to_row = {str(r.get("sampleID", "")).strip(): r for r in samples_rows if str(r.get("sampleID", "")).strip()}

    run_rows, sra_to_micro, _gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
    micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)

    subject_age_to_otus = {}
    for srs, info in micro_to_sample.items():
        subj = str(info.get("subject", "")).strip()
        sample_id = str(info.get("sample", "")).strip()
        if not subj or not sample_id:
            continue
        row = sample_to_row.get(sample_id)
        if row is None:
            continue
        age = row.get("age_at_collection", "")
        if age == "":
            continue
        age_key = str(int(round(float(age))))
        otus = micro_to_otus.get(srs, [])
        if not otus:
            continue
        subject_age_to_otus.setdefault((subj, age_key), set()).update(otus)

    subject_age_to_otus = {k: sorted(v) for k, v in subject_age_to_otus.items()}
    return subject_age_to_otus, subj_age_to_row, micro_to_otus


def compute_stability_scores(keys, subject_age_to_otus, micro_to_otus, cache_path):
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        cached_keys = [tuple(map(str, k)) for k in cache["keys"]]
        if cached_keys == keys:
            return cache["stability"].astype(float)

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    stability = np.full((len(keys),), np.nan, dtype=float)
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for i, (subj, age_key) in enumerate(tqdm(keys, desc="Stability (mean logit)", unit="sample")):
            otus = subject_age_to_otus.get((subj, age_key), [])
            logits = shared_utils.score_otu_list(otus, resolver=resolver, model=model, device=device, emb_group=emb_group)
            if logits:
                stability[i] = float(np.mean(list(logits.values())))

    np.savez(cache_path, keys=np.asarray(keys, dtype=object), stability=stability)
    return stability


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.stability_cache), exist_ok=True)

    real_cache = np.load(args.real_npz, allow_pickle=True)
    keys_raw = real_cache["keys"]
    emb = real_cache["emb"].astype(float)
    keys = [(str(keys_raw[i][0]), str(keys_raw[i][1])) for i in range(len(keys_raw))]

    if args.max_samples and len(keys) > args.max_samples:
        keys = keys[: args.max_samples]
        emb = emb[: args.max_samples]

    mu, comps = pca2_fit(emb)
    xy = pca2_transform(emb, mu, comps)

    pregnancy = load_csv_table(args.pregnancy_birth_csv, key_field="subjectID")
    subject_age_to_otus, subj_age_to_row, micro_to_otus = build_subject_age_to_otus(args.samples_csv)

    ages = np.asarray([float(t) for (_s, t) in keys], dtype=float)
    age_bins = np.asarray([age_bin_label(a) for a in ages.tolist()], dtype=str)
    countries = np.asarray([subj_age_to_row.get((s, t), {}).get("country", "").strip() for (s, t) in keys], dtype=str)
    cohorts = np.asarray([subj_age_to_row.get((s, t), {}).get("cohort", "").strip() for (s, t) in keys], dtype=str)

    genders = np.asarray([pregnancy.get(s, {}).get("gender", "").strip() for (s, _t) in keys], dtype=str)
    locations = np.asarray([pregnancy.get(s, {}).get("location", "").strip() for (s, _t) in keys], dtype=str)
    csections = np.asarray([pregnancy.get(s, {}).get("csection", "").strip() for (s, _t) in keys], dtype=str)
    abx_preg = np.asarray([pregnancy.get(s, {}).get("abx_while_pregnant", "").strip() for (s, _t) in keys], dtype=str)
    gest_diab = np.asarray([pregnancy.get(s, {}).get("gestational_diabetes", "").strip() for (s, _t) in keys], dtype=str)
    hla_risk = np.asarray([pregnancy.get(s, {}).get("HLA_risk_class", "").strip() for (s, _t) in keys], dtype=str)

    stability = compute_stability_scores(keys, subject_age_to_otus, micro_to_otus, args.stability_cache)

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    axes = axes.ravel()
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_categorical(ax, labels, title, legend_title, cmap, size=10):
        uniq = [x for x in sorted(set(labels.tolist())) if x]
        color_map = {x: cmap(i % cmap.N) for i, x in enumerate(uniq)}
        for x in uniq:
            m = labels == x
            ax.scatter(xy[m, 0], xy[m, 1], s=size, alpha=0.8, c=[color_map[x]], label=str(x))
        ax.set_title(title)
        ax.legend(title=legend_title, markerscale=1.5, fontsize=8)

    # Age bins (plasma gradient in chronological order)
    bin_order = ["27-216.2", "216.2-405.3", "405.3-594.5", "594.5-783.7", "783.7-972.8", "972.8-"]
    uniq_bins = [b for b in bin_order if b in set(age_bins.tolist())]
    cmap3 = plt.cm.plasma
    color_map_bins = {
        b: cmap3(i / float(len(uniq_bins) - 1)) if len(uniq_bins) > 1 else cmap3(0.5) for i, b in enumerate(uniq_bins)
    }
    ax = axes[0]
    for b in uniq_bins:
        m = age_bins == b
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.8, c=[color_map_bins[b]], label=b)
    ax.set_title("Age at collection (binned)")
    ax.legend(title="Age bin (days)", markerscale=1.5, fontsize=8)

    plot_categorical(axes[1], countries, "Country", "Country", plt.cm.Set1)
    plot_categorical(axes[2], cohorts, "Cohort", "Cohort", plt.cm.Set3)
    plot_categorical(axes[3], locations, "Location", "Location", plt.cm.tab20)
    plot_categorical(axes[4], genders, "Gender", "Gender", plt.cm.Set2)
    plot_categorical(axes[5], csections, "C-section", "C-section", plt.cm.Pastel1)
    plot_categorical(axes[6], abx_preg, "Antibiotics while pregnant", "Abx while pregnant", plt.cm.Pastel2)
    plot_categorical(axes[7], gest_diab, "Gestational diabetes", "Gestational diabetes", plt.cm.Accent)
    plot_categorical(axes[8], hla_risk, "HLA risk class", "HLA risk class", plt.cm.Dark2)

    # Stability score (mean logit over OTUs present)
    ax = axes[9]
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=10, alpha=0.85, c=stability, cmap="coolwarm")
    ax.set_title("Stability score (mean OTU logit)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("mean logit")

    for ax in axes[10:]:
        ax.axis("off")

    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"DIABIMMUNE real embeddings (n={len(keys)})", y=0.98)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")
    print(f"Stability cache: {args.stability_cache}")


if __name__ == "__main__":
    main()
