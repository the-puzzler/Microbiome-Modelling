#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.plot_diabimmune_embedding_metadata_panels import (  # noqa: E402
    PREGNANCY_BIRTH_CSV,
    REAL_NPZ,
    SAMPLES_CSV,
    STABILITY_NPZ,
    age_bin_label,
    build_subject_age_to_otus,
    compute_stability_scores,
    load_csv_table,
    pca2_fit,
    pca2_transform,
)


ENDPOINTS_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoints_cache.npz")
OUT_PNG = os.path.join("data", "diabimmune", "diabimmune_embedding_metadata_panels_with_rollouts.png")
ROLL_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob.tsv")
ENDPOINT_STABILITY_NPZ = os.path.join("data", "diabimmune", "diabimmune_rollout_endpoint_stability_cache.npz")


def parse_args():
    p = argparse.ArgumentParser(description="DIABIMMUNE: metadata PCA panels + rollout endpoints overlay.")
    p.add_argument("--real-npz", default=REAL_NPZ)
    p.add_argument("--endpoints-npz", default=ENDPOINTS_NPZ)
    p.add_argument("--rollout-tsv", default=ROLL_TSV)
    p.add_argument("--samples-csv", default=SAMPLES_CSV)
    p.add_argument("--pregnancy-birth-csv", default=PREGNANCY_BIRTH_CSV)
    p.add_argument("--stability-cache", default=STABILITY_NPZ)
    p.add_argument("--endpoint-stability-cache", default=ENDPOINT_STABILITY_NPZ)
    p.add_argument("--out", default=OUT_PNG)
    p.add_argument("--method", choices=["pca", "umap"], default="pca")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=0, help="Optional cap for faster debugging (0 = all).")
    return p.parse_args()


def parse_index_list(raw):
    if not raw:
        return []
    return [int(tok) for tok in str(raw).split(";") if str(tok).strip()]


def load_rollout_endpoint_indices(rollout_tsv):
    endpoint_by_parent = {}
    with open(rollout_tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            subj = row.get("subject", "").strip()
            t_start = row.get("t_start", "").strip()
            if not subj or not t_start:
                continue
            step = int(row.get("step", "0"))
            idxs = parse_index_list(row.get("current_otu_indices", ""))
            key = (subj, t_start)
            prev = endpoint_by_parent.get(key)
            if prev is None or step >= prev[0]:
                endpoint_by_parent[key] = (step, idxs)
    return {k: v[1] for k, v in endpoint_by_parent.items()}


def compute_endpoint_stability(endpoint_parents, endpoint_indices_by_parent, all_otus, micro_to_otus, cache_path):
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        cached_keys = [tuple(map(str, k)) for k in cache["keys"]]
        if cached_keys == endpoint_parents:
            return cache["endpoint_stability"].astype(float)

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

    out = np.full((len(endpoint_parents),), np.nan, dtype=float)
    with h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        for i, key in enumerate(tqdm(endpoint_parents, desc="Endpoint stability (mean logit)", unit="endpoint")):
            idxs = endpoint_indices_by_parent.get(key, [])
            otus = [all_otus[j] for j in idxs if 0 <= j < len(all_otus)]
            logits = shared_utils.score_otu_list(otus, resolver=resolver, model=model, device=device, emb_group=emb_group)
            if logits:
                out[i] = float(np.mean(list(logits.values())))

    np.savez(cache_path, keys=np.asarray(endpoint_parents, dtype=object), endpoint_stability=out)
    return out


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    real_cache = np.load(args.real_npz, allow_pickle=True)
    keys_raw = real_cache["keys"]
    emb = real_cache["emb"].astype(float)
    keys = [(str(keys_raw[i][0]), str(keys_raw[i][1])) for i in range(len(keys_raw))]

    endpoints_cache = np.load(args.endpoints_npz, allow_pickle=True)
    endpoints = endpoints_cache["endpoints"].astype(float)
    parent_subject = endpoints_cache["subject"].astype(str)
    parent_t1 = endpoints_cache["t1"].astype(str)
    endpoint_parents = [(parent_subject[i], parent_t1[i]) for i in range(len(parent_subject))]

    if args.max_samples and len(keys) > args.max_samples:
        keep = args.max_samples
        key_set = set(keys[:keep])
        keys = keys[:keep]
        emb = emb[:keep]
        mask_ep = np.asarray([k in key_set for k in endpoint_parents], dtype=bool)
        endpoints = endpoints[mask_ep]
        endpoint_parents = [k for k, m in zip(endpoint_parents, mask_ep.tolist()) if m]

    if args.method == "pca":
        mu, comps = pca2_fit(emb)
        xy = pca2_transform(emb, mu, comps)
        end_xy = pca2_transform(endpoints, mu, comps)
        xlab, ylab = "PC1", "PC2"
        method_title = "PCA"
    else:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            random_state=args.seed,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
        )
        all_xy = reducer.fit_transform(np.vstack([emb, endpoints]))
        xy = all_xy[: len(emb)]
        end_xy = all_xy[len(emb) :]
        xlab, ylab = "UMAP1", "UMAP2"
        method_title = "UMAP"

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
    key_to_stability = {keys[i]: stability[i] for i in range(len(keys))}

    all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
    endpoint_indices_by_parent = load_rollout_endpoint_indices(args.rollout_tsv)
    endpoint_stability = compute_endpoint_stability(
        endpoint_parents,
        endpoint_indices_by_parent,
        all_otus,
        micro_to_otus,
        args.endpoint_stability_cache,
    )

    def labels_for_parents(mapping):
        return np.asarray([mapping.get(k, "") for k in endpoint_parents], dtype=str)

    key_to_age_bin = {keys[i]: age_bins[i] for i in range(len(keys))}
    key_to_country = {keys[i]: countries[i] for i in range(len(keys))}
    key_to_cohort = {keys[i]: cohorts[i] for i in range(len(keys))}
    key_to_gender = {keys[i]: genders[i] for i in range(len(keys))}
    key_to_location = {keys[i]: locations[i] for i in range(len(keys))}
    key_to_csection = {keys[i]: csections[i] for i in range(len(keys))}
    key_to_abx = {keys[i]: abx_preg[i] for i in range(len(keys))}
    key_to_gd = {keys[i]: gest_diab[i] for i in range(len(keys))}
    key_to_hla = {keys[i]: hla_risk[i] for i in range(len(keys))}

    ep_age_bins = labels_for_parents(key_to_age_bin)
    ep_countries = labels_for_parents(key_to_country)
    ep_cohorts = labels_for_parents(key_to_cohort)
    ep_locations = labels_for_parents(key_to_location)
    ep_genders = labels_for_parents(key_to_gender)
    ep_csections = labels_for_parents(key_to_csection)
    ep_abx = labels_for_parents(key_to_abx)
    ep_gd = labels_for_parents(key_to_gd)
    ep_hla = labels_for_parents(key_to_hla)

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

    s_real = 10
    s_end = 4
    alpha_real = 0.80
    alpha_end = 0.55

    vmin = float(np.nanmin(stability)) if np.isfinite(stability).any() else None
    vmax = float(np.nanmax(stability)) if np.isfinite(stability).any() else None
    if np.isfinite(endpoint_stability).any():
        vmin_ep = float(np.nanmin(endpoint_stability))
        vmax_ep = float(np.nanmax(endpoint_stability))
        vmin = min(vmin, vmin_ep) if vmin is not None else vmin_ep
        vmax = max(vmax, vmax_ep) if vmax is not None else vmax_ep

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
        me = ep_age_bins == b
        if np.any(me):
            ax.scatter(
                end_xy[me, 0],
                end_xy[me, 1],
                s=s_end,
                alpha=alpha_end,
                c=endpoint_stability[me],
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
        if np.any(m):
            ax.scatter(xy[m, 0], xy[m, 1], s=s_real, alpha=alpha_real, c=[color_map_bins[b]], label=b)
    ax.set_title("Age at collection (binned)")
    ax.legend(title="Age bin (days)", markerscale=1.5, fontsize=8)

    def plot_categorical(ax, real_labels, end_labels, title, legend_title, cmap):
        uniq = [x for x in sorted(set(real_labels.tolist())) if x]
        color_map = {x: cmap(i % cmap.N) for i, x in enumerate(uniq)}
        for x in uniq:
            m = real_labels == x
            me = end_labels == x
            if np.any(me):
                ax.scatter(
                    end_xy[me, 0],
                    end_xy[me, 1],
                    s=s_end,
                    alpha=alpha_end,
                    c=endpoint_stability[me],
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                )
            if np.any(m):
                ax.scatter(xy[m, 0], xy[m, 1], s=s_real, alpha=alpha_real, c=[color_map[x]], label=str(x))
        ax.set_title(title)
        ax.legend(title=legend_title, markerscale=1.5, fontsize=8)

    plot_categorical(axes[1], countries, ep_countries, "Country", "Country", plt.cm.Set1)
    plot_categorical(axes[2], cohorts, ep_cohorts, "Cohort", "Cohort", plt.cm.Set3)
    plot_categorical(axes[3], locations, ep_locations, "Location", "Location", plt.cm.tab20)
    plot_categorical(axes[4], genders, ep_genders, "Gender", "Gender", plt.cm.Set2)
    plot_categorical(axes[5], csections, ep_csections, "C-section", "C-section", plt.cm.Pastel1)
    plot_categorical(axes[6], abx_preg, ep_abx, "Antibiotics while pregnant", "Abx while pregnant", plt.cm.Pastel2)
    plot_categorical(axes[7], gest_diab, ep_gd, "Gestational diabetes", "Gestational diabetes", plt.cm.Accent)
    plot_categorical(axes[8], hla_risk, ep_hla, "HLA risk class", "HLA risk class", plt.cm.Dark2)

    # Stability score (mean logit over OTUs present) + endpoints colored by endpoint stability
    ax = axes[9]
    if np.isfinite(endpoint_stability).any():
        ax.scatter(
            end_xy[:, 0],
            end_xy[:, 1],
            s=s_end,
            alpha=alpha_end,
            c=endpoint_stability,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=s_real, alpha=0.85, c=stability, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title("Stability score (mean OTU logit)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("mean logit")
    fig.text(0.99, 0.01, "Small points (endpoints) coloured by endpoint stability", ha="right", va="bottom", fontsize=9)

    for ax in axes[10:]:
        ax.axis("off")

    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    fig.suptitle(
        f"DIABIMMUNE real embeddings + rollout endpoints ({method_title}, n_real={len(keys)}, n_end={len(end_xy)})",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
