#!/usr/bin/env python3

import csv
import os
import random

import h5py
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_logits(otu_ids, model, device, emb_group, resolver=None):
    from scripts import utils as shared_utils

    return shared_utils.score_otu_list(
        otu_ids,
        resolver=resolver,
        model=model,
        device=device,
        emb_group=emb_group,
    )


def build_otu_index(micro_to_otus):
    all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
    otu_to_idx = {otu: i for i, otu in enumerate(all_otus)}
    return all_otus, otu_to_idx


def compute_embedding_from_otus(otu_ids, model, device, emb_group, resolver=None, scratch_tokens=0, d_model=None):
    import torch

    vecs = []
    for oid in otu_ids:
        key = resolver.get(oid, oid) if resolver else oid
        if key in emb_group:
            vecs.append(torch.tensor(emb_group[key][()], dtype=torch.float32, device=device))
    if not vecs:
        return None
    x1 = torch.stack(vecs, dim=0).unsqueeze(0)
    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        if scratch_tokens > 0:
            if d_model is None:
                raise ValueError("d_model must be provided when scratch_tokens > 0")
            z = torch.zeros((1, scratch_tokens, d_model), dtype=torch.float32, device=device)
            h = torch.cat([h1, z], dim=1)
        else:
            h = h1
        mask = torch.ones((1, h.shape[1]), dtype=torch.bool, device=device)
        h = model.transformer(h, src_key_padding_mask=~mask)
        vec = h.mean(dim=1).squeeze(0).cpu().numpy()
    return vec


def rollout_steps(
    *,
    start_otus,
    absent_pool,
    otu_to_idx,
    model,
    device,
    emb_group,
    resolver,
    rng,
    steps,
    temperature,
):
    current = set(start_otus)
    for step_idx in range(int(steps)):
        prev = set(current)
        candidate_pool = list(current) + absent_pool
        logits_step = score_logits(candidate_pool, model, device, emb_group, resolver)
        if not logits_step:
            break

        present = [(oid, logits_step.get(oid, None)) for oid in current if oid in logits_step]
        absent_step = [oid for oid in absent_pool if oid not in current]
        absent_scored = [(oid, logits_step.get(oid, None)) for oid in absent_step if oid in logits_step]
        if not present or not absent_scored:
            break

        kept = []
        for oid, score in present:
            p_keep = sigmoid(score / temperature)
            if rng.random() < p_keep:
                kept.append(oid)

        added = []
        for oid, score in absent_scored:
            p_add = sigmoid(score / temperature)
            if rng.random() < p_add:
                added.append(oid)

        current = set(kept) | set(added)
        n_same = int(len(prev & current))
        n_added = int(len(current - prev))
        n_removed = int(len(prev - current))
        current_indices = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]

        yield {
            "step": step_idx + 1,
            "n_current": int(len(current)),
            "n_same": n_same,
            "n_added": n_added,
            "n_removed": n_removed,
            "current_otu_indices": ";".join(str(i) for i in current_indices),
        }


def write_rollout_tsv(
    *,
    out_tsv,
    starts,  # list[tuple[subject, t_start, list_of_otus]]
    micro_to_otus,
    seed,
    max_candidates,
    steps_per_rollout,
    temperature,
    checkpoint_path,
    prokbert_path,
    rename_map_path,
    prefer_resolver="B",
    scratch_tokens=0,
    d_model=None,
):
    from scripts import utils as shared_utils
    from tqdm import tqdm

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    all_otus, otu_to_idx = build_otu_index(micro_to_otus)
    if not all_otus:
        raise SystemExit("No OTUs available for candidate pool.")

    model, device = shared_utils.load_microbiome_model(checkpoint_path)
    rename_map = shared_utils.load_otu_rename_map(rename_map_path) if os.path.exists(rename_map_path) else None
    resolver = (
        shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, prokbert_path, prefer=prefer_resolver)
        if rename_map
        else {}
    )

    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    fieldnames = [
        "subject",
        "t_start",
        "step",
        "n_current",
        "n_same",
        "n_added",
        "n_removed",
        "current_otu_indices",
    ]
    with open(out_tsv, "w", newline="") as out_f, h5py.File(prokbert_path) as emb_file:
        emb_group = emb_file["embeddings"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for subject, t_start, start_otus in tqdm(
            starts,
            desc="Rollouts",
            unit="start",
            dynamic_ncols=True,
        ):
            start_otus = sorted(set(start_otus))
            if not start_otus:
                continue
            absent = list(set(all_otus) - set(start_otus))
            if len(absent) > max_candidates:
                absent = rng.choice(np.asarray(absent, dtype=object), size=max_candidates, replace=False).tolist()

            for row in rollout_steps(
                start_otus=start_otus,
                absent_pool=absent,
                otu_to_idx=otu_to_idx,
                model=model,
                device=device,
                emb_group=emb_group,
                resolver=resolver,
                rng=rng,
                steps=steps_per_rollout,
                temperature=temperature,
            ):
                writer.writerow({"subject": subject, "t_start": t_start, **row})
