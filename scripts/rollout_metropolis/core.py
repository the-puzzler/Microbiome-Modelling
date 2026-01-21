#!/usr/bin/env python3

import csv
import os
import random

import h5py
import numpy as np
from tqdm import tqdm


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_otu_index(micro_to_otus):
    all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
    otu_to_idx = {otu: i for i, otu in enumerate(all_otus)}
    return all_otus, otu_to_idx


def _otu_ids_to_padded_batch(otu_id_lists, emb_group, resolver, device):
    import torch

    seqs = []
    kept_ids = []
    for otu_ids in otu_id_lists:
        vecs = []
        keep = []
        for oid in otu_ids:
            key = resolver.get(oid, oid) if resolver else oid
            if key in emb_group:
                vecs.append(torch.tensor(emb_group[key][()], dtype=torch.float32))
                keep.append(oid)
        if not vecs:
            seqs.append(torch.zeros((0, 384), dtype=torch.float32))
            kept_ids.append([])
        else:
            seqs.append(torch.stack(vecs, dim=0))
            kept_ids.append(keep)

    max_len = max((s.shape[0] for s in seqs), default=0)
    if max_len == 0:
        batch = torch.zeros((len(seqs), 0, 384), dtype=torch.float32, device=device)
        mask = torch.zeros((len(seqs), 0), dtype=torch.bool, device=device)
        return batch, mask, kept_ids

    padded = []
    mask_rows = []
    for s in seqs:
        n = s.shape[0]
        if n < max_len:
            pad = torch.zeros((max_len - n, s.shape[1]), dtype=torch.float32)
            padded.append(torch.cat([s, pad], dim=0))
            m = torch.zeros((max_len,), dtype=torch.bool)
            m[:n] = True
            mask_rows.append(m)
        else:
            padded.append(s)
            mask_rows.append(torch.ones((max_len,), dtype=torch.bool))

    batch = torch.stack(padded, dim=0).to(device)
    mask = torch.stack(mask_rows, dim=0).to(device)
    return batch, mask, kept_ids


def score_logits_for_sets(otu_id_lists, model, device, emb_group, resolver=None, txt_emb=1536):
    """
    Return list[dict] mapping OTU->logit for each set, scoring logits for OTUs
    present in that set under the full-set context.
    """
    import torch

    x1, mask, kept_ids = _otu_ids_to_padded_batch(otu_id_lists, emb_group, resolver, device)
    if x1.shape[1] == 0:
        return [dict() for _ in otu_id_lists]

    x2 = torch.zeros((x1.shape[0], 0, txt_emb), dtype=torch.float32, device=device)
    full_mask = torch.ones((x1.shape[0], x1.shape[1] + x2.shape[1]), dtype=torch.bool, device=device)
    with torch.no_grad():
        h1 = model.input_projection_type1(x1)
        h2 = model.input_projection_type2(x2)
        h = torch.cat([h1, h2], dim=1)
        h = model.transformer(h, src_key_padding_mask=~full_mask)
        logits = model.output_projection(h).squeeze(-1)
        logits_type1 = logits[:, : x1.shape[1]].detach().cpu().numpy()

    out = []
    for i in range(len(otu_id_lists)):
        keep = kept_ids[i]
        if not keep:
            out.append({})
            continue
        n = len(keep)
        out.append(dict(zip(keep, logits_type1[i, :n].tolist())))
    return out


def pick_anchor_set(start_otus, logits, p_threshold=0.95, temperature=1.0):
    """
    Anchors are a fixed subset of the initial set with sigmoid(logit/temperature) >= threshold.
    """
    scored = []
    for oid in start_otus:
        if oid in logits:
            p = sigmoid(float(logits[oid]) / float(temperature))
            scored.append((oid, p, float(logits[oid])))
    scored.sort(key=lambda x: x[1], reverse=True)
    anchors = [oid for (oid, p, _l) in scored if p >= p_threshold]
    if not anchors and scored:
        anchors = [scored[0][0]]
    return set(anchors)


def objective_anchor_mean_logit(anchor_set, logits):
    if not anchor_set:
        return float("-inf")
    vals = [float(logits[a]) for a in anchor_set if a in logits]
    if not vals:
        return float("-inf")
    return float(np.mean(vals))


def metropolis_steps(
    *,
    start_otus,
    all_otus,
    otu_to_idx,
    model,
    device,
    emb_group,
    resolver,
    rng,
    steps,
    temperature,
    p_anchor=0.95,
    p_add=0.34,
    p_drop=0.33,
    p_swap=0.33,
    max_candidates=200,
):
    current = set(start_otus)
    if not current:
        return

    # Candidate pool for additions (kept static for simplicity, like the existing rollout).
    absent_pool = list(set(all_otus) - set(start_otus))
    if len(absent_pool) > max_candidates:
        absent_pool = rng.choice(np.asarray(absent_pool, dtype=object), size=max_candidates, replace=False).tolist()

    # Initialize anchors using logits from the full current set.
    logits0 = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
    anchor_set = pick_anchor_set(sorted(current), logits0, p_threshold=p_anchor, temperature=temperature)
    if not anchor_set:
        return

    weights = np.asarray([p_add, p_drop, p_swap], dtype=float)
    weights = weights / np.sum(weights)

    for step_idx in range(int(steps)):
        prev = set(current)

        move = rng.choice(["add", "drop", "swap"], p=weights)
        current_list = sorted(current)
        candidates_absent = [o for o in absent_pool if o not in current]
        candidates_drop = [o for o in current if o not in anchor_set]

        if move == "add" and not candidates_absent:
            move = "drop" if candidates_drop else "swap"
        if move == "drop" and not candidates_drop:
            move = "add" if candidates_absent else "swap"
        if move == "swap" and (not candidates_absent or not candidates_drop):
            move = "add" if candidates_absent else "drop"

        proposed = set(current)
        if move == "add" and candidates_absent:
            proposed.add(rng.choice(np.asarray(candidates_absent, dtype=object)))
        elif move == "drop" and candidates_drop:
            proposed.remove(rng.choice(np.asarray(candidates_drop, dtype=object)))
        elif move == "swap" and candidates_absent and candidates_drop:
            proposed.remove(rng.choice(np.asarray(candidates_drop, dtype=object)))
            proposed.add(rng.choice(np.asarray(candidates_absent, dtype=object)))

        # Score current and proposed in a single batched forward pass.
        logits_cur, logits_prop = score_logits_for_sets(
            [sorted(current), sorted(proposed)],
            model,
            device,
            emb_group,
            resolver,
        )
        j_cur = objective_anchor_mean_logit(anchor_set, logits_cur)
        j_prop = objective_anchor_mean_logit(anchor_set, logits_prop)
        delta = j_prop - j_cur

        accept = False
        if delta >= 0:
            accept = True
        else:
            if rng.random() < np.exp(delta / float(temperature)):
                accept = True

        if accept:
            current = proposed

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
    steps_per_rollout,
    temperature,
    checkpoint_path,
    prokbert_path,
    rename_map_path,
    prefer_resolver="B",
    p_anchor=0.95,
    p_add=0.34,
    p_drop=0.33,
    p_swap=0.33,
    max_candidates=200,
):
    from scripts import utils as shared_utils

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

        for subject, t_start, start_otus in tqdm(starts, desc="Metropolis rollouts", unit="start", dynamic_ncols=True):
            start_otus = sorted(set(start_otus))
            if not start_otus:
                continue

            for row in metropolis_steps(
                start_otus=start_otus,
                all_otus=all_otus,
                otu_to_idx=otu_to_idx,
                model=model,
                device=device,
                emb_group=emb_group,
                resolver=resolver,
                rng=rng,
                steps=steps_per_rollout,
                temperature=temperature,
                p_anchor=p_anchor,
                p_add=p_add,
                p_drop=p_drop,
                p_swap=p_swap,
                max_candidates=max_candidates,
            ):
                writer.writerow({"subject": subject, "t_start": t_start, **row})

