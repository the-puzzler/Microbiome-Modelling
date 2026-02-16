#!/usr/bin/env python3

import csv
import os
import random

import h5py
import numpy as np
from tqdm import tqdm


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class EmbeddingCache:
    """
    Small utility to avoid repeated HDF5 reads / tensor materialization.

    - Caches per-OTU embedding tensors keyed by resolved embedding id.
    - Caches missing keys to avoid repeated membership checks.
    """

    def __init__(self, *, max_items=50000):
        from collections import OrderedDict

        self.max_items = int(max_items) if max_items else 0
        self._tensors = OrderedDict()
        self._missing = set()

    def _resolve_key(self, oid, resolver):
        return resolver.get(oid, oid) if resolver else oid

    def has(self, oid, emb_group, resolver=None):
        key = self._resolve_key(oid, resolver)
        if key in self._tensors:
            self._tensors.move_to_end(key)
            return True
        if key in self._missing:
            return False
        ok = key in emb_group
        if not ok:
            self._missing.add(key)
        return ok

    def get_tensor(self, oid, emb_group, resolver=None):
        import torch

        key = self._resolve_key(oid, resolver)
        if key in self._tensors:
            t = self._tensors[key]
            self._tensors.move_to_end(key)
            return t
        if key in self._missing:
            return None
        if key not in emb_group:
            self._missing.add(key)
            return None
        arr = emb_group[key][()]
        t = torch.as_tensor(arr, dtype=torch.float32)
        self._tensors[key] = t
        if self.max_items and len(self._tensors) > self.max_items:
            self._tensors.popitem(last=False)
        return t


def build_otu_index(micro_to_otus):
    all_otus = sorted({oid for otus in micro_to_otus.values() for oid in otus})
    otu_to_idx = {otu: i for i, otu in enumerate(all_otus)}
    return all_otus, otu_to_idx


def compute_embedding_from_otus(
    otu_ids,
    model,
    device,
    emb_group,
    resolver=None,
    scratch_tokens=0,
    d_model=None,
    *,
    emb_cache=None,
):
    import torch

    vecs = []
    for oid in otu_ids:
        if emb_cache is not None:
            t = emb_cache.get_tensor(oid, emb_group, resolver)
            if t is not None:
                vecs.append(t.to(device))
            continue
        key = resolver.get(oid, oid) if resolver else oid
        if key in emb_group:
            vecs.append(torch.as_tensor(emb_group[key][()], dtype=torch.float32, device=device))
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


def _otu_ids_to_padded_batch(otu_id_lists, emb_group, resolver, device, *, emb_cache=None):
    import torch

    seqs = []
    kept_ids = []
    for otu_ids in otu_id_lists:
        vecs = []
        keep = []
        for oid in otu_ids:
            if emb_cache is not None:
                t = emb_cache.get_tensor(oid, emb_group, resolver)
                if t is not None:
                    vecs.append(t)
                    keep.append(oid)
                continue
            key = resolver.get(oid, oid) if resolver else oid
            if key in emb_group:
                vecs.append(torch.as_tensor(emb_group[key][()], dtype=torch.float32))
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


def score_logits_for_sets(otu_id_lists, model, device, emb_group, resolver=None, txt_emb=1536, *, emb_cache=None):
    """
    Return list[dict] mapping OTU->logit for each set, scoring logits for OTUs
    present in that set under the full-set context.
    """
    import torch

    x1, mask, kept_ids = _otu_ids_to_padded_batch(otu_id_lists, emb_group, resolver, device, emb_cache=emb_cache)
    if x1.shape[1] == 0:
        return [dict() for _ in otu_id_lists]

    x2 = torch.zeros((x1.shape[0], 0, txt_emb), dtype=torch.float32, device=device)
    full_mask = torch.ones((x1.shape[0], x1.shape[1] + x2.shape[1]), dtype=torch.bool, device=device)
    with torch.inference_mode():
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


def pick_anchor_set(start_otus, logits, p_threshold=0.95, temperature=1.0, *, top_pct=None, cap_frac=0.5):
    """
    Pick a fixed anchor subset from the starting OTUs.

    Two modes:
    - Threshold mode (default): anchors are OTUs with sigmoid(logit/temperature) >= p_threshold.
    - Top-percent mode: set `top_pct` (0..100) to select the top `top_pct` percent by probability,
      capped to at most `cap_frac` of the scored OTUs (default 50%).
    """
    scored = []
    for oid in start_otus:
        if oid in logits:
            p = sigmoid(float(logits[oid]) / float(temperature))
            scored.append((oid, float(p), float(logits[oid])))
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return set()

    if top_pct is None:
        anchors = [oid for (oid, p, _l) in scored if p >= p_threshold]
        if not anchors:
            anchors = [scored[0][0]]
        return set(anchors)

    try:
        top_pct_f = float(top_pct)
    except Exception:
        top_pct_f = 0.0
    top_pct_f = max(0.0, min(100.0, top_pct_f))

    n_scored = len(scored)
    k = int(np.ceil((top_pct_f / 100.0) * float(n_scored)))
    try:
        cap_f = float(cap_frac)
    except Exception:
        cap_f = 0.5
    cap_f = max(0.0, min(1.0, cap_f))
    k_cap = int(np.floor(cap_f * float(n_scored)))
    k = min(k, k_cap if k_cap > 0 else n_scored)
    k = max(1, min(k, n_scored))

    return {oid for (oid, _p, _l) in scored[:k]}


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
    n_proposals=10,
):
    current = set(start_otus)
    if not current:
        return

    # Initialize anchors using logits from the full current set.
    logits0 = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
    anchor_set = pick_anchor_set(sorted(current), logits0, p_threshold=p_anchor, temperature=temperature)
    if not anchor_set:
        return
    anchor_indices = sorted({otu_to_idx[o] for o in anchor_set if o in otu_to_idx})
    anchor_indices_str = ";".join(str(i) for i in anchor_indices)
    n_anchors = int(len(anchor_indices))

    weights = np.asarray([p_add, p_drop, p_swap], dtype=float)
    weights = weights / np.sum(weights)

    def sample_absent():
        if len(current) >= len(all_otus):
            return None
        # Rejection-sample from the full OTU universe until we hit an absent one.
        for _ in range(1000):
            o = all_otus[int(rng.integers(0, len(all_otus)))]
            if o not in current:
                return o
        # Fallback: build the explicit list (slow, but should almost never happen).
        absent = [o for o in all_otus if o not in current]
        return rng.choice(np.asarray(absent, dtype=object)) if absent else None

    for step_idx in range(int(steps)):
        prev = set(current)

        candidates_drop = [o for o in current if o not in anchor_set]

        proposals = []
        n_prop = max(1, int(n_proposals))
        for _ in range(n_prop):
            move = rng.choice(["add", "drop", "swap"], p=weights)

            if move == "add" and len(current) >= len(all_otus):
                move = "drop" if candidates_drop else "swap"
            if move == "drop" and not candidates_drop:
                move = "add" if len(current) < len(all_otus) else "swap"
            if move == "swap" and (len(current) >= len(all_otus) or not candidates_drop):
                move = "add" if len(current) < len(all_otus) else "drop"

            proposed = set(current)
            if move == "add":
                o = sample_absent()
                if o is not None:
                    proposed.add(o)
            elif move == "drop" and candidates_drop:
                proposed.remove(rng.choice(np.asarray(candidates_drop, dtype=object)))
            elif move == "swap" and candidates_drop:
                proposed.remove(rng.choice(np.asarray(candidates_drop, dtype=object)))
                o = sample_absent()
                if o is not None:
                    proposed.add(o)

            proposals.append(sorted(proposed))

        # Score current + all proposals in a single batched forward pass.
        scored = score_logits_for_sets(
            [sorted(current)] + proposals,
            model,
            device,
            emb_group,
            resolver,
        )
        j_cur = objective_anchor_mean_logit(anchor_set, scored[0])
        j_props = [objective_anchor_mean_logit(anchor_set, d) for d in scored[1:]]

        best_idx = int(np.nanargmax(np.asarray(j_props, dtype=float))) if j_props else -1
        j_best = float(j_props[best_idx]) if best_idx >= 0 else j_cur
        proposed_best = set(proposals[best_idx]) if best_idx >= 0 else set(current)
        delta = j_best - j_cur

        accept = False
        if delta >= 0:
            accept = True
        else:
            if rng.random() < np.exp(delta / float(temperature)):
                accept = True

        anchor_mean_logit = j_cur
        if accept:
            current = proposed_best
            anchor_mean_logit = j_best

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
            "n_anchors": n_anchors,
            "anchor_otu_indices": anchor_indices_str,
            "anchor_mean_logit": float(anchor_mean_logit),
            "current_otu_indices": ";".join(str(i) for i in current_indices),
        }


def metropolis_steps_fixed_anchors(
    *,
    current_otus,
    anchor_set,
    all_otus,
    otu_to_idx,
    model,
    device,
    emb_group,
    resolver,
    rng,
    steps,
    temperature,
    step_offset=0,
    p_add=0.34,
    p_drop=0.33,
    p_swap=0.33,
    n_proposals=10,
):
    """
    Resume-compatible metropolis steps with a fixed anchor set.

    - `current_otus`: starting state (set/list of OTU ids)
    - `anchor_set`: OTU ids to keep fixed (not droppable)
    - `step_offset`: added to the output `step` (so you can continue numbering)
    """
    current = set(current_otus)
    anchor_set = set(anchor_set)
    if not current or not anchor_set:
        return

    anchor_indices = sorted({otu_to_idx[o] for o in anchor_set if o in otu_to_idx})
    anchor_indices_str = ";".join(str(i) for i in anchor_indices)
    n_anchors = int(len(anchor_indices))

    weights = np.asarray([p_add, p_drop, p_swap], dtype=float)
    weights = weights / np.sum(weights)

    def sample_absent():
        if len(current) >= len(all_otus):
            return None
        for _ in range(1000):
            o = all_otus[int(rng.integers(0, len(all_otus)))]
            if o not in current:
                return o
        absent = [o for o in all_otus if o not in current]
        return rng.choice(np.asarray(absent, dtype=object)) if absent else None

    for step_idx in range(int(steps)):
        prev = set(current)
        candidates_drop = [o for o in current if o not in anchor_set]

        proposals = []
        n_prop = max(1, int(n_proposals))
        for _ in range(n_prop):
            move = rng.choice(["add", "drop", "swap"], p=weights)

            if move == "add" and len(current) >= len(all_otus):
                move = "drop" if candidates_drop else "swap"
            if move == "drop" and not candidates_drop:
                move = "add" if len(current) < len(all_otus) else "swap"
            if move == "swap" and (len(current) >= len(all_otus) or not candidates_drop):
                move = "add" if len(current) < len(all_otus) else "drop"

            proposed = set(current)
            if move == "add":
                o = sample_absent()
                if o is not None:
                    proposed.add(o)
            elif move == "drop" and candidates_drop:
                proposed.remove(rng.choice(np.asarray(candidates_drop, dtype=object)))
            elif move == "swap" and candidates_drop:
                proposed.remove(rng.choice(np.asarray(candidates_drop, dtype=object)))
                o = sample_absent()
                if o is not None:
                    proposed.add(o)

            proposals.append(sorted(proposed))

        scored = score_logits_for_sets(
            [sorted(current)] + proposals,
            model,
            device,
            emb_group,
            resolver,
        )
        j_cur = objective_anchor_mean_logit(anchor_set, scored[0])
        j_props = [objective_anchor_mean_logit(anchor_set, d) for d in scored[1:]]

        best_idx = int(np.nanargmax(np.asarray(j_props, dtype=float))) if j_props else -1
        j_best = float(j_props[best_idx]) if best_idx >= 0 else j_cur
        proposed_best = set(proposals[best_idx]) if best_idx >= 0 else set(current)
        delta = j_best - j_cur

        accept = False
        if delta >= 0:
            accept = True
        else:
            if rng.random() < np.exp(delta / float(temperature)):
                accept = True

        anchor_mean_logit = j_cur
        if accept:
            current = proposed_best
            anchor_mean_logit = j_best

        n_same = int(len(prev & current))
        n_added = int(len(current - prev))
        n_removed = int(len(prev - current))
        current_indices = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]

        yield {
            "step": int(step_offset) + step_idx + 1,
            "n_current": int(len(current)),
            "n_same": n_same,
            "n_added": n_added,
            "n_removed": n_removed,
            "n_anchors": n_anchors,
            "anchor_otu_indices": anchor_indices_str,
            "anchor_mean_logit": float(anchor_mean_logit),
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
    n_proposals=10,
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
        "n_anchors",
        "anchor_otu_indices",
        "anchor_mean_logit",
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
                n_proposals=n_proposals,
            ):
                writer.writerow({"subject": subject, "t_start": t_start, **row})
