#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from scripts.diabimmune.utils import collect_micro_to_otus, load_run_data  # noqa: E402
from scripts.rollout_metropolis.core import (  # noqa: E402
    EmbeddingCache,
    build_otu_index,
    pick_anchor_set,
    score_logits_for_sets,
    sigmoid,
)


OUT_TSV = os.path.join("data", "diabimmune", "visionary_rollout_prob_oneoutoneinanchored.tsv")
SAMPLES_CSV = os.path.join("data", "diabimmune", "samples.csv")


def parse_args():
    p = argparse.ArgumentParser(
        description="DIABIMMUNE full rollout using one-out-one-in with fixed anchors (subject, age_at_collection_in_days)."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-starts", type=int, default=0, help="Randomly sample this many real start points (0 = all).")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--p-anchor", type=float, default=0.95)
    p.add_argument(
        "--anchor-top-pct",
        type=float,
        default=95.0,
        help="Anchors are the top PCT%% OTUs by sigmoid(logit/temperature), capped by --anchor-cap-frac.",
    )
    p.add_argument(
        "--anchor-cap-frac",
        type=float,
        default=0.5,
        help="Max fraction of the start set that can be anchors when using --anchor-top-pct (default 0.5).",
    )
    p.add_argument(
        "--anchor-use-threshold",
        action="store_true",
        help="Use threshold anchors (sigmoid(logit/temperature) >= --p-anchor) instead of top-percent anchors.",
    )
    p.add_argument("--candidate-pool", type=int, default=200)
    p.add_argument(
        "--early-stop-nobest",
        type=int,
        default=0,
        help="Per start: stop if anchor_mean_logit does not improve for N consecutive steps (0 = off).",
    )
    p.add_argument("--out", default=OUT_TSV)
    p.add_argument("--samples-csv", default=SAMPLES_CSV)
    return p.parse_args()


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def age_bin_days(age_value):
    return str(int(round(float(age_value))))


def load_samples_table(samples_csv):
    table = {}
    with open(samples_csv) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                header[0] = header[0].lstrip("\ufeff")
                continue
            row = dict(zip(header, parts))
            sid = row.get("sampleID", "").strip()
            if sid:
                table[sid] = row
    return table


def build_subject_age_to_otus(micro_to_sample, micro_to_otus, samples_table):
    subject_age_to_otus = defaultdict(set)
    for srs, info in micro_to_sample.items():
        subj = str(info.get("subject", "")).strip()
        sample_id = str(info.get("sample", "")).strip()
        if not subj or not sample_id:
            continue
        age = safe_float(samples_table.get(sample_id, {}).get("age_at_collection", ""))
        if age is None:
            continue
        age_key = age_bin_days(age)
        otus = micro_to_otus.get(srs, [])
        if otus:
            subject_age_to_otus[(subj, age_key)].update(otus)
    return subject_age_to_otus


def sample_absent_with_embedding(rng, all_otus, current, emb_group, resolver, n, *, emb_cache=None, max_tries=200000):
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        o = all_otus[int(rng.integers(0, len(all_otus)))]
        if o in current:
            continue
        if emb_cache is not None:
            if not emb_cache.has(o, emb_group, resolver):
                continue
        else:
            key = resolver.get(o, o) if resolver else o
            if key not in emb_group:
                continue
        out.append(o)
    if not out:
        return []
    return list(dict.fromkeys(out))


def choose_drop_lowest_prob_non_anchor(current, logits, temperature, anchor_set):
    anchor_set = set(anchor_set or [])
    best = None
    best_p = 1.0
    for o in current:
        if o in anchor_set:
            continue
        if o not in logits:
            return o
        p = float(sigmoid(float(logits[o]) / float(temperature)))
        if p < best_p:
            best_p = p
            best = o
    return best


def anchor_mean_logit(anchor_set, logits):
    vals = [float(logits[a]) for a in (anchor_set or []) if a in logits]
    return float(np.mean(vals)) if vals else float("nan")


def _parse_index_list(raw):
    if not raw:
        return []
    return [int(tok) for tok in str(raw).split(";") if str(tok).strip()]


def load_existing_state(tsv_path):
    if not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0:
        return {}
    state = {}
    with open(tsv_path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            subj = row.get("subject", "").strip()
            t_start = row.get("t_start", "").strip()
            if not subj or not t_start:
                continue
            try:
                step = int(row.get("step", "0"))
            except Exception:
                continue
            key = (subj, t_start)
            st = state.get(key)
            if st is None:
                st = {
                    "last_step": -1,
                    "last_row": None,
                    "best_anchor_mean": float("-inf"),
                    "nobest_run": 0,
                    "anchor_otu_indices": row.get("anchor_otu_indices", "").strip(),
                }
                state[key] = st

            try:
                v = float(row.get("anchor_mean_logit", "nan"))
            except Exception:
                v = float("nan")
            if np.isfinite(v) and v > float(st["best_anchor_mean"]):
                st["best_anchor_mean"] = float(v)
                st["nobest_run"] = 0
            else:
                st["nobest_run"] += 1

            if step >= int(st["last_step"]):
                st["last_step"] = int(step)
                st["last_row"] = row
                if row.get("anchor_otu_indices", "").strip():
                    st["anchor_otu_indices"] = row.get("anchor_otu_indices", "").strip()
    return state


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rng = np.random.default_rng(args.seed)

    _run_rows, sra_to_micro, _gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
    micro_to_otus = collect_micro_to_otus(sra_to_micro, micro_to_subject)
    samples_table = load_samples_table(args.samples_csv)
    subject_age_to_otus = build_subject_age_to_otus(micro_to_sample, micro_to_otus, samples_table)
    starts_all = [(subj, str(age), sorted(otus)) for (subj, age), otus in subject_age_to_otus.items() if otus]
    if not starts_all:
        raise SystemExit("No (subject,age) start points available for DIABIMMUNE.")

    existing = load_existing_state(args.out)
    starts_by_key = {(str(subj), str(age)): (subj, str(age), otus) for (subj, age, otus) in starts_all}

    # Always include existing keys (so resume continues the same run).
    existing_keys = list(existing.keys())
    starts = [starts_by_key[k] for k in existing_keys if k in starts_by_key]

    # Optionally add new starts up to n_starts total (including existing), else all.
    remaining = [v for k, v in starts_by_key.items() if k not in existing]
    if args.n_starts and args.n_starts > 0:
        target = int(args.n_starts)
        need = max(0, target - len(starts))
        if need and len(remaining) > need:
            idx = rng.choice(len(remaining), size=need, replace=False)
            starts.extend([remaining[i] for i in idx.tolist()])
        else:
            starts.extend(remaining)
    else:
        starts.extend(remaining)

    all_otus, otu_to_idx = build_otu_index(micro_to_otus)
    if not all_otus:
        raise SystemExit("No OTUs available for index mapping.")

    model, device = shared_utils.load_microbiome_model(shared_utils.CHECKPOINT_PATH)
    rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
    resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer="B")

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

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    mode = "a" if not write_header else "w"

    with open(args.out, mode, newline="") as out_f, h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        emb_cache = EmbeddingCache()
        w = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            w.writeheader()

        for subj, t_start, start_otus in tqdm(starts, desc="OneOutOneIn anchored rollouts", unit="start", dynamic_ncols=True):
            key = (str(subj), str(t_start))
            resume = existing.get(key)

            if resume and int(resume.get("last_step", -1)) >= int(args.steps):
                continue

            if resume and resume.get("last_row"):
                last_row = resume["last_row"]
                cur_idx = _parse_index_list(last_row.get("current_otu_indices", ""))
                current = {all_otus[i] for i in cur_idx if 0 <= i < len(all_otus)}
                anchor_idx = _parse_index_list(resume.get("anchor_otu_indices", ""))
                anchor_set = {all_otus[i] for i in anchor_idx if 0 <= i < len(all_otus)}
                if not anchor_set:
                    anchor_set = {sorted(current)[0]} if current else set()
                anchor_indices = sorted({otu_to_idx[o] for o in anchor_set if o in otu_to_idx})
                anchor_indices_str = ";".join(str(i) for i in anchor_indices)
                logits_cur = score_logits_for_sets(
                    [sorted(current)], model, device, emb_group, resolver, emb_cache=emb_cache
                )[0]
                best_anchor_mean = float(resume.get("best_anchor_mean", float("-inf")))
                nobest_run = int(resume.get("nobest_run", 0))
                start_step = int(resume.get("last_step", 0)) + 1
            else:
                current = set(start_otus)
                if not current:
                    continue

                logits_cur = score_logits_for_sets(
                    [sorted(current)], model, device, emb_group, resolver, emb_cache=emb_cache
                )[0]
                anchor_set = set(
                    pick_anchor_set(
                        sorted(current),
                        logits_cur,
                        p_threshold=args.p_anchor,
                        temperature=args.temperature,
                        top_pct=None if args.anchor_use_threshold else args.anchor_top_pct,
                        cap_frac=args.anchor_cap_frac,
                    )
                )
                if not anchor_set:
                    anchor_set = {sorted(current)[0]}
                anchor_indices = sorted({otu_to_idx[o] for o in anchor_set if o in otu_to_idx})
                anchor_indices_str = ";".join(str(i) for i in anchor_indices)

                anchor_mean0 = anchor_mean_logit(anchor_set, logits_cur)
                cur_idxs0 = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]
                w.writerow(
                    {
                        "subject": str(subj),
                        "t_start": str(t_start),
                        "step": 0,
                        "n_current": int(len(current)),
                        "n_same": int(len(current)),
                        "n_added": 0,
                        "n_removed": 0,
                        "n_anchors": int(len(anchor_indices)),
                        "anchor_otu_indices": anchor_indices_str,
                        "anchor_mean_logit": anchor_mean0,
                        "current_otu_indices": ";".join(str(i) for i in cur_idxs0),
                    }
                )

                nobest_run = 0
                best_anchor_mean = float(anchor_mean0) if np.isfinite(anchor_mean0) else float("-inf")
                start_step = 1

            for step_idx in range(int(start_step), int(args.steps) + 1):
                prev = set(current)

                drop = choose_drop_lowest_prob_non_anchor(current, logits_cur, args.temperature, anchor_set)
                if drop is None or len(current) <= max(1, len(anchor_set)):
                    cur_idxs = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]
                    anchor_mean = anchor_mean_logit(anchor_set, logits_cur)
                    w.writerow(
                        {
                            "subject": str(subj),
                            "t_start": str(t_start),
                            "step": int(step_idx),
                            "n_current": int(len(current)),
                            "n_same": int(len(current)),
                            "n_added": 0,
                            "n_removed": 0,
                            "n_anchors": int(len(anchor_indices)),
                            "anchor_otu_indices": anchor_indices_str,
                            "anchor_mean_logit": anchor_mean,
                            "current_otu_indices": ";".join(str(i) for i in cur_idxs),
                        }
                    )
                    if np.isfinite(anchor_mean) and anchor_mean > best_anchor_mean:
                        best_anchor_mean = float(anchor_mean)
                        nobest_run = 0
                    else:
                        nobest_run += 1
                    if args.early_stop_nobest and nobest_run >= int(args.early_stop_nobest):
                        break
                    continue

                base = set(current)
                base.remove(drop)

                cands = sample_absent_with_embedding(
                    rng,
                    all_otus,
                    base,
                    emb_group,
                    resolver,
                    n=int(args.candidate_pool),
                    emb_cache=emb_cache,
                )
                if not cands:
                    current = base
                    logits_cur = score_logits_for_sets(
                        [sorted(current)], model, device, emb_group, resolver, emb_cache=emb_cache
                    )[0]
                else:
                    proposals = [sorted(base | {c}) for c in cands]
                    scored = score_logits_for_sets(proposals, model, device, emb_group, resolver, emb_cache=emb_cache)
                    cand_logits = []
                    for i, c in enumerate(cands):
                        cand_logits.append(float(scored[i].get(c, float("-inf"))))
                    best_idx = int(np.argmax(np.asarray(cand_logits, dtype=float)))
                    current = set(proposals[best_idx])
                    logits_cur = scored[best_idx]

                n_same = int(len(prev & current))
                n_added = int(len(current - prev))
                n_removed = int(len(prev - current))
                anchor_mean = anchor_mean_logit(anchor_set, logits_cur)
                cur_idxs = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]

                w.writerow(
                    {
                        "subject": str(subj),
                        "t_start": str(t_start),
                        "step": int(step_idx),
                        "n_current": int(len(current)),
                        "n_same": n_same,
                        "n_added": n_added,
                        "n_removed": n_removed,
                        "n_anchors": int(len(anchor_indices)),
                        "anchor_otu_indices": anchor_indices_str,
                        "anchor_mean_logit": anchor_mean,
                        "current_otu_indices": ";".join(str(i) for i in cur_idxs),
                    }
                )

                if np.isfinite(anchor_mean) and anchor_mean > best_anchor_mean:
                    best_anchor_mean = float(anchor_mean)
                    nobest_run = 0
                else:
                    nobest_run += 1
                if args.early_stop_nobest and nobest_run >= int(args.early_stop_nobest):
                    break

    print(f"Saved: {args.out}")
    print("start points:", len(starts))


if __name__ == "__main__":
    main()
