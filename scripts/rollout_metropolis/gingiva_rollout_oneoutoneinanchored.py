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
from scripts.gingivitis.utils import collect_micro_to_otus, load_gingivitis_run_data  # noqa: E402
from scripts.rollout_metropolis.core import build_otu_index, pick_anchor_set, score_logits_for_sets, sigmoid  # noqa: E402


GINGIVA_CSV = os.path.join("data", "gingivitis", "gingiva.csv")
OUT_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob_oneoutoneinanchored.tsv")


def parse_args():
    p = argparse.ArgumentParser(description="Gingivitis full rollout using one-out-one-in with fixed anchors.")
    p.add_argument("--gingiva-csv", default=GINGIVA_CSV)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--p-anchor", type=float, default=0.95)
    p.add_argument("--candidate-pool", type=int, default=200)
    p.add_argument(
        "--early-stop-nochange",
        type=int,
        default=0,
        help="Per start: stop if the OTU set doesn't change for N consecutive steps (0 = off).",
    )
    p.add_argument("--out", default=OUT_TSV)
    return p.parse_args()


def time_key(t):
    s = str(t).strip()
    if s.upper() == "B":
        return -1e9
    try:
        return float(s)
    except Exception:
        return 1e9


def build_subject_time_to_otus(gingiva_csv, sra_to_micro, micro_to_otus):
    subject_time_to_otus = defaultdict(set)
    with open(gingiva_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("Run", "").strip()
            subj = row.get("subject_code", "").strip()
            tcode = row.get("time_code", "").strip()
            if not run or not subj or not tcode:
                continue
            srs = sra_to_micro.get(run)
            if not srs:
                continue
            otus = micro_to_otus.get(srs, [])
            if otus:
                subject_time_to_otus[(subj, tcode)].update(otus)
    return subject_time_to_otus


def sample_absent_with_embedding(rng, all_otus, current, emb_group, resolver, n, max_tries=200000):
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        o = all_otus[int(rng.integers(0, len(all_otus)))]
        if o in current:
            continue
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


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rng = np.random.default_rng(args.seed)

    _, sra_to_micro = load_gingivitis_run_data(gingivitis_path=args.gingiva_csv)
    micro_to_otus = collect_micro_to_otus(sra_to_micro)
    subject_time_to_otus = build_subject_time_to_otus(args.gingiva_csv, sra_to_micro, micro_to_otus)
    starts = [(subj, t, sorted(otus)) for (subj, t), otus in subject_time_to_otus.items() if otus]
    if not starts:
        raise SystemExit("No gingivitis start points found.")

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

    with open(args.out, "w", newline="") as out_f, h5py.File(shared_utils.PROKBERT_PATH) as emb_file:
        emb_group = emb_file["embeddings"]
        w = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()

        for subj, t_start, start_otus in tqdm(starts, desc="OneOutOneIn anchored rollouts", unit="start", dynamic_ncols=True):
            current = set(start_otus)
            if not current:
                continue

            # Define anchors at step 0.
            logits0 = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
            anchor_set = set(
                pick_anchor_set(sorted(current), logits0, p_threshold=args.p_anchor, temperature=args.temperature)
            )
            if not anchor_set:
                anchor_set = {sorted(current)[0]}
            anchor_indices = sorted({otu_to_idx[o] for o in anchor_set if o in otu_to_idx})
            anchor_indices_str = ";".join(str(i) for i in anchor_indices)

            # Record step 0 state
            anchor_vals0 = [float(logits0[a]) for a in anchor_set if a in logits0]
            anchor_mean0 = float(np.mean(anchor_vals0)) if anchor_vals0 else float("nan")
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

            nochange_run = 0
            for step_idx in range(1, int(args.steps) + 1):
                prev = set(current)

                logits_cur = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
                drop = choose_drop_lowest_prob_non_anchor(current, logits_cur, args.temperature, anchor_set)
                if drop is None or len(current) <= max(1, len(anchor_set)):
                    # No droppable non-anchor left: keep recording a steady state.
                    cur_idxs = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]
                    anchor_vals = [float(logits_cur[a]) for a in anchor_set if a in logits_cur]
                    anchor_mean = float(np.mean(anchor_vals)) if anchor_vals else float("nan")
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
                    nochange_run += 1
                    if args.early_stop_nochange and nochange_run >= int(args.early_stop_nochange):
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
                )
                if not cands:
                    current = base
                else:
                    proposals = [sorted(base | {c}) for c in cands]
                    scored = score_logits_for_sets(proposals, model, device, emb_group, resolver)
                    cand_logits = []
                    for i, c in enumerate(cands):
                        cand_logits.append(float(scored[i].get(c, float("-inf"))))
                    best_idx = int(np.argmax(np.asarray(cand_logits, dtype=float)))
                    current = set(proposals[best_idx])

                if current == prev:
                    nochange_run += 1
                else:
                    nochange_run = 0
                if args.early_stop_nochange and nochange_run >= int(args.early_stop_nochange):
                    cur_idxs = [otu_to_idx[o] for o in sorted(current) if o in otu_to_idx]
                    logits_acc = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
                    anchor_vals = [float(logits_acc[a]) for a in anchor_set if a in logits_acc]
                    anchor_mean = float(np.mean(anchor_vals)) if anchor_vals else float("nan")
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
                    break

                n_same = int(len(prev & current))
                n_added = int(len(current - prev))
                n_removed = int(len(prev - current))

                logits_acc = score_logits_for_sets([sorted(current)], model, device, emb_group, resolver)[0]
                anchor_vals = [float(logits_acc[a]) for a in anchor_set if a in logits_acc]
                anchor_mean = float(np.mean(anchor_vals)) if anchor_vals else float("nan")
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

    print(f"Saved: {args.out}")
    print("start points:", len(starts))


if __name__ == "__main__":
    main()
