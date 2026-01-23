#!/usr/bin/env python3

import argparse
import csv
from collections import Counter, defaultdict

import numpy as np


def parse_index_list(raw):
    if not raw:
        return []
    out = []
    for tok in str(raw).split(";"):
        tok = str(tok).strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def load_sets(tsv_path):
    rows = []
    with open(tsv_path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            step = int(row.get("step", "0"))
            cur = set(parse_index_list(row.get("current_otu_indices", "")))
            anc = set(parse_index_list(row.get("anchor_otu_indices", ""))) if "anchor_otu_indices" in row else set()
            rows.append((step, cur, anc))
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    p = argparse.ArgumentParser(description="Analyze OTU churn from a strategy debug TSV.")
    p.add_argument("--tsv", required=True)
    p.add_argument("--top", type=int, default=25, help="Top N churniest OTUs to print.")
    p.add_argument("--bins", type=int, default=4, help="Number of step bins for churn density (e.g. quarters).")
    p.add_argument("--tail", type=int, default=50, help="How many final steps to summarize for stabilization.")
    p.add_argument(
        "--stable-jaccard",
        type=float,
        default=0.99,
        help="Jaccard threshold to consider 'close to final set' (used for a simple stabilization heuristic).",
    )
    p.add_argument(
        "--stable-window",
        type=int,
        default=25,
        help="How many consecutive steps must satisfy the stable-jaccard threshold.",
    )
    args = p.parse_args()

    rows = load_sets(args.tsv)
    if not rows:
        raise SystemExit("No rows found.")

    steps = [s for (s, _c, _a) in rows]
    sets = [c for (_s, c, _a) in rows]
    anchors = rows[0][2] if rows[0][2] else set()

    # Per-step churn
    added_counts = []
    removed_counts = []
    same_counts = []
    churn_counts = []

    # Per-OTU transition counts
    enters = Counter()
    exits = Counter()
    toggles = Counter()
    lifetimes = defaultdict(list)  # otu -> list of run lengths in consecutive steps

    prev = sets[0]
    present = set(prev)

    # Initialize lifetimes tracking
    current_run = {o: 1 for o in prev}

    for i in range(1, len(sets)):
        cur = sets[i]
        added = cur - prev
        removed = prev - cur
        same = cur & prev

        added_counts.append(len(added))
        removed_counts.append(len(removed))
        same_counts.append(len(same))
        churn_counts.append(len(added) + len(removed))

        for o in added:
            enters[o] += 1
            toggles[o] += 1
            current_run[o] = 1
        for o in removed:
            exits[o] += 1
            toggles[o] += 1
            lifetimes[o].append(current_run.get(o, 1))
            current_run.pop(o, None)
        for o in same:
            current_run[o] = current_run.get(o, 0) + 1

        prev = cur

    # Close out runs still present at the end
    for o, runlen in current_run.items():
        lifetimes[o].append(runlen)

    n_steps = len(rows) - 1
    avg_added = sum(added_counts) / max(1, n_steps)
    avg_removed = sum(removed_counts) / max(1, n_steps)

    # Unique ever seen
    ever = set().union(*sets)
    start = sets[0]
    end = sets[-1]

    # Re-add events: enters >=2
    readded = [o for o, n in enters.items() if n >= 2]

    # Churniest OTUs: most toggles
    churniest = sorted(toggles.items(), key=lambda kv: kv[1], reverse=True)

    print(f"TSV: {args.tsv}")
    print(f"steps: {len(rows)} (0..{rows[-1][0]})")
    print(f"start size: {len(start)}  end size: {len(end)}  ever size: {len(ever)}")
    print(f"anchors: {len(anchors)}")
    print(f"avg added/step: {avg_added:.2f}  avg removed/step: {avg_removed:.2f}")
    print(f"total enter events: {sum(enters.values())}  total exit events: {sum(exits.values())}")
    print(f"otus re-added (entered >=2): {len(readded)}")

    # Lifetime summary (in steps)
    all_runlens = [l for runs in lifetimes.values() for l in runs]
    if all_runlens:
        all_runlens.sort()
        p50 = all_runlens[len(all_runlens) // 2]
        p90 = all_runlens[int(0.9 * (len(all_runlens) - 1))]
        print(f"lifetime (consecutive steps) median: {p50}  p90: {p90}  max: {max(all_runlens)}")

    if churniest:
        print(f"\nTop {args.top} churniest OTUs (by toggle count):")
        for o, n in churniest[: args.top]:
            e = enters.get(o, 0)
            x = exits.get(o, 0)
            runs = lifetimes.get(o, [])
            mean_run = (sum(runs) / len(runs)) if runs else 0.0
            print(f"- otu_idx={o} toggles={n} enters={e} exits={x} mean_runlen={mean_run:.2f} runs={len(runs)}")

    # Churn density over time (bins of steps)
    if n_steps > 0 and churn_counts:
        b = max(1, int(args.bins))
        churn_arr = churn_counts  # length n_steps, corresponds to transitions 0->1 ... (n_steps-1)->n_steps
        edges = np.linspace(0, n_steps, b + 1)
        print(f"\nChurn density over steps (added+removed, {b} bins):")
        for i in range(b):
            lo = int(round(edges[i]))
            hi = int(round(edges[i + 1]))
            lo = max(0, min(lo, n_steps))
            hi = max(lo, min(hi, n_steps))
            segment = churn_arr[lo:hi]
            seg_sum = sum(segment)
            seg_mean = (seg_sum / max(1, (hi - lo))) if (hi - lo) > 0 else 0.0
            # Map bin edges to actual step numbers (rows include step 0)
            step_lo = steps[lo]
            step_hi = steps[hi] if hi < len(steps) else steps[-1]
            print(f"- bin {i+1}/{b}: transitions[{lo}:{hi}) steps[{step_lo}->{step_hi}] mean={seg_mean:.2f} sum={seg_sum}")

    # Stabilization relative to the final set
    def jaccard(a, b):
        inter = len(a & b)
        union = len(a) + len(b) - inter
        return (float(inter) / float(union)) if union else 1.0

    j_to_final = [jaccard(s, end) for s in sets]
    tail = max(1, int(args.tail))
    tail = min(tail, len(j_to_final))
    tail_vals = j_to_final[-tail:]
    print(f"\nStabilization vs final set (Jaccard to final):")
    print(f"- last {tail} steps: mean={float(np.mean(tail_vals)):.4f} min={float(np.min(tail_vals)):.4f} max={float(np.max(tail_vals)):.4f}")

    # Last step with any change
    last_change_step = None
    for i in range(1, len(sets)):
        if sets[i] != sets[i - 1]:
            last_change_step = steps[i]
    if last_change_step is None:
        last_change_step = steps[0]
    print(f"- last step with any change: {last_change_step}")

    # First step after which we're 'stable' by the heuristic: jaccard >= threshold for window steps.
    thr = float(args.stable_jaccard)
    win = max(1, int(args.stable_window))
    stable_start = None
    ok = np.asarray([v >= thr for v in j_to_final], dtype=bool)
    if ok.size >= win:
        run = 0
        for i, flag in enumerate(ok.tolist()):
            run = run + 1 if flag else 0
            if run >= win:
                stable_start = steps[i - win + 1]
                break
    if stable_start is None:
        print(f"- heuristic stable start: not reached (need J>= {thr:g} for {win} steps)")
    else:
        print(f"- heuristic stable start: step {stable_start} (J>= {thr:g} for {win} steps)")


if __name__ == "__main__":
    main()
