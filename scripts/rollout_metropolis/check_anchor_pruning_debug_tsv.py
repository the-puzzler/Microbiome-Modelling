#!/usr/bin/env python3

import argparse
import csv


def parse_index_list(raw):
    if not raw:
        return []
    out = []
    for tok in str(raw).split(";"):
        tok = str(tok).strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            return []
    return out


def extract_indices_from_row(row, key):
    """
    Robustly extract semicolon-separated integer indices from a DictReader row.
    Handles legacy TSVs where extra columns were appended without updating header
    (DictReader stores extras under the None key).
    """
    idx = parse_index_list(row.get(key, ""))
    if idx:
        return idx
    extras = row.get(None, []) or []
    for extra in extras:
        idx = parse_index_list(extra)
        if idx:
            return idx
    for v in row.values():
        idx = parse_index_list(v)
        if idx:
            return idx
    return []


def main():
    p = argparse.ArgumentParser(
        description=(
            "Scan a metropolis single-rollout debug TSV and check whether the rollout is simply pruning "
            "non-anchor OTUs (i.e., converging to current == anchors)."
        )
    )
    p.add_argument("--tsv", required=True, help="Path to the debug TSV.")
    args = p.parse_args()

    rows = []
    with open(args.tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    if not rows:
        raise SystemExit("TSV has no rows.")

    first = rows[0]
    anchors0 = set(extract_indices_from_row(first, "anchor_otu_indices"))
    if not anchors0:
        raise SystemExit("Could not parse anchor_otu_indices from the TSV (is this an anchors-target run?).")

    missing_anchor_steps = 0
    changed_anchor_steps = 0
    reached_pure_anchors_step = None
    non_anchor_counts = []
    n_added_pos = 0
    n_removed_pos = 0

    for row in rows:
        step = int(row.get("step", "0"))
        anchors = set(extract_indices_from_row(row, "anchor_otu_indices"))
        current = set(extract_indices_from_row(row, "current_otu_indices"))

        if anchors != anchors0:
            changed_anchor_steps += 1

        if not anchors0.issubset(current):
            missing_anchor_steps += 1

        non_anchor = current - anchors0
        non_anchor_counts.append((step, len(non_anchor)))
        if len(non_anchor) == 0 and reached_pure_anchors_step is None:
            reached_pure_anchors_step = step

        try:
            if int(row.get("n_added", "0")) > 0:
                n_added_pos += 1
            if int(row.get("n_removed", "0")) > 0:
                n_removed_pos += 1
        except Exception:
            pass

    last = rows[-1]
    last_step = int(last.get("step", "0"))
    last_current = set(extract_indices_from_row(last, "current_otu_indices"))
    last_non_anchor = last_current - anchors0

    non_anchor_only_sizes = [c for (_s, c) in non_anchor_counts]
    is_monotone_nonincreasing = all(
        non_anchor_only_sizes[i + 1] <= non_anchor_only_sizes[i] for i in range(len(non_anchor_only_sizes) - 1)
    )

    print(f"TSV: {args.tsv}")
    print(f"rows: {len(rows)} (last step={last_step})")
    print(f"anchors (from step 0): {len(anchors0)}")
    print(f"anchor set changed across steps: {changed_anchor_steps} rows")
    print(f"steps where an anchor was missing from current: {missing_anchor_steps} rows")
    print(f"n_added>0 rows: {n_added_pos} (out of {len(rows)})")
    print(f"n_removed>0 rows: {n_removed_pos} (out of {len(rows)})")
    print(f"non-anchor count monotone non-increasing: {is_monotone_nonincreasing}")
    if reached_pure_anchors_step is None:
        print("current == anchors was never reached in this TSV.")
    else:
        print(f"first step where current == anchors: {reached_pure_anchors_step}")
    print(f"final non-anchor count: {len(last_non_anchor)}")

    if len(last_non_anchor) == 0 and missing_anchor_steps == 0 and changed_anchor_steps == 0:
        print("Conclusion: This rollout ends at exactly the fixed anchor set (pure pruning).")
    else:
        print("Conclusion: Not pure pruning (it keeps/creates non-anchors and/or anchors mismatch/change occurred).")


if __name__ == "__main__":
    main()

