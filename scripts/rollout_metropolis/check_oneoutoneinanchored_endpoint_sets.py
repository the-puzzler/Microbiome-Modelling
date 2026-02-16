#!/usr/bin/env python3

import argparse
import csv
import hashlib
import os
from collections import Counter


DEFAULT_TSV = os.path.join("data", "gingivitis", "visionary_rollout_prob_oneoutoneinanchored.tsv")


def parse_args():
    p = argparse.ArgumentParser(
        description="Check whether oneoutoneinanchored endpoints collapse (OTU sets + optional jacc_end_xy uniqueness)."
    )
    p.add_argument("--tsv", default=DEFAULT_TSV, help="Rollout TSV to scan.")
    p.add_argument("--window", type=int, default=10, help="Stop after N no-new-best steps.")
    p.add_argument(
        "--overlay-cache",
        default=os.path.join("data", "gingivitis", "gingivitis_rollout_trajectory_overlay_cache_oneoutoneinanchored.npz"),
        help="Optional overlay cache .npz (if present, reports jacc_end_xy uniqueness).",
    )
    return p.parse_args()


def _hash_indices(raw):
    toks = [t for t in str(raw or "").split(";") if t]
    norm = ";".join(sorted(toks))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest(), len(toks)


def iter_best_endpoints(tsv_path, *, window):
    cur_key = None
    best = float("-inf")
    best_row = None
    since_best = 0
    stopped = False

    def flush():
        nonlocal cur_key, best_row
        if cur_key is None or best_row is None:
            return None
        return cur_key, best_row

    with open(tsv_path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            key = (row.get("subject", "").strip(), row.get("t_start", "").strip())
            if cur_key is None:
                cur_key = key
            if key != cur_key:
                out = flush()
                if out is not None:
                    yield out
                cur_key = key
                best = float("-inf")
                best_row = None
                since_best = 0
                stopped = False

            if stopped:
                continue

            try:
                v = float(row.get("anchor_mean_logit", "nan"))
            except Exception:
                v = float("nan")
            if v == v and v > best:
                best = float(v)
                best_row = row
                since_best = 0
            else:
                since_best += 1
                if since_best >= int(window):
                    stopped = True

    out = flush()
    if out is not None:
        yield out


def main():
    args = parse_args()
    if not os.path.exists(args.tsv):
        raise SystemExit("Missing TSV: %s" % args.tsv)

    hashes = []
    sizes = []
    for (_subject, _t_start), row in iter_best_endpoints(args.tsv, window=args.window):
        h, n = _hash_indices(row.get("current_otu_indices", ""))
        hashes.append(h)
        sizes.append(n)

    c = Counter(hashes)
    print("TSV:", args.tsv)
    print("starts:", len(hashes))
    print("unique endpoint OTU sets:", len(c))
    if sizes:
        print("endpoint set size: min=%d max=%d" % (min(sizes), max(sizes)))
    if c:
        print("most common endpoint set count:", c.most_common(1)[0][1])

    cache_path = args.overlay_cache
    if os.path.exists(cache_path):
        try:
            import numpy as np

            z = np.load(cache_path, allow_pickle=True)
            if "jacc_end_xy" in z.files:
                xy = np.asarray(z["jacc_end_xy"], dtype=float)
                uniq = np.unique(np.round(xy, 6), axis=0).shape[0] if xy.size else 0
                print("\nOverlay cache:", cache_path)
                print("jacc_end_xy:", tuple(xy.shape), "unique (rounded 1e-6):", uniq)
            else:
                print("\nOverlay cache:", cache_path)
                print("no jacc_end_xy in cache")
        except Exception as e:
            print("\nOverlay cache:", cache_path)
            print("failed to read cache via numpy:", e)
    else:
        print("\nOverlay cache missing (skipping jacc_end_xy check):", cache_path)


if __name__ == "__main__":
    main()

