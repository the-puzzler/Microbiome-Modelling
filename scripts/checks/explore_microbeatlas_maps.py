#%% Overview
import os
import sys
from pathlib import Path
from collections import Counter

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import utils as shared_utils  # noqa: E402

DATA_DIR = Path('data/microbeatlas')
MAPPED_PATH = DATA_DIR / 'samples-otus.97.mapped'
RENAME_MAP_PATH = DATA_DIR / 'otus.rename.map1'
DIABIMMUNE_SAMPLES_TSV = Path('data/diabimmune/microbeatlas_samples.tsv')


def fmt_size(n):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


print('Files:')
for p in [MAPPED_PATH, RENAME_MAP_PATH, DIABIMMUNE_SAMPLES_TSV]:
    if p.exists():
        st = p.stat()
        print(f"- {p} | size={fmt_size(st.st_size)}")
    else:
        print(f"- {p} | MISSING")


#%% Helper: peek first N lines and try to infer delimiter and column count
def peek_lines(path, n=10):
    lines = []
    with open(path, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            lines.append(line.rstrip('\n'))
            if i + 1 >= n:
                break
    return lines


def infer_delimiter(lines):
    cand = ['\t', ',', ' ', ';', '|']
    scores = {}
    for d in cand:
        try:
            counts = [len([p for p in l.split(d) if p != '']) for l in lines if l]
        except Exception:
            counts = []
        if counts:
            # Prefer consistent column counts across lines
            var = max(counts) - min(counts)
            scores[d] = (var, -Counter(counts).most_common(1)[0][1])
    if not scores:
        return None
    # Choose delimiter with smallest variance, break ties by most common count
    best = sorted(scores.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0][0]
    return best


#%% Explore otus.rename.map1
if RENAME_MAP_PATH.exists():
    lines = peek_lines(RENAME_MAP_PATH, n=20)
    delim = infer_delimiter(lines) or '\t'
    print(f"\nPreview: {RENAME_MAP_PATH}")
    print(f"  inferred delimiter: {repr(delim)}")
    for i, l in enumerate(lines[:5]):
        print(f"  L{i+1}: {l}")
    # Attempt to parse into pairs
    pairs = []
    for l in lines:
        if not l:
            continue
        parts = [p for p in l.split(delim) if p != '']
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    if pairs:
        print(f"  sample pairs (first 5): {pairs[:5]}")
        left_ex = [p[0] for p in pairs]
        right_ex = [p[1] for p in pairs]
        print(f"  left example ids: {left_ex[:3]}")
        print(f"  right example ids: {right_ex[:3]}")
    else:
        print("  could not parse pairs from preview lines")
else:
    print(f"\nMissing file: {RENAME_MAP_PATH}")


#%% Explore samples-otus.97.mapped (very large, preview only)
if MAPPED_PATH.exists():
    lines = peek_lines(MAPPED_PATH, n=20)
    delim = infer_delimiter(lines) or '\t'
    print(f"\nPreview: {MAPPED_PATH}")
    print(f"  inferred delimiter: {repr(delim)}")
    for i, l in enumerate(lines[:5]):
        print(f"  L{i+1}: {l}")
    # Heuristic: does it look like header? check first line tokens
    if lines:
        first = [p for p in lines[0].split(delim) if p != '']
        looks_header = any(tok.isalpha() and not tok.isdigit() for tok in first)
        print(f"  header-like first row: {looks_header}")
        print(f"  first row fields: {len(first)}")
    # Inspect column count distribution in preview
    counts = [len([p for p in l.split(delim) if p != '']) for l in lines if l]
    print(f"  preview column counts: {counts}")
else:
    print(f"\nMissing file: {MAPPED_PATH}")


#%% Optional: tie to microbeatlas_samples.tsv if present (SRS ids)
SAMPLES_TSV = DIABIMMUNE_SAMPLES_TSV
if SAMPLES_TSV.exists():
    tsv_lines = peek_lines(SAMPLES_TSV, n=5)
    print(f"\nPreview: {SAMPLES_TSV}")
    for i, l in enumerate(tsv_lines):
        print(f"  L{i+1}: {l}")
else:
    print(f"\nMissing file: {SAMPLES_TSV}")


#%% Notes
print("\nNotes:")
print("- This script only previews the first few lines to avoid loading huge files.")
print("- If you want deeper parsing (counts, schema), we can add streaming readers.")

# %%
