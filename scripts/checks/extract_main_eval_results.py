#!/usr/bin/env python3
"""
Extract key metrics from run_main_evals.sh logs into CSV.

Parses:
- Zero-shot dropout/colonisation AUC/AP lines (base + text where present)
- Delta AUC lines
- IBS cross-country mean AUC matrix
- Infants CV mean±std metrics
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


SECTION_RE = re.compile(r"^===== (?P<section>.+?) =====\s*$")
OK_RE = re.compile(r"^\[OK\]\s+")

OVERALL_DROPOUT_RE = re.compile(
    r"^Overall(?:\s+dropout)?\s+ROC AUC:\s*(?P<auc>[0-9]*\.?[0-9]+)\s*\|\s*AP:\s*(?P<ap>[0-9]*\.?[0-9]+)\s*\|\s*dropout rate:\s*(?P<rate>[0-9]*\.?[0-9]+)\s*$"
)
WITH_TEXT_DROPOUT_RE = re.compile(
    r"^With text dropout ROC AUC:\s*(?P<auc>[0-9]*\.?[0-9]+)\s*\|\s*AP:\s*(?P<ap>[0-9]*\.?[0-9]+)\s*\|\s*dropout rate:\s*(?P<rate>[0-9]*\.?[0-9]+)\s*$"
)
BASELINE_AUC_AP_RE = re.compile(
    r"^Baseline AUC/AP \(no text\):\s*(?P<auc>[0-9]*\.?[0-9]+)\s*/\s*(?P<ap>[0-9]*\.?[0-9]+)\s*$"
)
WITH_TEXT_AUC_AP_RE = re.compile(
    r"^With text AUC/AP\s*:\s*(?P<auc>[0-9]*\.?[0-9]+)\s*/\s*(?P<ap>[0-9]*\.?[0-9]+)\s*$"
)
COLON_RE = re.compile(
    r"^(?:Colonization|Colonisation)\s+\((?P<mode>[^)]+)\)\s+[—-]\s+AUC:\s*(?P<auc>[0-9]*\.?[0-9]+)\s*\|\s*AP:\s*(?P<ap>[0-9]*\.?[0-9]+)\s*\|\s*pos_rate:\s*(?P<rate>[0-9]*\.?[0-9]+)"
)
COLON_TEXT_RE = re.compile(
    r"^(?:Colonization|Colonisation)\s+\(\+\s*text\)\s+[—-]\s+AUC:\s*(?P<auc>[0-9]*\.?[0-9]+)\s*\|\s*AP:\s*(?P<ap>[0-9]*\.?[0-9]+)\s*\|\s*(?:ΔAUC:\s*(?P<delta>[+\-]?[0-9]*\.?[0-9]+)|pos_rate:\s*(?P<rate>[0-9]*\.?[0-9]+))"
)
DELTA_AUC_RE = re.compile(r"^Delta AUC .*:\s*(?P<delta>[+\-]?[0-9]*\.?[0-9]+)\s*$")

CV_ACC_RE = re.compile(r"^CV accuracy \(mean±std\):\s*(?P<mean>[0-9]*\.?[0-9]+)\s*±\s*(?P<std>[0-9]*\.?[0-9]+)\s*$")
CV_MACRO_RE = re.compile(r"^CV macro ROC AUC \(mean±std\):\s*(?P<mean>[0-9]*\.?[0-9]+)\s*±\s*(?P<std>[0-9]*\.?[0-9]+)\s*$")

COUNTRIES_RE = re.compile(r"^Countries:\s*(?P<countries>\[.*\])\s*$")
MATRIX_LINE_RE = re.compile(r"^\[.*\]\s*$")


def parse_matrix(lines: list[str]) -> list[list[float]]:
    rows = []
    for line in lines:
        stripped = line.strip().strip("[]").strip()
        if not stripped:
            continue
        vals = [float(tok) for tok in stripped.split()]
        rows.append(vals)
    return rows


def parse_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    section = ""
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        m = SECTION_RE.match(line)
        if m:
            section = m.group("section").strip()
            continue
        if OK_RE.match(line):
            continue

        m = OVERALL_DROPOUT_RE.match(line)
        if m:
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": "dropout_base",
                    "auc": float(m.group("auc")),
                    "ap": float(m.group("ap")),
                    "rate": float(m.group("rate")),
                    "delta_auc": "",
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": "",
                    "cv_std": "",
                }
            )
            continue

        m = WITH_TEXT_DROPOUT_RE.match(line)
        if m:
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": "dropout_text",
                    "auc": float(m.group("auc")),
                    "ap": float(m.group("ap")),
                    "rate": float(m.group("rate")),
                    "delta_auc": "",
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": "",
                    "cv_std": "",
                }
            )
            continue

        m = BASELINE_AUC_AP_RE.match(line)
        if m:
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": "base_auc_ap",
                    "auc": float(m.group("auc")),
                    "ap": float(m.group("ap")),
                    "rate": "",
                    "delta_auc": "",
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": "",
                    "cv_std": "",
                }
            )
            continue

        m = WITH_TEXT_AUC_AP_RE.match(line)
        if m:
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": "text_auc_ap",
                    "auc": float(m.group("auc")),
                    "ap": float(m.group("ap")),
                    "rate": "",
                    "delta_auc": "",
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": "",
                    "cv_std": "",
                }
            )
            continue

        m = COLON_RE.match(line)
        if m:
            mode = m.group("mode").strip().lower()
            metric_type = "colonisation_text" if "text" in mode else "colonisation_base"
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": metric_type,
                    "auc": float(m.group("auc")),
                    "ap": float(m.group("ap")),
                    "rate": float(m.group("rate")),
                    "delta_auc": "",
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": "",
                    "cv_std": "",
                }
            )
            continue

        m = COLON_TEXT_RE.match(line)
        if m:
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": "colonisation_text",
                    "auc": float(m.group("auc")),
                    "ap": float(m.group("ap")),
                    "rate": float(m.group("rate")) if m.group("rate") else "",
                    "delta_auc": float(m.group("delta")) if m.group("delta") else "",
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": "",
                    "cv_std": "",
                }
            )
            continue

        m = DELTA_AUC_RE.match(line)
        if m:
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": "delta_auc",
                    "auc": "",
                    "ap": "",
                    "rate": "",
                    "delta_auc": float(m.group("delta")),
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": "",
                    "cv_std": "",
                }
            )
            continue

        m = CV_ACC_RE.match(line)
        if m:
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": "cv_accuracy",
                    "auc": "",
                    "ap": "",
                    "rate": "",
                    "delta_auc": "",
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": float(m.group("mean")),
                    "cv_std": float(m.group("std")),
                }
            )
            continue

        m = CV_MACRO_RE.match(line)
        if m:
            rows.append(
                {
                    "log_file": str(path),
                    "section": section,
                    "metric_type": "cv_macro_roc_auc",
                    "auc": "",
                    "ap": "",
                    "rate": "",
                    "delta_auc": "",
                    "source_country": "",
                    "target_country": "",
                    "cv_mean": float(m.group("mean")),
                    "cv_std": float(m.group("std")),
                }
            )
            continue

        if line.startswith("Cross-country mean AUC"):
            # Expect next lines: Countries: [...] then matrix lines
            if i < len(lines):
                m_c = COUNTRIES_RE.match(lines[i].strip())
                if m_c:
                    countries = ast.literal_eval(m_c.group("countries"))
                    i += 1
                    matrix_lines = []
                    while i < len(lines) and MATRIX_LINE_RE.match(lines[i].strip()):
                        matrix_lines.append(lines[i].strip())
                        i += 1
                        if lines[i - 1].strip().endswith("]]"):
                            break
                    mat = parse_matrix(matrix_lines)
                    for r, src in enumerate(countries):
                        for c, tgt in enumerate(countries):
                            if r < len(mat) and c < len(mat[r]):
                                rows.append(
                                    {
                                        "log_file": str(path),
                                        "section": section,
                                        "metric_type": "ibs_cross_country_auc",
                                        "auc": float(mat[r][c]),
                                        "ap": "",
                                        "rate": "",
                                        "delta_auc": "",
                                        "source_country": src,
                                        "target_country": tgt,
                                        "cv_mean": "",
                                        "cv_std": "",
                                    }
                                )
            continue
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Extract run_main_evals metrics into CSV.")
    p.add_argument("--log", required=True, help="Path to experiment log.")
    p.add_argument("--out-csv", default="", help="Output CSV path.")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    rows = parse_log(log_path)
    if not rows:
        raise SystemExit("No metrics parsed from log.")

    out_csv = Path(args.out_csv) if args.out_csv else log_path.with_name(log_path.stem + "_results.csv")
    fieldnames = [
        "log_file",
        "section",
        "metric_type",
        "auc",
        "ap",
        "rate",
        "delta_auc",
        "source_country",
        "target_country",
        "cv_mean",
        "cv_std",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Parsed {len(rows)} metrics.")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

