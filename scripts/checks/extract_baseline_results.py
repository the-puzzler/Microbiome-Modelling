#!/usr/bin/env python3
"""
Extract baseline results from log files.

Parses:
1) Loose random 80/20 split macro AUC lines
2) Grouped k-fold macro AUC mean ± std lines

Works with both MLP baseline logs and classic LR/RF baseline logs.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict


SECTION_RE = re.compile(r"^===== (?P<section>.+?) =====\s*$")
TASK_HEADER_RE = re.compile(r"^=== (?P<dataset>[^:]+): .*?(?P<task>Colonisation|Dropout).*===\s*$", re.IGNORECASE)

RANDOM_MLP_RE = re.compile(
    r"^\[(?P<task>[^\]]+)\]\s+Random split macro AUC \(MLP\):\s*"
    r"(?P<auc>[0-9]*\.?[0-9]+)\s*\(labels with both classes:\s*(?P<labels>\d+)\)\s*$"
)
RANDOM_PLAIN_RE = re.compile(
    r"^Random split macro AUC:\s*(?P<auc>[0-9]*\.?[0-9]+)\s*"
    r"\(labels with both classes:\s*(?P<labels>\d+)\)\s*$"
)
GROUPED_RE = re.compile(
    r"^\[(?P<model>[^\]]+)\]\s+Grouped\s+(?P<nfold>\d+)-fold macro AUC:\s*"
    r"(?P<mean>[0-9]*\.?[0-9]+)\s*±\s*(?P<std>[0-9]*\.?[0-9]+)\s*$"
)


def norm_task(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "colon" in s:
        return "Colonisation"
    if "drop" in s:
        return "Dropout"
    return (raw or "").strip()


def infer_task_from_section(section: str) -> str:
    s = (section or "").lower()
    if "colon" in s:
        return "Colonisation"
    if "drop" in s:
        return "Dropout"
    return ""


def infer_dataset_from_section(section: str) -> str:
    s = (section or "").lower()
    if "gingivitis" in s or "gingiva" in s:
        return "Gingivitis"
    if "snowmelt" in s:
        return "Snowmelt"
    if "diabimmune" in s:
        return "DIABIMMUNE"
    return ""


def model_family_from_name(name: str) -> str:
    s = (name or "").lower()
    if "mlp" in s:
        return "MLP"
    if "logreg" in s:
        return "LogReg"
    if "randforest" in s or "rf" in s:
        return "RandForest"
    return (name or "").strip()


def parse_log(log_path: Path) -> list[dict]:
    rows = []
    section = ""
    dataset = ""
    current_task = ""
    with log_path.open("r", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            m = SECTION_RE.match(line)
            if m:
                section = m.group("section").strip()
                if not dataset:
                    dataset = infer_dataset_from_section(section)
                if not current_task:
                    current_task = infer_task_from_section(section)
                continue

            m = TASK_HEADER_RE.match(line)
            if m:
                dataset = m.group("dataset").strip()
                current_task = norm_task(m.group("task"))
                continue

            m = RANDOM_MLP_RE.match(line)
            if m:
                task = norm_task(m.group("task"))
                rows.append(
                    {
                        "log_file": str(log_path),
                        "section": section,
                        "dataset": dataset,
                        "task": task,
                        "model_name": "MLP",
                        "model_family": "MLP",
                        "result_type": "random_split_80_20",
                        "macro_auc_mean": float(m.group("auc")),
                        "macro_auc_std": "",
                        "n_folds_or_runs": 1,
                        "labels_with_both_classes": int(m.group("labels")),
                        "line_no": lineno,
                    }
                )
                continue

            m = RANDOM_PLAIN_RE.match(line)
            if m:
                task = current_task or infer_task_from_section(section)
                ds = dataset or infer_dataset_from_section(section)
                rows.append(
                    {
                        "log_file": str(log_path),
                        "section": section,
                        "dataset": ds,
                        "task": task,
                        "model_name": "LogReg (OvR)",
                        "model_family": "LogReg",
                        "result_type": "random_split_80_20",
                        "macro_auc_mean": float(m.group("auc")),
                        "macro_auc_std": "",
                        "n_folds_or_runs": 1,
                        "labels_with_both_classes": int(m.group("labels")),
                        "line_no": lineno,
                    }
                )
                continue

            m = GROUPED_RE.match(line)
            if m:
                model_name = m.group("model").strip()
                task = norm_task(model_name) or current_task or infer_task_from_section(section)
                ds = dataset or infer_dataset_from_section(section)
                rows.append(
                    {
                        "log_file": str(log_path),
                        "section": section,
                        "dataset": ds,
                        "task": task,
                        "model_name": model_name,
                        "model_family": model_family_from_name(model_name),
                        "result_type": "grouped_cv",
                        "macro_auc_mean": float(m.group("mean")),
                        "macro_auc_std": float(m.group("std")),
                        "n_folds_or_runs": int(m.group("nfold")),
                        "labels_with_both_classes": "",
                        "line_no": lineno,
                    }
                )
                continue
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract random-split and grouped-CV baseline AUCs from a log.")
    p.add_argument("--log", required=True, help="Path to baseline log file.")
    p.add_argument("--out-csv", default="", help="Detailed output CSV path.")
    p.add_argument("--out-summary-csv", default="", help="Pivot/summary output CSV path.")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    rows = parse_log(log_path)
    if not rows:
        raise SystemExit("No baseline result lines found in log.")

    out_csv = Path(args.out_csv) if args.out_csv else log_path.with_name(log_path.stem + "_baseline_results.csv")
    out_summary = (
        Path(args.out_summary_csv)
        if args.out_summary_csv
        else log_path.with_name(log_path.stem + "_baseline_results_summary.csv")
    )

    fieldnames = [
        "log_file",
        "section",
        "dataset",
        "task",
        "model_name",
        "model_family",
        "result_type",
        "macro_auc_mean",
        "macro_auc_std",
        "n_folds_or_runs",
        "labels_with_both_classes",
        "line_no",
    ]
    write_csv(out_csv, fieldnames, rows)

    # Summary: one row per (section, dataset, task, model_family)
    by_key = defaultdict(dict)
    for r in rows:
        key = (r["section"], r["dataset"], r["task"], r["model_family"])
        if r["result_type"] == "random_split_80_20":
            by_key[key]["random_split_macro_auc"] = r["macro_auc_mean"]
            by_key[key]["random_split_labels_with_both_classes"] = r["labels_with_both_classes"]
        elif r["result_type"] == "grouped_cv":
            by_key[key]["grouped_cv_macro_auc_mean"] = r["macro_auc_mean"]
            by_key[key]["grouped_cv_macro_auc_std"] = r["macro_auc_std"]
            by_key[key]["grouped_cv_n_folds"] = r["n_folds_or_runs"]

    summary_rows = []
    for (section, dataset, task, model_family), vals in sorted(by_key.items()):
        mean = vals.get("grouped_cv_macro_auc_mean")
        std = vals.get("grouped_cv_macro_auc_std")
        grouped_pm = ""
        if mean is not None and std is not None:
            grouped_pm = f"{float(mean):.3f} ± {float(std):.3f}"
        summary_rows.append(
            {
                "section": section,
                "dataset": dataset,
                "task": task,
                "model_family": model_family,
                "random_split_macro_auc": vals.get("random_split_macro_auc", ""),
                "random_split_labels_with_both_classes": vals.get("random_split_labels_with_both_classes", ""),
                "grouped_cv_macro_auc_mean": vals.get("grouped_cv_macro_auc_mean", ""),
                "grouped_cv_macro_auc_std": vals.get("grouped_cv_macro_auc_std", ""),
                "grouped_cv_n_folds": vals.get("grouped_cv_n_folds", ""),
                "grouped_cv_macro_auc_mean_pm_std": grouped_pm,
            }
        )

    summary_fieldnames = [
        "section",
        "dataset",
        "task",
        "model_family",
        "random_split_macro_auc",
        "random_split_labels_with_both_classes",
        "grouped_cv_macro_auc_mean",
        "grouped_cv_macro_auc_std",
        "grouped_cv_n_folds",
        "grouped_cv_macro_auc_mean_pm_std",
    ]
    write_csv(out_summary, summary_fieldnames, summary_rows)

    print(f"Parsed {len(rows)} result lines.")
    print(f"Wrote detailed CSV: {out_csv}")
    print(f"Wrote summary CSV: {out_summary}")
    print("")
    for r in summary_rows:
        print(
            f"{r['section']} | {r['task']} | {r['model_family']} | "
            f"random={r['random_split_macro_auc']} | grouped={r['grouped_cv_macro_auc_mean_pm_std']}"
        )


if __name__ == "__main__":
    main()
