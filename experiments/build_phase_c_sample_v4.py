"""Build a thesis-grade Phase C annotation sample.

Design goals:
- source from a single frozen baseline root when possible
- exclude scaffold/fallback runs
- include both successes and failures
- cover all available frameworks x tasks with balanced cells
- default target n >= 60

Usage:
  uv run python experiments/build_phase_c_sample_v4.py \
    results/coordination_baseline_2026-08-09/experiments \
    --output-dir results/judge_validation_phase_c_v4 \
    --sample-size 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.judge_alignment import (
    ANNOTATION_TEMPLATE_COLUMNS,
    filter_annotation_candidates,
    load_runs_for_annotation,
    sample_annotation_rows,
)
from evaluation.run_validity import annotate_validity, validity_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase C v4 annotation sample.")
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/judge_validation_phase_c_v4"))
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-frameworks", nargs="*", default=["metagpt"])
    parser.add_argument("--success-only", action="store_true", help="Deprecated legacy mode; do not use for H5.")
    return parser.parse_args()


def _ensure_annotation_item_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "annotation_item_id" in out.columns and out["annotation_item_id"].astype(str).str.strip().ne("").all():
        return out
    framework_series = out["framework"] if "framework" in out.columns else pd.Series([""] * len(out))
    benchmark_series = out["benchmark"] if "benchmark" in out.columns else pd.Series([""] * len(out))
    run_index_series = out["run_index"] if "run_index" in out.columns else pd.Series(range(len(out)))
    parts = (
        framework_series.astype(str)
        + "::"
        + benchmark_series.astype(str)
        + "::"
        + run_index_series.astype(str)
    )
    out["annotation_item_id"] = parts
    return out


def _build_annotation_worklist(sampled: pd.DataFrame, reviewers: tuple[str, str] = ("reviewer1", "reviewer2")) -> pd.DataFrame:
    required_cols = ["annotation_item_id", "framework", "benchmark", "run_id", "raw_log_path"]
    work = sampled.copy()
    for col in required_cols:
        if col not in work.columns:
            work[col] = ""

    rows: list[dict[str, object]] = []
    grouped = {
        key: group.copy().reset_index(drop=True)
        for key, group in work.groupby(["framework", "benchmark"], dropna=False, sort=True)
    }
    ordered_keys = sorted(grouped.keys(), key=lambda k: (str(k[0]), str(k[1])))

    for reviewer in reviewers:
        sequence = 1
        cursors = {k: 0 for k in ordered_keys}
        remaining = True
        while remaining:
            remaining = False
            for key in ordered_keys:
                group = grouped[key]
                cursor = cursors[key]
                if cursor >= len(group):
                    continue
                remaining = True
                current = group.iloc[cursor]
                cursors[key] = cursor + 1
                rows.append(
                    {
                        "reviewer": reviewer,
                        "sequence": sequence,
                        "annotation_item_id": current.get("annotation_item_id", ""),
                        "framework": current.get("framework", ""),
                        "benchmark": current.get("benchmark", ""),
                        "run_id": current.get("run_id", ""),
                        "raw_log_path": current.get("raw_log_path", ""),
                        "missing_fields": "",
                    }
                )
                sequence += 1
    return pd.DataFrame(rows)


def _write_v4_readme(output_dir: Path) -> None:
    text = """# Phase C v4 Commands

## QC

uv run python experiments/check_phase_c_annotation_quality.py \\
  --sample-root results/judge_validation_phase_c_v4

## Adjudication build

uv run python experiments/prepare_phase_c_adjudication.py \\
  results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer1.csv \\
  results/judge_validation_phase_c_v4/annotation_sheet_blinded_reviewer2.csv \\
  --output-csv results/judge_validation_phase_c_v4/annotation_adjudication.csv

## Agreement

uv run python experiments/run_phase_c_agreement.py \\
  --adjudication-csv results/judge_validation_phase_c_v4/annotation_adjudication.csv \\
  --master-csv results/judge_validation_phase_c_v4/annotation_master.csv \\
  --output-dir results/judge_validation_phase_c_v4/agreement
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = load_runs_for_annotation(args.results_root)
    pool = annotate_validity(pool)
    pool = filter_annotation_candidates(
        pool,
        real_model_only=True,
        success_only=args.success_only,
    )
    if args.exclude_frameworks and "framework" in pool.columns:
        pool = pool.loc[~pool["framework"].astype(str).str.lower().isin([f.lower() for f in args.exclude_frameworks])].copy()

    if pool.empty:
        raise SystemExit("No eligible runs left after validity filtering.")

    sampled = sample_annotation_rows(
        pool,
        sample_size=min(args.sample_size, len(pool)),
        seed=args.seed,
        balance_success=not args.success_only,
    )
    sampled = _ensure_annotation_item_id(sampled)

    if "success" in sampled.columns:
        success_series = sampled["success"].fillna(False).astype(bool)
        n_success = int(success_series.sum())
        n_failure = int((~success_series).sum())
    else:
        n_success = 0
        n_failure = 0

    if (not args.success_only) and n_failure == 0:
        raise SystemExit("Phase C v4 sample rejected: n_failure == 0. Rebuild sample for H5 with failure coverage.")

    export_columns = ["annotation_item_id"] + [c for c in ANNOTATION_TEMPLATE_COLUMNS if c in sampled.columns]
    template_path = args.output_dir / "annotation_template.csv"
    sampled[export_columns].to_csv(template_path, index=False)

    # Blinded sheets for two reviewers (judge labels removed from the working view).
    blinded = sampled[export_columns].copy()
    for column in ("judge_task_successful", "judge_primary_failure_modes", "judge_summary"):
        if column in blinded.columns:
            blinded[column] = ""
    blinded_path_1 = args.output_dir / "annotation_sheet_blinded_reviewer1.csv"
    blinded_path_2 = args.output_dir / "annotation_sheet_blinded_reviewer2.csv"
    blinded.to_csv(blinded_path_1, index=False)
    blinded.to_csv(blinded_path_2, index=False)

    master_path = args.output_dir / "annotation_master.csv"
    sampled[export_columns].to_csv(master_path, index=False)

    worklist_path = args.output_dir / "annotation_worklist.csv"
    _build_annotation_worklist(sampled).to_csv(worklist_path, index=False)
    _write_v4_readme(args.output_dir)

    validity = validity_summary(pool)
    validity.to_csv(args.output_dir / "source_pool_validity.csv", index=False)

    success_counts = {"true": n_success, "false": n_failure}

    cell_counts: dict[str, dict[str, int]] = {}
    unbalanced_cells: list[dict[str, object]] = []
    if {"framework", "benchmark", "success"}.issubset(sampled.columns):
        for (framework, benchmark), group in sampled.groupby(["framework", "benchmark"], dropna=False):
            success_bool = group["success"].fillna(False).astype(bool)
            cell_success = int(success_bool.sum())
            cell_failure = int((~success_bool).sum())
            key = f"{framework}::{benchmark}"
            cell_counts[key] = {"n_total": int(len(group)), "n_success": cell_success, "n_failure": cell_failure}
            if cell_success == 0 or cell_failure == 0:
                unbalanced_cells.append(
                    {
                        "framework": str(framework),
                        "benchmark": str(benchmark),
                        "n_total": int(len(group)),
                        "n_success": cell_success,
                        "n_failure": cell_failure,
                    }
                )

    manifest = {
        "source": str(args.results_root),
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "n_pool": int(len(pool)),
        "n_sample": int(len(sampled)),
        "exclude_frameworks": args.exclude_frameworks,
        "success_only": args.success_only,
        "framework_counts": sampled["framework"].value_counts().to_dict() if "framework" in sampled.columns else {},
        "benchmark_counts": sampled["benchmark"].value_counts().to_dict() if "benchmark" in sampled.columns else {},
        "success_counts": success_counts,
        "n_success": n_success,
        "n_failure": n_failure,
        "cell_counts": cell_counts,
        "unbalanced_cells": unbalanced_cells,
        "files": {
            "template": str(template_path),
            "master": str(master_path),
            "blinded_reviewer1": str(blinded_path_1),
            "blinded_reviewer2": str(blinded_path_2),
            "worklist": str(worklist_path),
        },
        "design_notes": [
            "Failures are retained unless --success-only is explicitly set.",
            "Scaffold/fallback runs are excluded.",
            "MetaGPT is excluded by default until a native non-scaffold baseline exists.",
            "Double-coding and adjudication are still required before H5 claims.",
        ],
    }
    (args.output_dir / "annotation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if cell_counts:
        print("framework_x_benchmark_cell_counts:")
        print(json.dumps(cell_counts, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
