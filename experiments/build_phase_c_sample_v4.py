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

    export_columns = [c for c in ANNOTATION_TEMPLATE_COLUMNS if c in sampled.columns]
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

    validity = validity_summary(pool)
    validity.to_csv(args.output_dir / "source_pool_validity.csv", index=False)

    success_counts = {}
    if "success" in sampled.columns:
        success_counts = sampled["success"].fillna(False).astype(bool).value_counts().to_dict()

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
        "success_counts": {str(k): int(v) for k, v in success_counts.items()},
        "files": {
            "template": str(template_path),
            "master": str(master_path),
            "blinded_reviewer1": str(blinded_path_1),
            "blinded_reviewer2": str(blinded_path_2),
        },
        "design_notes": [
            "Failures are retained unless --success-only is explicitly set.",
            "Scaffold/fallback runs are excluded.",
            "MetaGPT is excluded by default until a native non-scaffold baseline exists.",
            "Double-coding and adjudication are still required before H5 claims.",
        ],
    }
    (args.output_dir / "annotation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
