"""Reanalyze baseline artifacts with thesis-grade validity filters and protocol-aligned tests.

Usage:
  uv run python experiments/reanalyze_baseline.py results/coordination_baseline_2026-08-09/experiments \
    --output-dir results/coordination_baseline_2026-08-09/reports/thesis_clean_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics import build_statistical_report, summarize_conditions
from evaluation.multiple_comparisons import add_holm_adjustment
from evaluation.plots import load_experiment_dataframe
from evaluation.run_validity import annotate_validity, filter_valid_runs, validity_summary
from evaluation.statistical_analysis import compare_continuous, compare_success_rates
from evaluation.task_success import annotate_criteria_success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reanalyze baseline runs under validity filters.")
    parser.add_argument("results_root", type=Path, help="Experiment root with batch artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-frameworks", nargs="*", default=["metagpt"], help="Frameworks excluded from primary tables.")
    parser.add_argument("--require-model-judge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_experiment_dataframe(args.results_root)
    if raw.empty:
        raise SystemExit(f"No experiment records found under {args.results_root}")

    annotated = annotate_validity(raw)
    annotated = annotate_criteria_success(annotated)
    validity = validity_summary(annotated)
    validity.to_csv(args.output_dir / "validity_summary.csv", index=False)
    annotated.to_csv(args.output_dir / "all_runs_annotated.csv", index=False)

    clean = filter_valid_runs(
        annotated,
        exclude_scaffold=True,
        require_model_judge=args.require_model_judge,
    )
    if args.exclude_frameworks:
        clean = clean.loc[~clean["framework"].astype(str).str.lower().isin([f.lower() for f in args.exclude_frameworks])].copy()

    clean.to_csv(args.output_dir / "valid_runs.csv", index=False)

    # Runner-claimed success report
    runner_report = build_statistical_report(clean, condition_column="framework")
    runner_report["summary"].to_csv(args.output_dir / "summary_runner_success.csv", index=False)
    runner_report["mast_modes"].to_csv(args.output_dir / "mast_modes.csv", index=False)
    runner_report["mast_categories"].to_csv(args.output_dir / "mast_categories.csv", index=False)

    success_tests = add_holm_adjustment(compare_success_rates(clean, "framework"))
    success_tests.to_csv(args.output_dir / "pairwise_success_chi2_fisher_holm.csv", index=False)

    for metric in ("latency_seconds", "cost_usd"):
        if metric in clean.columns:
            continuous = add_holm_adjustment(compare_continuous(clean, "framework", metric))
            continuous.to_csv(args.output_dir / f"pairwise_{metric}_holm.csv", index=False)

    # Independent criteria success, where available
    if "criteria_success" in clean.columns and clean["criteria_success"].notna().any():
        criteria_df = clean.dropna(subset=["criteria_success"]).copy()
        criteria_df["success"] = criteria_df["criteria_success"].astype(bool)
        criteria_summary = summarize_conditions(criteria_df, condition_column="framework")
        criteria_summary.to_csv(args.output_dir / "summary_criteria_success.csv", index=False)
        criteria_tests = add_holm_adjustment(compare_success_rates(criteria_df, "framework"))
        criteria_tests.to_csv(args.output_dir / "pairwise_criteria_success_holm.csv", index=False)

    manifest = {
        "source": str(args.results_root),
        "n_raw": int(len(annotated)),
        "n_valid": int(len(clean)),
        "excluded_frameworks": args.exclude_frameworks,
        "require_model_judge": args.require_model_judge,
        "notes": [
            "Scaffold/fallback runs are excluded from primary tables.",
            "Success pairwise tests use Chi-square/Fisher with Holm adjustment.",
            "criteria_success is an auditable keyword-anchor approximation of the codebook.",
            "Human adjudication remains the reference standard for Phase C.",
        ],
    }
    (args.output_dir / "reanalysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
