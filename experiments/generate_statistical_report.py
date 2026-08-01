"""Generate a thesis-style statistical report from existing experiment results.

Generated with GitHub Copilot assistance - reviewed and adapted by author.

Optional dependency installation:
    pip install scipy
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics import (
    build_statistical_report,
    one_way_anova,
    pairwise_condition_comparisons,
    summarize_conditions,
)
from evaluation.plots import load_experiment_dataframe, render_thesis_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a statistical report for existing experiment batches.")
    parser.add_argument(
        "results_root",
        nargs="?",
        type=Path,
        default=Path("results/experiments"),
        help="Root directory containing experiment batch folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the report and figures should be written.",
    )
    parser.add_argument(
        "--condition-column",
        type=str,
        default="framework",
        help="Column used to separate conditions/frameworks.",
    )
    parser.add_argument(
        "--failure-modes-column",
        type=str,
        default="primary_failure_modes",
        help="Column containing the MAST primary failure mode lists.",
    )
    parser.add_argument(
        "--comparison-test",
        choices=["welch", "mannwhitney"],
        default="welch",
        help="Pairwise test to use for condition comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df = load_experiment_dataframe(args.results_root)
    if results_df.empty:
        raise SystemExit(f"No experiment batches found under {args.results_root}")

    report = build_statistical_report(
        results_df,
        condition_column=args.condition_column,
        failure_modes_column=args.failure_modes_column,
    )

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / "statistical_reports" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    render_thesis_report(
        results_df,
        figures_dir,
        condition_column=args.condition_column,
        failure_modes_column=args.failure_modes_column,
    )

    summary_df = summarize_conditions(results_df, condition_column=args.condition_column)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)

    mast_modes_df = report["mast_modes"]
    mast_categories_df = report["mast_categories"]
    mast_modes_df.to_csv(output_dir / "mast_failure_modes.csv", index=False)
    mast_categories_df.to_csv(output_dir / "mast_categories.csv", index=False)

    advanced = report.get("advanced_statistics", {})
    advanced_tables: dict[str, str] = {}
    if isinstance(advanced, dict):
        descriptive_df = advanced.get("descriptive_bootstrap")
        if hasattr(descriptive_df, "to_csv"):
            path = output_dir / "descriptive_bootstrap.csv"
            descriptive_df.to_csv(path, index=False)
            advanced_tables["descriptive_bootstrap"] = path.name

        success_tests_df = advanced.get("success_rate_tests")
        if hasattr(success_tests_df, "to_csv"):
            path = output_dir / "success_rate_tests.csv"
            success_tests_df.to_csv(path, index=False)
            advanced_tables["success_rate_tests"] = path.name

        mast_test_df = advanced.get("mast_distribution_test")
        if hasattr(mast_test_df, "to_csv"):
            path = output_dir / "mast_distribution_test.csv"
            mast_test_df.to_csv(path, index=False)
            advanced_tables["mast_distribution_test"] = path.name

        continuous_tables = advanced.get("continuous_tests", {})
        if isinstance(continuous_tables, dict):
            for metric_name, table_df in continuous_tables.items():
                if hasattr(table_df, "to_csv"):
                    path = output_dir / f"continuous_{metric_name}_tests.csv"
                    table_df.to_csv(path, index=False)
                    advanced_tables[f"continuous_{metric_name}_tests"] = path.name

    comparison_tables: dict[str, str] = {}
    for metric_name in ("success", "latency_seconds", "cost_usd"):
        if metric_name in results_df.columns:
            comparison_df = pairwise_condition_comparisons(
                results_df,
                metric_column=metric_name,
                condition_column=args.condition_column,
                test=args.comparison_test,
            )
            comparison_path = output_dir / f"pairwise_{metric_name}.csv"
            comparison_df.to_csv(comparison_path, index=False)
            comparison_tables[metric_name] = comparison_path.name

    anova_results: dict[str, dict[str, float | int | str]] = {}
    if results_df[args.condition_column].nunique() > 2:
        for metric_name in ("success", "latency_seconds", "cost_usd"):
            if metric_name in results_df.columns:
                anova_results[metric_name] = one_way_anova(
                    results_df,
                    metric_column=metric_name,
                    condition_column=args.condition_column,
                )

    report_payload = {
        "results_root": str(args.results_root),
        "num_rows": int(len(results_df)),
        "conditions": sorted(results_df[args.condition_column].astype(str).unique().tolist()),
        "summary_statistics": summary_df.to_dict(orient="records"),
        "mast_failure_modes": mast_modes_df.to_dict(orient="records"),
        "mast_categories": mast_categories_df.to_dict(orient="records"),
        "pairwise_comparison_files": comparison_tables,
        "anova": anova_results,
        "advanced_statistics_files": advanced_tables,
    }
    (output_dir / "report.json").write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_markdown_report(report_payload, summary_df), encoding="utf-8")

    print(f"Report written to {output_dir}")
    print(f"Figures written to {figures_dir}")


def _render_markdown_report(report_payload: dict, summary_df) -> str:
    lines: list[str] = []
    lines.append("# Statistical Report")
    lines.append("")
    lines.append(f"- Results root: {report_payload['results_root']}")
    lines.append(f"- Conditions: {', '.join(report_payload['conditions'])}")
    lines.append(f"- Observations: {report_payload['num_rows']}")
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(summary_df.to_string(index=False))
    lines.append("")
    lines.append("## Pairwise Comparison Files")
    lines.append("")
    for metric_name, filename in report_payload["pairwise_comparison_files"].items():
        lines.append(f"- {metric_name}: {filename}")
    lines.append("")
    lines.append("## Advanced Statistics Files")
    lines.append("")
    advanced_files = report_payload.get("advanced_statistics_files", {})
    if advanced_files:
        for key, filename in advanced_files.items():
            lines.append(f"- {key}: {filename}")
    else:
        lines.append("No advanced statistics files were generated.")
    lines.append("")
    lines.append("## ANOVA")
    lines.append("")
    if report_payload["anova"]:
        for metric_name, anova_result in report_payload["anova"].items():
            lines.append(f"### {metric_name}")
            lines.append("")
            for key, value in anova_result.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
    else:
        lines.append("ANOVA was skipped because fewer than three conditions were available.")
        lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- figures/success_rate_grouped.png")
    lines.append("- figures/success_rate_grouped.pdf")
    lines.append("- figures/mast_categories_grouped.png")
    lines.append("- figures/mast_categories_grouped.pdf")
    lines.append("- figures/latency_distribution.png")
    lines.append("- figures/latency_distribution.pdf")
    lines.append("- figures/cost_distribution.png")
    lines.append("- figures/cost_distribution.pdf")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
