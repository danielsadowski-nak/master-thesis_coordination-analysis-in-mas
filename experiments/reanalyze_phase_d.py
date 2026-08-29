"""Reanalyze Phase D mitigation artifacts with confirmatory settings.

No experiment execution is performed by this script. It consumes existing run artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics import pairwise_condition_comparisons
from evaluation.multiple_comparisons import add_holm_adjustment
from evaluation.plots import load_experiment_dataframe
from evaluation.run_validity import annotate_validity, filter_valid_runs, validity_summary
from evaluation.statistical_analysis import compare_success_rates
from evaluation.task_success import annotate_criteria_success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reanalyze Phase D runs with confirmatory criteria_success and latency tests.")
    parser.add_argument("results_root", nargs="?", type=Path, default=None, help="Optional primary experiment root.")
    parser.add_argument(
        "--results-root",
        dest="results_roots",
        action="append",
        type=Path,
        default=[],
        help="Repeatable experiment root argument.",
    )
    parser.add_argument("--extra-roots", nargs="*", type=Path, default=[], help="Additional experiment roots.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-frameworks",
        nargs="*",
        default=["metagpt"],
        help="Frameworks excluded from confirmatory tables (default: metagpt).",
    )
    parser.add_argument("--require-model-judge", action="store_true")
    return parser.parse_args()


def _resolve_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    if args.results_root is not None:
        roots.append(args.results_root)
    roots.extend(args.results_roots)
    roots.extend(args.extra_roots)

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _with_task_condition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    benchmark_parts = out["benchmark"].astype(str).str.split("/")
    out["condition"] = benchmark_parts.str[-1]
    out["task_id"] = benchmark_parts.str[-2]
    return out


def _is_nonempty_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "none"}


def _filter_nonempty_traces(df: pd.DataFrame) -> pd.DataFrame:
    if "raw_log_path" not in df.columns:
        return df
    mask = df["raw_log_path"].apply(_is_nonempty_text)
    return df.loc[mask].copy()


def _criteria_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["framework", "condition"], dropna=False)
    for (framework, condition), group in grouped:
        values = group["criteria_success"].astype(bool)
        n = int(len(values))
        n_success = int(values.sum())
        rows.append(
            {
                "framework": framework,
                "condition": condition,
                "n": n,
                "n_success": n_success,
                "success_rate": float(n_success / n) if n else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["framework", "condition", "n", "n_success", "success_rate"])
    return pd.DataFrame(rows).sort_values(["framework", "condition"]).reset_index(drop=True)


def _add_holm_per_framework(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    adjusted_frames: list[pd.DataFrame] = []
    for framework, group in df.groupby("framework", dropna=False):
        adjusted = add_holm_adjustment(group.reset_index(drop=True))
        adjusted["framework"] = framework
        adjusted_frames.append(adjusted)
    return pd.concat(adjusted_frames, ignore_index=True)


def _pairwise_criteria_success(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for framework, framework_df in df.groupby("framework", dropna=False):
        available_conditions = sorted(set(framework_df["condition"].astype(str)))
        if "none" not in available_conditions:
            continue
        for strategy in [c for c in available_conditions if c != "none"]:
            subset = framework_df.loc[framework_df["condition"].astype(str).isin(["none", strategy]), ["condition", "criteria_success"]].copy()
            subset["success"] = subset["criteria_success"].astype(bool)
            tests = compare_success_rates(subset, "condition")
            if tests.empty:
                continue
            row = tests.iloc[0].to_dict()
            none_rate = float(row["success_rate_a"]) if str(row["group_a"]) == "none" else float(row["success_rate_b"])
            strategy_rate = float(row["success_rate_b"]) if str(row["group_b"]) == strategy else float(row["success_rate_a"])
            rows.append(
                {
                    "framework": framework,
                    "baseline_condition": "none",
                    "strategy_condition": strategy,
                    "test": row.get("test"),
                    "statistic": row.get("statistic"),
                    "p_value": row.get("p_value"),
                    "odds_ratio": row.get("odds_ratio"),
                    "n_none": row.get("n_a") if str(row.get("group_a")) == "none" else row.get("n_b"),
                    "n_strategy": row.get("n_b") if str(row.get("group_b")) == strategy else row.get("n_a"),
                    "success_rate_none": none_rate,
                    "success_rate_strategy": strategy_rate,
                    "risk_difference": strategy_rate - none_rate,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _add_holm_per_framework(out)
    return out.sort_values(["framework", "strategy_condition"]).reset_index(drop=True)


def _pairwise_latency(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for framework, framework_df in df.groupby("framework", dropna=False):
        available_conditions = sorted(set(framework_df["condition"].astype(str)))
        if "none" not in available_conditions:
            continue
        for strategy in [c for c in available_conditions if c != "none"]:
            subset = framework_df.loc[
                framework_df["condition"].astype(str).isin(["none", strategy]),
                ["condition", "latency_seconds"],
            ].copy()
            subset["latency_seconds"] = pd.to_numeric(subset["latency_seconds"], errors="coerce")
            subset = subset.dropna(subset=["latency_seconds"])
            if subset.empty:
                continue
            tests = pairwise_condition_comparisons(
                subset,
                metric_column="latency_seconds",
                condition_column="condition",
                test="mannwhitney",
            )
            if tests.empty:
                continue
            row = tests.iloc[0].to_dict()
            rows.append(
                {
                    "framework": framework,
                    "baseline_condition": "none",
                    "strategy_condition": strategy,
                    "test_name": row.get("test_name"),
                    "statistic": row.get("statistic"),
                    "p_value": row.get("p_value"),
                    "effect_size": row.get("effect_size"),
                    "effect_size_name": row.get("effect_size_name"),
                    "n_none": row.get("n_a") if str(row.get("condition_a")) == "none" else row.get("n_b"),
                    "n_strategy": row.get("n_b") if str(row.get("condition_b")) == strategy else row.get("n_a"),
                    "median_none": row.get("median_a") if str(row.get("condition_a")) == "none" else row.get("median_b"),
                    "median_strategy": row.get("median_b") if str(row.get("condition_b")) == strategy else row.get("median_a"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _add_holm_per_framework(out)
    return out.sort_values(["framework", "strategy_condition"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots = _resolve_roots(args)
    if not roots:
        raise SystemExit("No experiment roots provided.")

    raw_frames: list[pd.DataFrame] = []
    per_root_counts: list[dict[str, Any]] = []
    for root in roots:
        loaded = load_experiment_dataframe(root)
        per_root_counts.append({"root": str(root), "n_rows": int(len(loaded))})
        if not loaded.empty:
            loaded = loaded.copy()
            loaded["source_root"] = str(root)
            raw_frames.append(loaded)

    if not raw_frames:
        raise SystemExit(f"No experiment records found under requested roots: {[str(root) for root in roots]}")

    raw = pd.concat(raw_frames, ignore_index=True)
    annotated = annotate_criteria_success(annotate_validity(raw))
    annotated = _with_task_condition(annotated)
    annotated.to_csv(args.output_dir / "all_runs_annotated.csv", index=False)

    validity = validity_summary(annotated)
    validity.to_csv(args.output_dir / "validity_summary.csv", index=False)

    clean = filter_valid_runs(annotated, exclude_scaffold=True, require_model_judge=args.require_model_judge)
    clean = _filter_nonempty_traces(clean)

    excluded_frameworks = [f.lower() for f in (args.exclude_frameworks or [])]
    if excluded_frameworks:
        clean = clean.loc[~clean["framework"].astype(str).str.lower().isin(excluded_frameworks)].copy()

    clean.to_csv(args.output_dir / "valid_runs.csv", index=False)

    criteria_df = clean.dropna(subset=["criteria_success", "condition", "framework"]).copy()
    criteria_df["criteria_success"] = criteria_df["criteria_success"].astype(bool)

    summary_criteria = _criteria_summary(criteria_df)
    summary_criteria.to_csv(args.output_dir / "summary_criteria_success.csv", index=False)

    pairwise_success = _pairwise_criteria_success(criteria_df)
    pairwise_success.to_csv(args.output_dir / "pairwise_criteria_success_holm.csv", index=False)

    pairwise_latency = _pairwise_latency(clean)
    pairwise_latency.to_csv(args.output_dir / "pairwise_latency_holm.csv", index=False)

    manifest = {
        "sources": [str(root) for root in roots],
        "per_root_counts": per_root_counts,
        "n_raw": int(len(annotated)),
        "n_valid": int(len(clean)),
        "excluded_frameworks": args.exclude_frameworks,
        "require_model_judge": args.require_model_judge,
        "primary_outcome": "criteria_success",
        "primary_pairwise_test": "chi2_or_fisher",
        "latency_pairwise_test": "mannwhitney",
        "holm_scope": "per_framework_over_three_none_vs_strategy_tests",
        "judge_mode_tables_in_primary_holm": False,
        "notes": [
            "No imputation is applied.",
            "Scaffold and runtime-failure rows are excluded by validity filters.",
            "Rows with empty raw_log_path are excluded.",
        ],
    }
    (args.output_dir / "reanalysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
