"""Metric aggregation helpers for experiment runs.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import sqrt
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from evaluation.mast_classifier import MASTCategory, MASTFailureMode
from evaluation.statistical_analysis import (
    compare_continuous,
    compare_mast_distributions,
    compare_success_rates,
    compute_descriptive_stats,
)
from frameworks.base_runner import TraceResult


def summarize_results(results: Sequence[TraceResult]) -> dict[str, Any]:
    """Compute aggregate metrics for a collection of runs."""

    if not results:
        return {
            "num_runs": 0,
            "success_rate": 0.0,
            "mean_latency_seconds": 0.0,
            "mean_total_tokens": 0.0,
            "mean_cost_usd": 0.0,
        }

    dataframe = pd.DataFrame(
        {
            "success": [result.success for result in results],
            "latency_seconds": [result.metrics.latency_seconds for result in results],
            "total_tokens": [result.metrics.total_tokens for result in results],
            "cost_usd": [result.metrics.cost_usd for result in results],
        }
    )
    return {
        "num_runs": int(len(results)),
        "success_rate": float(dataframe["success"].mean()),
        "mean_latency_seconds": float(dataframe["latency_seconds"].mean()),
        "median_latency_seconds": float(dataframe["latency_seconds"].median()),
        "p95_latency_seconds": float(np.percentile(dataframe["latency_seconds"], 95)),
        "mean_total_tokens": float(dataframe["total_tokens"].mean()),
        "mean_cost_usd": float(dataframe["cost_usd"].mean()),
    }


def results_to_frame(results: Sequence[TraceResult]) -> pd.DataFrame:
    """Convert trace results into a flat pandas DataFrame for analysis."""

    rows = []
    for run_index, result in enumerate(results):
        rows.append(
            {
                "run_index": run_index,
                "success": result.success,
                "final_output": result.final_output,
                "latency_seconds": result.metrics.latency_seconds,
                "prompt_tokens": result.metrics.prompt_tokens,
                "completion_tokens": result.metrics.completion_tokens,
                "total_tokens": result.metrics.total_tokens,
                "tool_calls": result.metrics.tool_calls,
                "steps_executed": result.metrics.steps_executed,
                "cost_usd": result.metrics.cost_usd,
                "raw_log_path": result.raw_log_path,
                "run_id": result.run_id,
            }
        )
    return pd.DataFrame(rows)


def summarize_conditions(
    results_df: pd.DataFrame,
    *,
    condition_column: str = "framework",
    success_column: str = "success",
    latency_column: str = "latency_seconds",
    cost_column: str = "cost_usd",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Calculate mean, std, and 95% confidence intervals per condition."""

    if results_df.empty:
        return pd.DataFrame()

    required_columns = {condition_column, success_column, latency_column, cost_column}
    missing_columns = required_columns.difference(results_df.columns)
    if missing_columns:
        raise KeyError(f"Results dataframe is missing required columns: {sorted(missing_columns)}")

    rows: list[dict[str, Any]] = []
    for condition_value, group in results_df.groupby(condition_column, dropna=False):
        row: dict[str, Any] = {condition_column: condition_value, "n": int(len(group))}
        row.update(_summarize_metric(group[success_column], prefix="success_rate", confidence=confidence))
        row.update(_summarize_metric(group[latency_column], prefix="latency_seconds", confidence=confidence))
        row.update(_summarize_metric(group[cost_column], prefix="token_cost_usd", confidence=confidence))
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(condition_column).reset_index(drop=True)


def mast_failure_distribution(
    results_df: pd.DataFrame,
    *,
    condition_column: str = "framework",
    failure_modes_column: str = "primary_failure_modes",
    level: str = "mode",
) -> pd.DataFrame:
    """Calculate absolute and percentage MAST distributions per condition."""

    if results_df.empty or failure_modes_column not in results_df.columns:
        return pd.DataFrame(columns=[condition_column, "mast_label", "absolute_count", "percentage_of_runs", "n_runs"])

    rows: list[dict[str, Any]] = []
    for condition_value, group in results_df.groupby(condition_column, dropna=False):
        n_runs = int(len(group))
        counts: dict[str, int] = {}
        for raw_modes in group[failure_modes_column]:
            labels = {_normalize_mast_label(mode, level=level) for mode in _extract_failure_modes(raw_modes)}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1

        for label, absolute_count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    condition_column: condition_value,
                    "mast_label": label,
                    "absolute_count": int(absolute_count),
                    "percentage_of_runs": float(absolute_count / n_runs * 100.0) if n_runs else 0.0,
                    "n_runs": n_runs,
                    "level": level,
                }
            )

    return pd.DataFrame(rows)


def pairwise_condition_comparisons(
    results_df: pd.DataFrame,
    *,
    metric_column: str,
    condition_column: str = "framework",
    test: str = "welch",
) -> pd.DataFrame:
    """Perform pairwise statistical comparisons between conditions."""

    if results_df.empty:
        return pd.DataFrame()

    if condition_column not in results_df.columns or metric_column not in results_df.columns:
        raise KeyError(f"Results dataframe must contain '{condition_column}' and '{metric_column}'.")

    grouped_values = {
        str(condition_value): _clean_numeric_series(group[metric_column])
        for condition_value, group in results_df.groupby(condition_column, dropna=False)
    }
    condition_names = list(grouped_values.keys())
    rows: list[dict[str, Any]] = []

    for condition_a, condition_b in combinations(condition_names, 2):
        sample_a = grouped_values[condition_a]
        sample_b = grouped_values[condition_b]
        if len(sample_a) == 0 or len(sample_b) == 0:
            continue

        if test == "welch":
            statistic, p_value = cast(
                tuple[float, float],
                scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False),
            )
            effect_size = cohens_d(sample_a, sample_b)
            effect_size_name = "cohens_d"
            test_name = "Welch t-test"
        elif test == "mannwhitney":
            statistic, p_value = cast(
                tuple[float, float],
                scipy_stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided"),
            )
            effect_size = cliffs_delta(sample_a, sample_b)
            effect_size_name = "cliffs_delta"
            test_name = "Mann-Whitney U"
        else:
            raise ValueError("test must be 'welch' or 'mannwhitney'")

        rows.append(
            {
                "metric": metric_column,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "test_name": test_name,
                "statistic": statistic,
                "p_value": p_value,
                "effect_size": float(effect_size),
                "effect_size_name": effect_size_name,
                "n_a": int(len(sample_a)),
                "n_b": int(len(sample_b)),
                "mean_a": float(np.mean(sample_a)),
                "mean_b": float(np.mean(sample_b)),
                "median_a": float(np.median(sample_a)),
                "median_b": float(np.median(sample_b)),
            }
        )

    return pd.DataFrame(rows)


def one_way_anova(
    results_df: pd.DataFrame,
    *,
    metric_column: str,
    condition_column: str = "framework",
) -> dict[str, Any]:
    """Run one-way ANOVA for a metric across multiple conditions."""

    if results_df.empty:
        return {
            "metric": metric_column,
            "condition_column": condition_column,
            "n_groups": 0,
            "n_observations": 0,
            "f_statistic": np.nan,
            "p_value": np.nan,
            "eta_squared": np.nan,
        }

    if condition_column not in results_df.columns or metric_column not in results_df.columns:
        raise KeyError(f"Results dataframe must contain '{condition_column}' and '{metric_column}'.")

    groups = [
        _clean_numeric_series(group[metric_column])
        for _, group in results_df.groupby(condition_column, dropna=False)
    ]
    groups = [group for group in groups if len(group) > 0]
    if len(groups) < 2:
        return {
            "metric": metric_column,
            "condition_column": condition_column,
            "n_groups": len(groups),
            "n_observations": int(sum(len(group) for group in groups)),
            "f_statistic": np.nan,
            "p_value": np.nan,
            "eta_squared": np.nan,
        }

    f_statistic, p_value = scipy_stats.f_oneway(*groups)
    all_values = np.concatenate(groups)
    grand_mean = float(np.mean(all_values))
    ss_between = sum(len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups)
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    eta_squared = float(ss_between / ss_total) if ss_total else np.nan

    return {
        "metric": metric_column,
        "condition_column": condition_column,
        "n_groups": len(groups),
        "n_observations": int(len(all_values)),
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "eta_squared": eta_squared,
    }


def build_statistical_report(
    results_df: pd.DataFrame,
    *,
    condition_column: str = "framework",
    failure_modes_column: str = "primary_failure_modes",
) -> dict[str, Any]:
    """Build all tables required for a thesis-grade statistical report."""

    summary = summarize_conditions(results_df, condition_column=condition_column)
    mast_modes = mast_failure_distribution(
        results_df,
        condition_column=condition_column,
        failure_modes_column=failure_modes_column,
        level="mode",
    )
    mast_categories = mast_failure_distribution(
        results_df,
        condition_column=condition_column,
        failure_modes_column=failure_modes_column,
        level="category",
    )

    comparisons = {
        metric: pairwise_condition_comparisons(results_df, metric_column=metric, condition_column=condition_column)
        for metric in ("success", "latency_seconds", "cost_usd")
        if metric in results_df.columns
    }
    anova = {
        metric: one_way_anova(results_df, metric_column=metric, condition_column=condition_column)
        for metric in ("latency_seconds", "cost_usd")
        if metric in results_df.columns
    }

    continuous_tables: dict[str, pd.DataFrame] = {}
    for metric in ("latency_seconds", "cost_usd", "token_cost"):
        if metric in results_df.columns:
            continuous_tables[metric] = compare_continuous(results_df, condition_column, metric)

    advanced_statistics = {
        "descriptive_bootstrap": compute_descriptive_stats(results_df, [condition_column]),
        "success_rate_tests": compare_success_rates(results_df, condition_column),
        "continuous_tests": continuous_tables,
        "mast_distribution_test": compare_mast_distributions(results_df, condition_column),
    }

    return {
        "summary": summary,
        "mast_modes": mast_modes,
        "mast_categories": mast_categories,
        "comparisons": comparisons,
        "anova": anova,
        "advanced_statistics": advanced_statistics,
    }


def cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""

    if len(sample_a) < 2 or len(sample_b) < 2:
        return 0.0
    variance_a = float(np.var(sample_a, ddof=1))
    variance_b = float(np.var(sample_b, ddof=1))
    pooled_denom = len(sample_a) + len(sample_b) - 2
    if pooled_denom <= 0:
        return 0.0
    pooled_variance = (((len(sample_a) - 1) * variance_a) + ((len(sample_b) - 1) * variance_b)) / pooled_denom
    if pooled_variance <= 0:
        return 0.0
    pooled_std = sqrt(pooled_variance)
    return float((float(np.mean(sample_a)) - float(np.mean(sample_b))) / pooled_std)


def cliffs_delta(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute Cliff's delta for two independent samples."""

    if len(sample_a) == 0 or len(sample_b) == 0:
        return 0.0
    greater = 0
    lower = 0
    for value_a in sample_a:
        greater += int(np.sum(value_a > sample_b))
        lower += int(np.sum(value_a < sample_b))
    return float((greater - lower) / (len(sample_a) * len(sample_b)))


def _summarize_metric(values: pd.Series, *, prefix: str, confidence: float) -> dict[str, float]:
    data = _clean_numeric_series(values)
    if len(data) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_ci_low": np.nan,
            f"{prefix}_ci_high": np.nan,
        }

    mean = float(np.mean(data))
    if len(data) == 1:
        std = 0.0
        ci_low = mean
        ci_high = mean
    else:
        std = float(np.std(data, ddof=1))
        standard_error = std / sqrt(len(data))
        critical_value = scipy_stats.t.ppf((1.0 + confidence) / 2.0, df=len(data) - 1)
        margin = float(critical_value * standard_error)
        ci_low = mean - margin
        ci_high = mean + margin

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_ci_low": ci_low,
        f"{prefix}_ci_high": ci_high,
    }


def _clean_numeric_series(values: pd.Series | Sequence[Any]) -> np.ndarray:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()


def _extract_failure_modes(raw_value: Any) -> list[str]:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []
    if isinstance(raw_value, MASTFailureMode):
        return [raw_value.value]
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import json

                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
        return [stripped]
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item.value if isinstance(item, MASTFailureMode) else item) for item in raw_value]
    return [str(raw_value)]


def _normalize_mast_label(mode: str, *, level: str) -> str:
    normalized = str(mode)
    if level == "mode":
        return normalized
    if normalized.startswith("1."):
        return MASTCategory.TASK_COMPLIANCE.value
    if normalized.startswith("2."):
        return MASTCategory.COORDINATION.value
    if normalized.startswith("3."):
        return MASTCategory.TERMINATION_AND_VERIFICATION.value
    return "Unknown"
