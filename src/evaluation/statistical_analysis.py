"""Statistical analysis helpers for thesis-grade MAS experiments.

This module was drafted with AI assistance and then reviewed and adapted
for methodological transparency in the thesis context.
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def add_significance_stars(p: float | None) -> str:
    """Map a p-value to common significance stars used in thesis tables."""

    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_descriptive_stats(df: pd.DataFrame, group_cols: str | list[str]) -> pd.DataFrame:
    """Compute descriptive statistics for numeric metrics by group.

    Statistics per metric and group:
    - n
    - mean
    - std
    - bootstrap 95% confidence interval for the mean
    - median
    - IQR

    Args:
        df: Result dataframe from the experiment harness.
        group_cols: One or multiple grouping columns such as framework/task.

    Returns:
        A tidy dataframe with one row per (group, metric).
    """

    if df.empty:
        return pd.DataFrame()

    groups = _as_list(group_cols)
    missing = [column for column in groups if column not in df.columns]
    if missing:
        raise KeyError(f"Missing grouping columns: {missing}")

    numeric_cols = [
        column
        for column in df.columns
        if column not in groups and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric_cols:
        return pd.DataFrame(columns=[*groups, "metric", "n", "mean", "std", "ci95_low", "ci95_high", "median", "iqr"])

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(groups, dropna=False)
    for group_key, group_df in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_mapping = dict(zip(groups, key_values, strict=False))
        for metric in numeric_cols:
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            ci_low, ci_high = _bootstrap_mean_ci(values, confidence=0.95, n_resamples=2000, random_state=42)
            q1, q3 = np.percentile(values, [25, 75])
            rows.append(
                {
                    **group_mapping,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "median": float(np.median(values)),
                    "iqr": float(q3 - q1),
                }
            )

    return pd.DataFrame(rows).sort_values([*groups, "metric"]).reset_index(drop=True)


def compare_success_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compare binary success rates between groups.

    Uses Fisher's exact test for 2x2 tables with small expected counts,
    otherwise Chi-square. The effect size is reported as odds ratio for
    pairwise comparisons.

    Args:
        df: Result dataframe from the experiment harness.
        group_col: Grouping column, e.g. ``framework`` or ``condition``.

    Returns:
        Dataframe with one row per pairwise group comparison and test outputs.
    """

    success_col = _resolve_success_column(df)
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    work_df = df[[group_col, success_col]].dropna().copy()
    work_df[success_col] = work_df[success_col].astype(bool)

    rows: list[dict[str, Any]] = []
    group_values = sorted(work_df[group_col].astype(str).unique())
    for group_a, group_b in combinations(group_values, 2):
        subset = work_df[work_df[group_col].astype(str).isin([group_a, group_b])].copy()
        subset[group_col] = subset[group_col].astype(str)
        contingency = pd.crosstab(subset[group_col], subset[success_col])
        contingency = contingency.reindex(index=[group_a, group_b], columns=[False, True], fill_value=0)

        a_fail, a_success = int(contingency.loc[group_a, False]), int(contingency.loc[group_a, True])
        b_fail, b_success = int(contingency.loc[group_b, False]), int(contingency.loc[group_b, True])

        expected = stats.contingency.expected_freq(contingency.values)
        use_fisher = contingency.shape == (2, 2) and np.any(expected < 5)

        if use_fisher:
            odds_ratio, p_value = stats.fisher_exact([[a_success, a_fail], [b_success, b_fail]], alternative="two-sided")
            statistic = np.nan
            test_name = "Fisher exact"
        else:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency.values, correction=False)
            odds_ratio = _odds_ratio_2x2(a_success, a_fail, b_success, b_fail)
            statistic = float(chi2)
            test_name = "Chi-square"

        rows.append(
            {
                "group_col": group_col,
                "group_a": group_a,
                "group_b": group_b,
                "test": test_name,
                "statistic": statistic,
                "p_value": float(p_value),
                "significance": add_significance_stars(float(p_value)),
                "odds_ratio": float(odds_ratio),
                "n_a": int(a_success + a_fail),
                "n_b": int(b_success + b_fail),
                "success_rate_a": float(a_success / (a_success + a_fail)) if (a_success + a_fail) else np.nan,
                "success_rate_b": float(b_success / (b_success + b_fail)) if (b_success + b_fail) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compare_continuous(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    """Compare a continuous metric between groups.

    Selection rule:
    - Welch t-test when both groups look approximately normal (Shapiro p>0.05)
      and each group has >= 8 observations.
    - Mann-Whitney U otherwise.

    Effect size:
    - Cohen's d for Welch t-test.
    - Cliff's delta for Mann-Whitney U.

    Args:
        df: Result dataframe from the experiment harness.
        group_col: Grouping column, e.g. framework.
        metric: Numeric metric column, e.g. latency_seconds or token_cost.

    Returns:
        Dataframe with one row per pairwise group comparison.
    """

    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")
    if metric not in df.columns:
        raise KeyError(f"Missing metric column: {metric}")

    work_df = df[[group_col, metric]].copy()
    work_df[metric] = pd.to_numeric(work_df[metric], errors="coerce")
    work_df = work_df.dropna()

    rows: list[dict[str, Any]] = []
    group_values = sorted(work_df[group_col].astype(str).unique())
    for group_a, group_b in combinations(group_values, 2):
        sample_a = work_df.loc[work_df[group_col].astype(str) == group_a, metric].to_numpy(dtype=float)
        sample_b = work_df.loc[work_df[group_col].astype(str) == group_b, metric].to_numpy(dtype=float)
        if sample_a.size == 0 or sample_b.size == 0:
            continue

        normal_a = _is_approximately_normal(sample_a)
        normal_b = _is_approximately_normal(sample_b)

        if normal_a and normal_b and sample_a.size >= 8 and sample_b.size >= 8:
            statistic, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
            effect_size = _cohens_d(sample_a, sample_b)
            effect_name = "cohens_d"
            test_name = "Welch t-test"
        else:
            statistic, p_value = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
            effect_size = _cliffs_delta(sample_a, sample_b)
            effect_name = "cliffs_delta"
            test_name = "Mann-Whitney U"

        rows.append(
            {
                "group_col": group_col,
                "metric": metric,
                "group_a": group_a,
                "group_b": group_b,
                "test": test_name,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significance": add_significance_stars(float(p_value)),
                "effect_size": float(effect_size),
                "effect_name": effect_name,
                "n_a": int(sample_a.size),
                "n_b": int(sample_b.size),
                "mean_a": float(np.mean(sample_a)),
                "mean_b": float(np.mean(sample_b)),
                "median_a": float(np.median(sample_a)),
                "median_b": float(np.median(sample_b)),
            }
        )

    return pd.DataFrame(rows)


def compare_mast_distributions(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compare MAST category distributions across groups using Chi-square.

    The function supports either a ``mast_category`` column or a
    ``primary_failure_modes`` column (from which category prefixes are inferred).

    Args:
        df: Result dataframe from the experiment harness.
        group_col: Grouping column, e.g. framework.

    Returns:
        Single-row dataframe with Chi-square statistic, p-value, and Cramer's V.
    """

    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    categories_df = _extract_mast_categories(df, group_col=group_col)
    if categories_df.empty:
        return pd.DataFrame(
            [
                {
                    "group_col": group_col,
                    "test": "Chi-square",
                    "chi2": np.nan,
                    "p_value": np.nan,
                    "dof": 0,
                    "n": 0,
                    "cramers_v": np.nan,
                    "significance": "",
                }
            ]
        )

    contingency = pd.crosstab(categories_df[group_col].astype(str), categories_df["mast_category"].astype(str))
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency.values)
    n_obs = float(contingency.values.sum())
    min_dim = min(contingency.shape)
    cramers_v = sqrt(chi2 / (n_obs * (min_dim - 1))) if n_obs > 0 and min_dim > 1 else np.nan

    return pd.DataFrame(
        [
            {
                "group_col": group_col,
                "test": "Chi-square",
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "n": int(n_obs),
                "cramers_v": float(cramers_v) if not np.isnan(cramers_v) else np.nan,
                "significance": add_significance_stars(float(p_value)),
            }
        ]
    )


def _as_list(group_cols: str | list[str]) -> list[str]:
    if isinstance(group_cols, str):
        return [group_cols]
    return list(group_cols)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float,
    n_resamples: int,
    random_state: int,
) -> tuple[float, float]:
    if values.size == 1:
        single = float(values[0])
        return single, single

    rng = np.random.default_rng(random_state)
    samples = rng.choice(values, size=(n_resamples, values.size), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def _resolve_success_column(df: pd.DataFrame) -> str:
    for candidate in ("success", "mast_task_successful", "task_successful"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No success column found. Expected one of: success, mast_task_successful, task_successful")


def _odds_ratio_2x2(a_success: int, a_fail: int, b_success: int, b_fail: int) -> float:
    # Haldane-Anscombe correction keeps OR finite when one cell is zero.
    a_s = a_success + 0.5
    a_f = a_fail + 0.5
    b_s = b_success + 0.5
    b_f = b_fail + 0.5
    return float((a_s * b_f) / (a_f * b_s))


def _is_approximately_normal(values: np.ndarray) -> bool:
    if values.size < 8:
        return False
    if np.allclose(values, values[0]):
        return False
    if values.size <= 5000:
        _, p_value = stats.shapiro(values)
        return bool(p_value > 0.05)
    _, p_value = stats.normaltest(values)
    return bool(p_value > 0.05)


def _cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if sample_a.size < 2 or sample_b.size < 2:
        return 0.0
    variance_a = float(np.var(sample_a, ddof=1))
    variance_b = float(np.var(sample_b, ddof=1))
    pooled_denom = sample_a.size + sample_b.size - 2
    if pooled_denom <= 0:
        return 0.0
    pooled_variance = (((sample_a.size - 1) * variance_a) + ((sample_b.size - 1) * variance_b)) / pooled_denom
    if pooled_variance <= 0:
        return 0.0
    return float((float(np.mean(sample_a)) - float(np.mean(sample_b))) / sqrt(pooled_variance))


def _cliffs_delta(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if sample_a.size == 0 or sample_b.size == 0:
        return 0.0
    diffs = np.subtract.outer(sample_a, sample_b)
    greater = float(np.sum(diffs > 0))
    lower = float(np.sum(diffs < 0))
    return float((greater - lower) / (sample_a.size * sample_b.size))


def _extract_mast_categories(df: pd.DataFrame, *, group_col: str) -> pd.DataFrame:
    if "mast_category" in df.columns:
        source_col = "mast_category"
    elif "primary_failure_modes" in df.columns:
        source_col = "primary_failure_modes"
    else:
        return pd.DataFrame(columns=[group_col, "mast_category"])

    rows: list[dict[str, str]] = []
    for _, row in df[[group_col, source_col]].dropna(subset=[source_col]).iterrows():
        group_value = str(row[group_col])
        raw_value = row[source_col]
        for category in _normalize_mast_value(raw_value, source_col=source_col):
            rows.append({group_col: group_value, "mast_category": category})
    return pd.DataFrame(rows)


def _normalize_mast_value(value: Any, *, source_col: str) -> list[str]:
    parsed: list[str] = []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                evaluated = literal_eval(text)
                raw_items = list(evaluated) if isinstance(evaluated, (list, tuple, set)) else [text]
            except (ValueError, SyntaxError):
                raw_items = [text]
        elif "," in text:
            raw_items = [part.strip() for part in text.split(",")]
        else:
            raw_items = [text]
    else:
        raw_items = [str(value)]

    for item in raw_items:
        token = str(item).strip()
        if not token:
            continue
        if source_col == "primary_failure_modes" and ":" in token:
            parsed.append(token.split(":", 1)[0].strip())
        else:
            parsed.append(token)
    return parsed


def _create_synthetic_example() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frameworks = ["langgraph", "autogen", "crewai", "metagpt"]
    rows: list[dict[str, Any]] = []
    for framework in frameworks:
        base_success = {
            "langgraph": 0.82,
            "autogen": 0.76,
            "crewai": 0.71,
            "metagpt": 0.68,
        }[framework]
        for run_idx in range(30):
            success = rng.random() < base_success
            rows.append(
                {
                    "framework": framework,
                    "task": "mixed",
                    "run_id": f"{framework}-{run_idx}",
                    "success": bool(success),
                    "latency_seconds": float(rng.normal(32.0, 6.0) + (0.0 if success else 3.0)),
                    "token_cost": float(max(0.01, rng.normal(0.38, 0.09))),
                    "mast_category": rng.choice(["Memory", "Planning", "ToolUse", "Coordination"]),
                }
            )
    return pd.DataFrame(rows)


def _example_cli() -> None:
    parser = argparse.ArgumentParser(description="Run a compact statistical summary for thesis reporting.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/example_results.csv"),
        help="CSV with experiment results. Falls back to synthetic data if missing.",
    )
    parser.add_argument("--group-col", default="framework", help="Primary grouping column.")
    parser.add_argument("--metric", default="latency_seconds", help="Continuous metric for pairwise tests.")
    args = parser.parse_args()

    if args.input_csv.exists():
        df = pd.read_csv(args.input_csv)
        print(f"Loaded results from {args.input_csv}")
    else:
        df = _create_synthetic_example()
        print("Input file not found. Using synthetic example data.")

    descriptive = compute_descriptive_stats(df, [args.group_col])
    success_comp = compare_success_rates(df, args.group_col)
    continuous_comp = compare_continuous(df, args.group_col, args.metric)
    mast_comp = compare_mast_distributions(df, args.group_col)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("\n=== Descriptive Statistics (excerpt) ===")
    print(descriptive.head(20).to_string(index=False))

    print("\n=== Success Rate Comparison ===")
    print(success_comp.to_string(index=False))

    print(f"\n=== Continuous Comparison ({args.metric}) ===")
    print(continuous_comp.to_string(index=False))

    print("\n=== MAST Distribution Comparison ===")
    print(mast_comp.to_string(index=False))


if __name__ == "__main__":
    _example_cli()