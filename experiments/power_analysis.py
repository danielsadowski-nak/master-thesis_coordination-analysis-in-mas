"""A-priori and post-hoc power analysis for thesis experiment planning.

Generated with GitHub Copilot assistance - reviewed and adapted by author.

The script focuses on two families of outcomes:
1) Binary success rates (difference in proportions)
2) Continuous outcomes like latency/token cost (Cohen's d)

It prints thesis-ready, human-readable interpretations to justify run counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Power analysis for MAS experiment planning.")
    parser.add_argument(
        "--pilot-csv",
        type=Path,
        default=None,
        help="Optional pilot results CSV (e.g., results/.../all_runs.csv).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Target power for a-priori analyses.",
    )
    parser.add_argument(
        "--posthoc-n",
        type=int,
        default=40,
        help="Per-group N for post-hoc power calculations.",
    )
    return parser.parse_args()


def estimate_pilot_success_rates(pilot_df: pd.DataFrame) -> tuple[float, float] | None:
    """Estimate lower/higher success rates from pilot data if possible."""

    for success_col in ("success", "mast_task_successful", "task_successful"):
        if success_col in pilot_df.columns:
            break
    else:
        return None

    if "framework" in pilot_df.columns:
        grouped = pilot_df.groupby("framework", dropna=False)[success_col].mean().sort_values()
        if grouped.size >= 2:
            return float(grouped.iloc[0]), float(grouped.iloc[-1])

    success_values = pd.to_numeric(pilot_df[success_col], errors="coerce").dropna()
    if success_values.empty:
        return None

    center = float(success_values.mean())
    return max(0.01, center - 0.08), min(0.99, center + 0.08)


def estimate_pilot_effect_size_d(pilot_df: pd.DataFrame, metric: str) -> float | None:
    """Estimate a pilot-based Cohen's d using min/max framework means."""

    if metric not in pilot_df.columns:
        return None
    metric_values = pd.to_numeric(pilot_df[metric], errors="coerce")
    if metric_values.dropna().empty:
        return None

    if "framework" not in pilot_df.columns:
        std_all = float(metric_values.std(ddof=1))
        return None if std_all <= 0 else 0.5

    grouped = (
        pd.DataFrame({"framework": pilot_df["framework"], metric: metric_values})
        .dropna()
        .groupby("framework", dropna=False)[metric]
    )
    if grouped.ngroups < 2:
        return None

    means = grouped.mean().sort_values()
    stds = grouped.std(ddof=1).replace(0, np.nan)
    low_name = means.index[0]
    high_name = means.index[-1]
    pooled_std = np.nanmean([stds.get(low_name, np.nan), stds.get(high_name, np.nan)])
    if np.isnan(pooled_std) or pooled_std <= 0:
        return None
    d_est = abs(float(means.iloc[-1] - means.iloc[0])) / float(pooled_std)
    return float(d_est)


def required_n_for_success_difference(
    p1: float,
    p2: float,
    *,
    alpha: float,
    power: float,
) -> float:
    """Solve required per-group N for a two-sided difference in proportions."""

    analysis = NormalIndPower()
    effect_size = proportion_effectsize(p1, p2)
    return float(analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0, alternative="two-sided"))


def posthoc_power_success(
    p1: float,
    p2: float,
    *,
    alpha: float,
    n_per_group: int,
) -> float:
    """Compute post-hoc power for a proportion difference with fixed N."""

    analysis = NormalIndPower()
    effect_size = proportion_effectsize(p1, p2)
    return float(analysis.power(effect_size=effect_size, nobs1=n_per_group, alpha=alpha, ratio=1.0, alternative="two-sided"))


def required_n_for_cohens_d(d: float, *, alpha: float, power: float) -> float:
    """Solve required per-group N for a two-sample t-test with effect size d."""

    analysis = TTestIndPower()
    return float(analysis.solve_power(effect_size=d, power=power, alpha=alpha, ratio=1.0, alternative="two-sided"))


def posthoc_power_cohens_d(d: float, *, alpha: float, n_per_group: int) -> float:
    """Compute post-hoc power for a two-sample t-test with fixed per-group N."""

    analysis = TTestIndPower()
    return float(analysis.power(effect_size=d, nobs1=n_per_group, alpha=alpha, ratio=1.0, alternative="two-sided"))


def main() -> None:
    args = parse_args()

    # Requested thesis scenario: detect around 15-20 percentage points.
    default_success_pairs = [(0.65, 0.80), (0.65, 0.82), (0.65, 0.85)]
    pilot_df: pd.DataFrame | None = None

    if args.pilot_csv is not None and args.pilot_csv.exists():
        pilot_df = pd.read_csv(args.pilot_csv)
        pilot_pair = estimate_pilot_success_rates(pilot_df)
        if pilot_pair is not None:
            p_low, p_high = pilot_pair
            default_success_pairs = [(p_low, min(0.999, p_low + 0.15)), (p_low, min(0.999, p_low + 0.20)), (p_low, p_high)]

    print("=== A-priori Power Analysis for MAS Experiments ===")
    print(f"alpha = {args.alpha:.3f}, target power = {args.power:.2f}")
    print("")

    print("1) Success rate differences (two-sided test for proportions)")
    success_rows = []
    for p1, p2 in default_success_pairs:
        n_req = required_n_for_success_difference(p1, p2, alpha=args.alpha, power=args.power)
        p_post = posthoc_power_success(p1, p2, alpha=args.alpha, n_per_group=args.posthoc_n)
        success_rows.append(
            {
                "p1": p1,
                "p2": p2,
                "delta_pp": (p2 - p1) * 100.0,
                "required_n_per_group": float(np.ceil(n_req)),
                "posthoc_power_n40": p_post,
            }
        )
    success_df = pd.DataFrame(success_rows)
    print(success_df.to_string(index=False, formatters={"p1": "{:.3f}".format, "p2": "{:.3f}".format, "delta_pp": "{:.1f}".format, "required_n_per_group": "{:.0f}".format, "posthoc_power_n40": "{:.3f}".format}))
    print("")

    print("2) Continuous outcomes (latency/token cost) with medium effect d=0.5")
    d_default = 0.5
    d_latency_pilot = None
    d_cost_pilot = None
    if pilot_df is not None:
        d_latency_pilot = estimate_pilot_effect_size_d(pilot_df, "latency_seconds")
        d_cost_pilot = estimate_pilot_effect_size_d(pilot_df, "cost_usd") or estimate_pilot_effect_size_d(pilot_df, "token_cost")

    n_req_d05 = required_n_for_cohens_d(d_default, alpha=args.alpha, power=args.power)
    posthoc_d05 = posthoc_power_cohens_d(d_default, alpha=args.alpha, n_per_group=args.posthoc_n)
    print(f"Required N per group for d=0.5: {np.ceil(n_req_d05):.0f}")
    print(f"Post-hoc power at N={args.posthoc_n} per group for d=0.5: {posthoc_d05:.3f}")
    print("")

    if d_latency_pilot is not None or d_cost_pilot is not None:
        print("Pilot-based optional effect size estimates:")
        if d_latency_pilot is not None:
            n_req = required_n_for_cohens_d(d_latency_pilot, alpha=args.alpha, power=args.power)
            p_post = posthoc_power_cohens_d(d_latency_pilot, alpha=args.alpha, n_per_group=args.posthoc_n)
            print(f"- latency_seconds: d~{d_latency_pilot:.3f}, required N~{np.ceil(n_req):.0f}, post-hoc power(N={args.posthoc_n})={p_post:.3f}")
        if d_cost_pilot is not None:
            n_req = required_n_for_cohens_d(d_cost_pilot, alpha=args.alpha, power=args.power)
            p_post = posthoc_power_cohens_d(d_cost_pilot, alpha=args.alpha, n_per_group=args.posthoc_n)
            print(f"- cost metric: d~{d_cost_pilot:.3f}, required N~{np.ceil(n_req):.0f}, post-hoc power(N={args.posthoc_n})={p_post:.3f}")
        print("")

    min_req_success = int(np.ceil(success_df["required_n_per_group"].max()))
    min_req_continuous = int(np.ceil(n_req_d05))
    print("=== Thesis Interpretation (copy-ready) ===")
    print(
        "For binary task success, detecting differences of 15-20 percentage points "
        f"requires approximately N={min_req_success} per group under alpha={args.alpha:.2f}, power={args.power:.2f}."
    )
    print(
        f"For medium continuous effects (Cohen's d=0.5), required N is about {min_req_continuous} per group."
    )
    print(
        f"At N={args.posthoc_n} per group, post-hoc power for binary effects in the 15-20 pp range is typically below 0.80, "
        "while continuous medium effects are closer but often still below 0.80."
    )
    print(
        "Therefore, N=30-50 per condition is best framed as a compute-constrained, exploratory design with transparent limitations in statistical power. "
        "If binary success-rate differences (15-20 pp) are a primary endpoint, larger N or pooled/multi-task analyses are recommended."
    )


if __name__ == "__main__":
    main()