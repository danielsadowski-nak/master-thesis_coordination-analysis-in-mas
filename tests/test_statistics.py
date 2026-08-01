"""Tests for thesis-style statistical analysis helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.metrics import (
    build_statistical_report,
    mast_failure_distribution,
    one_way_anova,
    pairwise_condition_comparisons,
    summarize_conditions,
)
from evaluation.plots import render_thesis_report


def _sample_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "framework": "autogen",
                "success": True,
                "latency_seconds": 1.0,
                "cost_usd": 0.01,
                "primary_failure_modes": ["1.1 Disobey Task Specification"],
            },
            {
                "framework": "autogen",
                "success": False,
                "latency_seconds": 1.4,
                "cost_usd": 0.015,
                "primary_failure_modes": ["2.5 Ignored Other Agent's Input"],
            },
            {
                "framework": "crewai",
                "success": True,
                "latency_seconds": 2.0,
                "cost_usd": 0.02,
                "primary_failure_modes": ["1.1 Disobey Task Specification", "3.2 Weak Verification"],
            },
            {
                "framework": "crewai",
                "success": True,
                "latency_seconds": 2.4,
                "cost_usd": 0.018,
                "primary_failure_modes": ["3.1 Premature Termination"],
            },
            {
                "framework": "metagpt",
                "success": False,
                "latency_seconds": 3.0,
                "cost_usd": 0.03,
                "primary_failure_modes": ["2.2 Fail to Ask for Clarification"],
            },
            {
                "framework": "metagpt",
                "success": True,
                "latency_seconds": 3.6,
                "cost_usd": 0.025,
                "primary_failure_modes": ["3.3 No or Incorrect Verification"],
            },
        ]
    )


def test_summarize_conditions_and_mast_distribution() -> None:
    results_df = _sample_results_frame()

    summary_df = summarize_conditions(results_df)
    mast_modes_df = mast_failure_distribution(results_df, level="mode")
    mast_categories_df = mast_failure_distribution(results_df, level="category")

    assert set(summary_df["framework"]) == {"autogen", "crewai", "metagpt"}
    assert {"success_rate_mean", "latency_seconds_ci_low", "token_cost_usd_ci_high"}.issubset(summary_df.columns)
    assert mast_modes_df["absolute_count"].sum() == 7
    assert mast_categories_df["mast_label"].isin(["1. Task Compliance", "2. Coordination", "3. Termination and Verification"]).any()


def test_pairwise_comparisons_and_anova() -> None:
    results_df = _sample_results_frame()

    comparisons = pairwise_condition_comparisons(results_df, metric_column="latency_seconds")
    anova_result = one_way_anova(results_df, metric_column="latency_seconds")

    assert len(comparisons) == 3
    assert set(comparisons["test_name"]) == {"Welch t-test"}
    assert set(comparisons["effect_size_name"]) == {"cohens_d"}
    assert anova_result["n_groups"] == 3
    assert anova_result["f_statistic"] == anova_result["f_statistic"]


def test_render_thesis_report_creates_pdf_and_png(tmp_path: Path) -> None:
    results_df = _sample_results_frame()
    output_dir = tmp_path / "report"

    report = build_statistical_report(results_df)
    generated = render_thesis_report(results_df, output_dir)

    assert report["summary"].shape[0] == 3
    assert set(generated.keys()) == {"success_rate", "mast_categories", "latency", "cost"}
    assert all(path.exists() for paths in generated.values() for path in paths)
    assert (output_dir / "success_rate_grouped.png").exists()
    assert (output_dir / "success_rate_grouped.pdf").exists()


def test_one_way_anova_skips_underpowered_groups() -> None:
    results_df = pd.DataFrame(
        [
            {"framework": "langgraph", "latency_seconds": 1.0},
            {"framework": "autogen", "latency_seconds": 2.0},
        ]
    )

    result = one_way_anova(results_df, metric_column="latency_seconds")

    assert result["n_groups"] == 2
    assert result["f_statistic"] != result["f_statistic"]
