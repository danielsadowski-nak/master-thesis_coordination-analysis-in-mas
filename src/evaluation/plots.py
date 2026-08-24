"""Visualization helpers for thesis-grade experiment reporting.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.metrics import (
    build_statistical_report,
    mast_failure_distribution,
    summarize_conditions,
)


def add_ci_error_bars_95(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    order: list[str] | None = None,
    n_resamples: int = 2000,
    random_state: int = 42,
    color: str = "#111111",
    marker: str = "D",
    capsize: int = 4,
) -> plt.Axes:
    """Overlay mean points with bootstrap 95% CI error bars on categorical plots.

    This helper works with existing barplots or boxplots that use a categorical
    x-axis and a numeric y-axis.
    """

    if data.empty or x not in data.columns or y not in data.columns:
        return ax

    work = data[[x, y]].dropna().copy()
    if work.empty:
        return ax
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[y])
    if work.empty:
        return ax

    if order is None:
        order = [str(value) for value in sorted(work[x].astype(str).unique())]

    rng = np.random.default_rng(random_state)
    means: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []
    for category in order:
        values = work.loc[work[x].astype(str) == str(category), y].to_numpy(dtype=float)
        if values.size == 0:
            means.append(np.nan)
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            continue
        mean_value = float(np.mean(values))
        if values.size == 1:
            lower, upper = mean_value, mean_value
        else:
            samples = rng.choice(values, size=(n_resamples, values.size), replace=True)
            sample_means = samples.mean(axis=1)
            lower = float(np.percentile(sample_means, 2.5))
            upper = float(np.percentile(sample_means, 97.5))
        means.append(mean_value)
        ci_low.append(lower)
        ci_high.append(upper)

    x_positions = np.arange(len(order))
    means_arr = np.array(means, dtype=float)
    low_arr = np.array(ci_low, dtype=float)
    high_arr = np.array(ci_high, dtype=float)
    valid = ~np.isnan(means_arr)
    if not np.any(valid):
        return ax

    yerr_low = means_arr[valid] - low_arr[valid]
    yerr_high = high_arr[valid] - means_arr[valid]
    ax.errorbar(
        x_positions[valid],
        means_arr[valid],
        yerr=[yerr_low, yerr_high],
        fmt=marker,
        color=color,
        ecolor=color,
        markersize=4,
        capsize=capsize,
        linewidth=1,
        zorder=5,
    )
    return ax


@dataclass(slots=True)
class ExperimentBatchArtifacts:
    """Loaded files for one experiment batch."""

    batch_dir: Path
    summary: dict[str, Any]
    records: pd.DataFrame
    run_payloads: list[dict[str, Any]]


def discover_batch_directories(root_dir: Path | str) -> list[Path]:
    """Find all experiment batch directories containing summary and record files."""

    resolved_root = Path(root_dir)
    if not resolved_root.exists():
        return []

    candidates = set()
    if (resolved_root / "summary.json").exists() and (resolved_root / "records.csv").exists():
        candidates.add(resolved_root)
    for summary_path in resolved_root.rglob("summary.json"):
        batch_dir = summary_path.parent
        if (batch_dir / "records.csv").exists():
            candidates.add(batch_dir)
    return sorted(candidates)


def build_batch_dataframe(artifacts: ExperimentBatchArtifacts) -> pd.DataFrame:
    """Join per-run records with MAST judgement payloads for downstream analysis."""

    records = artifacts.records.copy()
    if records.empty:
        return records

    records = records.copy()
    if "run_index" not in records.columns:
        records = records.reset_index().rename(columns={"index": "run_index"})

    payload_rows: list[dict[str, Any]] = []
    for payload in artifacts.run_payloads:
        judgement = payload.get("judgement", {}) if isinstance(payload, dict) else {}
        payload_rows.append(
            {
                "run_index": payload.get("run_index") if isinstance(payload, dict) else None,
                "mast_task_successful": judgement.get("task_successful"),
                "mast_summary": judgement.get("summary"),
                "primary_failure_modes": judgement.get("primary_failure_modes", []),
                "runtime_mode": payload.get("runtime_mode") if isinstance(payload, dict) else None,
                "is_valid_analytical": payload.get("is_valid_analytical") if isinstance(payload, dict) else None,
                "criteria_success": payload.get("criteria_success") if isinstance(payload, dict) else None,
                "criteria_matched": payload.get("criteria_matched") if isinstance(payload, dict) else None,
                "criteria_total": payload.get("criteria_total") if isinstance(payload, dict) else None,
                "criteria_scorer": payload.get("criteria_scorer") if isinstance(payload, dict) else None,
                "task_id": payload.get("task_id") if isinstance(payload, dict) else None,
                "is_scaffold": payload.get("is_scaffold") if isinstance(payload, dict) else None,
                "is_heuristic_judge": payload.get("is_heuristic_judge") if isinstance(payload, dict) else None,
                "is_runtime_failure": payload.get("is_runtime_failure") if isinstance(payload, dict) else None,
                "validity_reason": payload.get("validity_reason") if isinstance(payload, dict) else None,
                "mast_judge_enabled": payload.get("mast_judge_enabled") if isinstance(payload, dict) else None,
                "mast_judge_runtime": payload.get("mast_judge_runtime") if isinstance(payload, dict) else None,
            }
        )

    if payload_rows:
        payload_frame = pd.DataFrame(payload_rows)
        if payload_frame["run_index"].notna().any() and records["run_index"].notna().any():
            records = records.merge(payload_frame, on="run_index", how="left")
        else:
            payload_frame = payload_frame.reset_index(drop=True)
            records = records.reset_index(drop=True)
            records = pd.concat([records, payload_frame.drop(columns=["run_index"])] , axis=1)

    records["framework"] = artifacts.summary.get("framework", artifacts.batch_dir.name)
    records["benchmark"] = artifacts.summary.get("benchmark", artifacts.batch_dir.parent.name)
    return records


def load_experiment_dataframe(root_dir: Path | str) -> pd.DataFrame:
    """Load and combine all available experiment batches into one dataframe."""

    frames: list[pd.DataFrame] = []
    for batch_dir in discover_batch_directories(root_dir):
        artifacts = load_batch_artifacts(batch_dir)
        batch_frame = build_batch_dataframe(artifacts)
        if not batch_frame.empty:
            frames.append(batch_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_batch_artifacts(batch_dir: Path | str) -> ExperimentBatchArtifacts:
    """Load summary, run records, and per-run payloads from a batch directory."""

    resolved_batch_dir = Path(batch_dir)
    summary_path = resolved_batch_dir / "summary.json"
    records_path = resolved_batch_dir / "records.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    records = pd.read_csv(records_path) if records_path.exists() else pd.DataFrame()

    run_payloads: list[dict[str, Any]] = []
    for run_path in sorted(resolved_batch_dir.glob("run_*.json")):
        try:
            run_payloads.append(json.loads(run_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue

    return ExperimentBatchArtifacts(
        batch_dir=resolved_batch_dir,
        summary=summary,
        records=records,
        run_payloads=run_payloads,
    )


def render_experiment_reports(batch_dir: Path | str, output_dir: Path | str | None = None) -> list[Path]:
    """Render a set of standard plots for a batch directory."""

    artifacts = load_batch_artifacts(batch_dir)
    resolved_output_dir = Path(output_dir) if output_dir is not None else artifacts.batch_dir / "figures"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = [
        plot_success_rate(artifacts, resolved_output_dir),
        plot_latency_distribution(artifacts, resolved_output_dir),
        plot_mast_failure_distribution(artifacts, resolved_output_dir),
    ]
    return [path for path in generated_files if path is not None]


def plot_success_rate(artifacts: ExperimentBatchArtifacts, output_dir: Path) -> Path:
    """Plot the binary success rate for the batch."""

    output_path = output_dir / "success_rate.png"
    success_rate = float(artifacts.summary.get("success_rate", 0.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Success rate"], [success_rate], color="#2a9d8f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.set_title("Experiment Success Rate")
    ax.text(0, min(success_rate + 0.03, 0.98), f"{success_rate:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_latency_distribution(artifacts: ExperimentBatchArtifacts, output_dir: Path) -> Path:
    """Plot the latency distribution for a batch."""

    output_path = output_dir / "latency_distribution.png"
    if artifacts.records.empty or "latency_seconds" not in artifacts.records.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No latency data available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(artifacts.records["latency_seconds"], kde=True, ax=ax, color="#264653")
    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("Run count")
    ax.set_title("Latency Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_mast_failure_distribution(artifacts: ExperimentBatchArtifacts, output_dir: Path) -> Path:
    """Plot a bar chart for primary MAST failure modes in the batch."""

    output_path = output_dir / "mast_failure_distribution.png"
    counter: Counter[str] = Counter()

    for payload in artifacts.run_payloads:
        judgement = payload.get("judgement", {})
        modes = judgement.get("primary_failure_modes", []) or []
        for mode in modes:
            counter[str(mode)] += 1

    if not counter:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No MAST failure labels available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    distribution = pd.DataFrame({"failure_mode": list(counter.keys()), "count": list(counter.values())})
    distribution = distribution.sort_values("count", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=distribution, x="count", y="failure_mode", ax=ax, color="#264653")
    ax.set_xlabel("Count")
    ax.set_ylabel("MAST failure mode")
    ax.set_title("Primary MAST Failure Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def export_batch_summary(artifacts: ExperimentBatchArtifacts, output_dir: Path) -> Path:
    """Export a compact CSV summary for downstream analysis."""

    output_path = output_dir / "batch_summary.csv"
    summary_frame = pd.DataFrame(
        [
            {
                "framework": artifacts.summary.get("framework"),
                "benchmark": artifacts.summary.get("benchmark"),
                "num_runs": artifacts.summary.get("num_runs", 0),
                "success_rate": artifacts.summary.get("success_rate", 0.0),
                "mean_latency_seconds": artifacts.summary.get("mean_latency_seconds", 0.0),
                "mean_total_tokens": artifacts.summary.get("mean_total_tokens", 0.0),
                "mean_cost_usd": artifacts.summary.get("mean_cost_usd", 0.0),
            }
        ]
    )
    summary_frame.to_csv(output_path, index=False)
    return output_path


def render_thesis_report(
    results_df: pd.DataFrame,
    output_dir: Path | str,
    *,
    condition_column: str = "framework",
    failure_modes_column: str = "primary_failure_modes",
) -> dict[str, list[Path]]:
    """Render thesis-ready comparative plots for multiple frameworks."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    report = build_statistical_report(
        results_df,
        condition_column=condition_column,
        failure_modes_column=failure_modes_column,
    )
    summary_df = report["summary"]
    mast_categories_df = report["mast_categories"]

    generated: dict[str, list[Path]] = {}
    generated["success_rate"] = _plot_grouped_success_rate(summary_df, resolved_output_dir, condition_column=condition_column)
    generated["mast_categories"] = _plot_grouped_mast_categories(
        mast_categories_df,
        resolved_output_dir,
        condition_column=condition_column,
    )
    generated["latency"] = _plot_metric_boxplot(
        results_df,
        resolved_output_dir,
        metric_column="latency_seconds",
        condition_column=condition_column,
        title="Latency Distribution Across Frameworks",
        ylabel="Latency (seconds)",
        stem="latency_distribution",
    )
    generated["cost"] = _plot_metric_boxplot(
        results_df,
        resolved_output_dir,
        metric_column="cost_usd",
        condition_column=condition_column,
        title="Token Cost Distribution Across Frameworks",
        ylabel="Token cost (USD)",
        stem="cost_distribution",
    )
    return generated


def _save_figure_bundle(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _add_caption(fig: plt.Figure, caption: str) -> None:
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)


def _plot_grouped_success_rate(
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    condition_column: str,
) -> list[Path]:
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No summary statistics available", ha="center", va="center")
        ax.axis("off")
        return _save_figure_bundle(fig, output_dir, "success_rate_grouped")

    data = summary_df.copy()
    data = data.sort_values(condition_column)
    x_positions = np.arange(len(data))
    values = data["success_rate_mean"].to_numpy(dtype=float)
    yerr_low = values - data["success_rate_ci_low"].to_numpy(dtype=float)
    yerr_high = data["success_rate_ci_high"].to_numpy(dtype=float) - values
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x_positions, values, color="#2a9d8f", width=0.65)
    ax.errorbar(x_positions, values, yerr=[yerr_low, yerr_high], fmt="none", ecolor="#222222", capsize=4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(data[condition_column], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Framework")
    ax.set_title("Success Rate by Framework")
    for x_pos, value in zip(x_positions, values, strict=False):
        ax.text(x_pos, min(value + 0.03, 0.98), f"{value:.1%}", ha="center", va="bottom", fontsize=10)
    _add_caption(fig, "Bars show the mean success rate with 95% confidence intervals.")
    return _save_figure_bundle(fig, output_dir, "success_rate_grouped")


def _plot_grouped_mast_categories(
    mast_df: pd.DataFrame,
    output_dir: Path,
    *,
    condition_column: str,
) -> list[Path]:
    if mast_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No MAST category data available", ha="center", va="center")
        ax.axis("off")
        return _save_figure_bundle(fig, output_dir, "mast_categories_grouped")

    pivot = mast_df.pivot_table(index=condition_column, columns="mast_label", values="percentage_of_runs", fill_value=0.0)
    pivot = pivot.sort_index()
    categories = list(pivot.columns)
    conditions = list(pivot.index)
    x_positions = np.arange(len(conditions))
    width = 0.8 / max(len(categories), 1)
    palette = sns.color_palette("colorblind", n_colors=max(len(categories), 1))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, category in enumerate(categories):
        offsets = x_positions + (index - (len(categories) - 1) / 2) * width
        values = pivot[category].to_numpy(dtype=float)
        ax.bar(offsets, values, width=width, label=category, color=palette[index])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Runs with category (%)")
    ax.set_xlabel("Framework")
    ax.set_title("MAST Category Distribution by Framework")
    ax.legend(title="MAST category", bbox_to_anchor=(1.02, 1), loc="upper left")
    _add_caption(fig, "Percentages denote the share of runs exhibiting at least one mode in the category.")
    return _save_figure_bundle(fig, output_dir, "mast_categories_grouped")


def _plot_metric_boxplot(
    results_df: pd.DataFrame,
    output_dir: Path,
    *,
    metric_column: str,
    condition_column: str,
    title: str,
    ylabel: str,
    stem: str,
) -> list[Path]:
    if results_df.empty or metric_column not in results_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"No {metric_column} data available", ha="center", va="center")
        ax.axis("off")
        return _save_figure_bundle(fig, output_dir, stem)

    data = results_df[[condition_column, metric_column]].dropna().copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"No {metric_column} data available", ha="center", va="center")
        ax.axis("off")
        return _save_figure_bundle(fig, output_dir, stem)

    order = sorted(data[condition_column].astype(str).unique())
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.boxplot(data=data, x=condition_column, y=metric_column, order=order, ax=ax, color="#457b9d", showfliers=False)
    sns.stripplot(data=data, x=condition_column, y=metric_column, order=order, ax=ax, color="#111111", size=3, alpha=0.5)
    add_ci_error_bars_95(ax, data, x=condition_column, y=metric_column, order=order)
    ax.set_xlabel("Framework")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    _add_caption(fig, "Boxplots show the median/IQR; diamonds and whiskers indicate mean and bootstrap 95% CI.")
    return _save_figure_bundle(fig, output_dir, stem)
