"""Utilities for validating automated MAST judgements against human annotations."""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.plots import build_batch_dataframe, discover_batch_directories, load_batch_artifacts


ANNOTATION_TEMPLATE_COLUMNS = [
    "framework",
    "benchmark",
    "run_index",
    "run_id",
    "raw_log_path",
    "success",
    "final_output",
    "judge_task_successful",
    "judge_primary_failure_modes",
    "judge_summary",
    "manual_task_successful",
    "manual_primary_failure_modes",
    "manual_summary",
    "reviewer_id",
    "adjudicated_task_successful",
    "adjudicated_primary_failure_modes",
    "adjudicated_summary",
    "notes",
]

SCAFFOLD_OR_FALLBACK_MARKERS = (
    "No external model configured. This LangGraph runner is operating in fallback mode.",
    "scaffold completed a reproducible placeholder run for the task.",
)


def load_runs_for_annotation(results_root: Path | str) -> pd.DataFrame:
    """Load run-level records with judge outputs for annotation workflows."""

    rows: list[pd.DataFrame] = []
    for batch_dir in discover_batch_directories(results_root):
        artifacts = load_batch_artifacts(batch_dir)
        batch_frame = build_batch_dataframe(artifacts)
        if batch_frame.empty:
            continue
        batch_frame = batch_frame.copy()
        batch_frame["batch_dir"] = str(batch_dir)
        batch_frame["judge_task_successful"] = batch_frame.get("mast_task_successful")
        batch_frame["judge_primary_failure_modes"] = batch_frame.get("primary_failure_modes")
        batch_frame["judge_summary"] = batch_frame.get("mast_summary")
        rows.append(batch_frame)

    if not rows:
        return pd.DataFrame(columns=ANNOTATION_TEMPLATE_COLUMNS)

    merged = pd.concat(rows, ignore_index=True)
    available_columns = [
        "framework",
        "benchmark",
        "run_index",
        "run_id",
        "raw_log_path",
        "success",
        "final_output",
        "judge_task_successful",
        "judge_primary_failure_modes",
        "judge_summary",
    ]
    template = merged[available_columns].copy()
    for column in ANNOTATION_TEMPLATE_COLUMNS:
        if column not in template.columns:
            template[column] = ""
    return template[ANNOTATION_TEMPLATE_COLUMNS]


def sample_annotation_rows(df: pd.DataFrame, sample_size: int | None, seed: int = 42) -> pd.DataFrame:
    """Sample rows for human annotation while preserving group coverage where possible."""

    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.reset_index(drop=True)

    work_df = df.reset_index(drop=True).copy()
    if not {"framework", "benchmark"}.issubset(work_df.columns):
        return work_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    grouped = list(work_df.groupby(["framework", "benchmark"], dropna=False, sort=True))
    if sample_size <= len(grouped):
        chosen_group_indices = work_df[["framework", "benchmark"]].drop_duplicates().sample(
            n=sample_size,
            random_state=seed,
        )
        sampled_indices = []
        for _, chosen_row in chosen_group_indices.iterrows():
            matching_group = work_df[
                (work_df["framework"] == chosen_row["framework"]) &
                (work_df["benchmark"] == chosen_row["benchmark"])
            ]
            sampled_indices.append(int(matching_group.sample(n=1, random_state=seed).index[0]))
        return work_df.loc[sorted(set(sampled_indices))].reset_index(drop=True)

    sampled_indices: list[int] = []
    for _, group_df in grouped:
        sampled_indices.append(int(group_df.sample(n=1, random_state=seed).index[0]))

    remaining = sample_size - len(sampled_indices)
    if remaining > 0:
        remaining_pool = work_df.drop(index=sampled_indices)
        if not remaining_pool.empty:
            extra = remaining_pool.sample(n=min(remaining, len(remaining_pool)), random_state=seed).index.tolist()
            sampled_indices.extend(int(index) for index in extra)

    sampled = work_df.loc[sorted(set(sampled_indices))]
    if len(sampled) < sample_size:
        top_up = work_df.drop(index=sampled.index).sample(n=sample_size - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, top_up], ignore_index=False)

    return sampled.reset_index(drop=True)


def filter_annotation_candidates(df: pd.DataFrame, *, real_model_only: bool = False) -> pd.DataFrame:
    """Filter annotation candidates, optionally excluding scaffold and fallback runs."""

    if not real_model_only or df.empty or "final_output" not in df.columns:
        return df.reset_index(drop=True)

    work_df = df.copy()
    if "success" in work_df.columns:
        work_df = work_df.loc[work_df["success"].fillna(False).astype(bool)]
    final_output = work_df["final_output"].fillna("").astype(str)
    mask = ~final_output.apply(_contains_scaffold_or_fallback_marker)
    return work_df.loc[mask].reset_index(drop=True)


def write_annotation_template(
    results_root: Path | str,
    output_csv: Path | str,
    *,
    sample_size: int | None = None,
    seed: int = 42,
    real_model_only: bool = False,
) -> pd.DataFrame:
    """Write an annotation template CSV from experiment results."""

    template = load_runs_for_annotation(results_root)
    template = filter_annotation_candidates(template, real_model_only=real_model_only)
    sampled = sample_annotation_rows(template, sample_size=sample_size, seed=seed)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)
    return sampled


def compute_cohens_kappa(left: list[bool], right: list[bool]) -> float:
    """Compute Cohen's kappa for binary labels without external dependencies."""

    if len(left) != len(right):
        raise ValueError("Both label sequences must have equal length.")
    if not left:
        return float("nan")

    observed = sum(a == b for a, b in zip(left, right, strict=False)) / len(left)
    left_true = sum(left) / len(left)
    left_false = 1.0 - left_true
    right_true = sum(right) / len(right)
    right_false = 1.0 - right_true
    expected = (left_true * right_true) + (left_false * right_false)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def parse_mode_set(value: Any) -> set[str]:
    """Normalize a mode list from CSV cells, Python literals, or sequences."""

    if value is None:
        return set()
    if isinstance(value, float) and np.isnan(value):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}

    text = str(value).strip()
    if not text:
        return set()

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return {str(item).strip() for item in parsed if str(item).strip()}

    return {part.strip() for part in text.split(";") if part.strip()}


def load_annotation_reference(annotations_csv: Path | str) -> pd.DataFrame:
    """Load human annotation CSV and resolve adjudicated labels when present."""

    df = pd.read_csv(annotations_csv)
    required = {"run_id", "manual_task_successful", "manual_primary_failure_modes"}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required annotation columns: {missing}")

    df = df.copy()
    task_source = "manual_task_successful"
    if "adjudicated_task_successful" in df.columns:
        df["reference_task_successful"] = df["adjudicated_task_successful"].where(
            df["adjudicated_task_successful"].notna(),
            df[task_source],
        )
    else:
        df["reference_task_successful"] = df[task_source]

    mode_source = "manual_primary_failure_modes"
    if "adjudicated_primary_failure_modes" in df.columns:
        df["reference_primary_failure_modes"] = df["adjudicated_primary_failure_modes"].where(
            df["adjudicated_primary_failure_modes"].notna() & (df["adjudicated_primary_failure_modes"].astype(str).str.strip() != ""),
            df[mode_source],
        )
    else:
        df["reference_primary_failure_modes"] = df[mode_source]

    return df


def build_judge_validation_report(annotations_df: pd.DataFrame) -> dict[str, Any]:
    """Compute human-vs-judge agreement statistics from annotation rows."""

    work_df = annotations_df.copy()
    work_df = work_df.dropna(subset=["judge_task_successful", "reference_task_successful", "judge_primary_failure_modes", "reference_primary_failure_modes"])
    if work_df.empty:
        raise ValueError("No comparable annotated rows available for judge validation.")

    judge_success = work_df["judge_task_successful"].astype(bool).tolist()
    reference_success = work_df["reference_task_successful"].astype(bool).tolist()

    judge_mode_sets = work_df["judge_primary_failure_modes"].apply(parse_mode_set)
    reference_mode_sets = work_df["reference_primary_failure_modes"].apply(parse_mode_set)

    exact_matches = [judge == reference for judge, reference in zip(judge_mode_sets, reference_mode_sets, strict=False)]
    jaccards = [_jaccard_score(judge, reference) for judge, reference in zip(judge_mode_sets, reference_mode_sets, strict=False)]

    all_modes = sorted(set().union(*judge_mode_sets.tolist(), *reference_mode_sets.tolist()))
    per_mode_rows: list[dict[str, Any]] = []
    for mode in all_modes:
        tp = fp = fn = tn = 0
        for judge_modes, reference_modes in zip(judge_mode_sets, reference_mode_sets, strict=False):
            judge_has = mode in judge_modes
            reference_has = mode in reference_modes
            if judge_has and reference_has:
                tp += 1
            elif judge_has and not reference_has:
                fp += 1
            elif not judge_has and reference_has:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        per_mode_rows.append(
            {
                "mode": mode,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support_reference": tp + fn,
                "support_judge": tp + fp,
            }
        )

    disagreements = work_df.loc[
        [not (success_match and mode_match) for success_match, mode_match in zip((work_df["judge_task_successful"].astype(bool) == work_df["reference_task_successful"].astype(bool)).tolist(), exact_matches, strict=False)],
        [
            "framework",
            "benchmark",
            "run_index",
            "run_id",
            "raw_log_path",
            "judge_task_successful",
            "reference_task_successful",
            "judge_primary_failure_modes",
            "reference_primary_failure_modes",
            "manual_summary",
            "adjudicated_summary",
            "notes",
        ],
    ].copy()

    summary = {
        "n_runs": int(len(work_df)),
        "task_success_raw_agreement": float(sum(a == b for a, b in zip(judge_success, reference_success, strict=False)) / len(work_df)),
        "task_success_cohens_kappa": float(compute_cohens_kappa(judge_success, reference_success)),
        "primary_mode_exact_match_rate": float(sum(exact_matches) / len(exact_matches)),
        "primary_mode_mean_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
        "macro_mode_f1": float(np.mean([row["f1"] for row in per_mode_rows])) if per_mode_rows else float("nan"),
    }

    return {
        "summary": summary,
        "per_mode": pd.DataFrame(per_mode_rows).sort_values(["support_reference", "mode"], ascending=[False, True]).reset_index(drop=True),
        "disagreements": disagreements.reset_index(drop=True),
    }


def write_judge_validation_report(report: dict[str, Any], output_dir: Path | str) -> dict[str, Path]:
    """Persist judge-validation tables and a markdown summary."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolved_output_dir / "judge_validation_summary.json"
    per_mode_path = resolved_output_dir / "judge_validation_per_mode.csv"
    disagreements_path = resolved_output_dir / "judge_validation_disagreements.csv"
    markdown_path = resolved_output_dir / "judge_validation_report.md"

    summary_path.write_text(pd.Series(report["summary"]).to_json(indent=2), encoding="utf-8")
    report["per_mode"].to_csv(per_mode_path, index=False)
    report["disagreements"].to_csv(disagreements_path, index=False)
    markdown_path.write_text(render_judge_validation_markdown(report), encoding="utf-8")
    return {
        "summary": summary_path,
        "per_mode": per_mode_path,
        "disagreements": disagreements_path,
        "markdown": markdown_path,
    }


def render_judge_validation_markdown(report: dict[str, Any]) -> str:
    """Render a short thesis-ready markdown summary."""

    summary = report["summary"]
    lines = [
        "# Judge Validation Report",
        "",
        f"- Runs compared: {summary['n_runs']}",
        f"- Task-success raw agreement: {summary['task_success_raw_agreement']:.3f}",
        f"- Task-success Cohen's kappa: {summary['task_success_cohens_kappa']:.3f}",
        f"- Primary-mode exact match rate: {summary['primary_mode_exact_match_rate']:.3f}",
        f"- Primary-mode mean Jaccard: {summary['primary_mode_mean_jaccard']:.3f}",
        f"- Macro per-mode F1: {summary['macro_mode_f1']:.3f}",
        "",
        "## Per-Mode Metrics",
        "",
        report["per_mode"].to_string(index=False) if not report["per_mode"].empty else "No per-mode rows available.",
        "",
        "## Disagreements",
        "",
        report["disagreements"].to_string(index=False) if not report["disagreements"].empty else "No disagreement rows.",
    ]
    return "\n".join(lines)


def _jaccard_score(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _contains_scaffold_or_fallback_marker(text: str) -> bool:
    normalized = text.strip()
    return any(marker in normalized for marker in SCAFFOLD_OR_FALLBACK_MARKERS)