"""Classify analytical validity of experiment runs for thesis-grade reporting.

Scaffold/fallback runs and heuristic-only judgements must not enter primary
inference. This module provides deterministic, rule-based filters that can be
applied to already collected artifacts without re-running experiments.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


SCAFFOLD_MARKERS = (
    "scaffold completed a reproducible placeholder run for the task",
    "framework adapter scaffold",
    "this adapter currently runs in scaffold mode",
    "no external model configured",
    "operating in fallback mode",
    "heuristic fallback only",
)

HEURISTIC_JUDGE_MARKERS = (
    "heuristic fallback only",
    "configure a judge model",
    "keyword match in trace text",
)


def classify_run_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Return validity flags for a single run-level record."""

    final_output = str(row.get("final_output") or "")
    mast_summary = str(row.get("mast_summary") or row.get("judge_summary") or "")
    latency = _as_float(row.get("latency_seconds"))
    framework = str(row.get("framework") or "").lower()

    is_scaffold = _contains_any(final_output, SCAFFOLD_MARKERS) or _contains_any(
        mast_summary, SCAFFOLD_MARKERS
    )
    # Extremely low latency with perfect success is a strong scaffold signature,
    # especially for MetaGPT which falls back when the package is missing.
    if framework == "metagpt" and latency is not None and latency < 0.05:
        is_scaffold = True
    if latency is not None and latency < 0.02 and _contains_any(final_output.lower(), ("scaffold", "placeholder")):
        is_scaffold = True

    is_heuristic_judge = _contains_any(mast_summary, HEURISTIC_JUDGE_MARKERS)
    is_runtime_failure = bool(row.get("runtime_failure", False))
    is_valid_analytical = (not is_scaffold) and (not is_runtime_failure)

    return {
        "is_scaffold": bool(is_scaffold),
        "is_heuristic_judge": bool(is_heuristic_judge),
        "is_runtime_failure": bool(is_runtime_failure),
        "is_valid_analytical": bool(is_valid_analytical),
        "validity_reason": _reason(is_scaffold, is_runtime_failure, is_heuristic_judge),
    }


def annotate_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Add validity columns to a run-level dataframe."""

    if df.empty:
        out = df.copy()
        for column in (
            "is_scaffold",
            "is_heuristic_judge",
            "is_runtime_failure",
            "is_valid_analytical",
            "validity_reason",
        ):
            out[column] = []
        return out

    flags = [classify_run_row(row) for _, row in df.iterrows()]
    flag_df = pd.DataFrame(flags)
    out = df.reset_index(drop=True).copy()
    for column in flag_df.columns:
        out[column] = flag_df[column]
    return out


def filter_valid_runs(
    df: pd.DataFrame,
    *,
    require_model_judge: bool = False,
    exclude_scaffold: bool = True,
) -> pd.DataFrame:
    """Return only runs eligible for primary thesis analyses."""

    annotated = annotate_validity(df)
    if annotated.empty:
        return annotated

    mask = pd.Series([True] * len(annotated))
    if exclude_scaffold:
        mask &= annotated["is_valid_analytical"].astype(bool)
    if require_model_judge:
        mask &= ~annotated["is_heuristic_judge"].astype(bool)
    return annotated.loc[mask].reset_index(drop=True)


def validity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize validity status by framework."""

    annotated = annotate_validity(df)
    if annotated.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_col = "framework" if "framework" in annotated.columns else None
    groups = annotated.groupby(group_col, dropna=False) if group_col else [(None, annotated)]
    for key, group in groups:
        rows.append(
            {
                "framework": key,
                "n_total": int(len(group)),
                "n_scaffold": int(group["is_scaffold"].sum()),
                "n_heuristic_judge": int(group["is_heuristic_judge"].sum()),
                "n_valid_analytical": int(group["is_valid_analytical"].sum()),
                "valid_share": float(group["is_valid_analytical"].mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _reason(is_scaffold: bool, is_runtime_failure: bool, is_heuristic_judge: bool) -> str:
    reasons: list[str] = []
    if is_scaffold:
        reasons.append("scaffold_or_fallback")
    if is_runtime_failure:
        reasons.append("runtime_failure")
    if is_heuristic_judge:
        reasons.append("heuristic_judge")
    if not reasons:
        return "valid"
    return ";".join(reasons)
