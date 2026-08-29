"""Tests for analytical run validity classification."""

from __future__ import annotations

import pandas as pd

from evaluation.run_validity import annotate_validity, classify_run_row


def test_classify_run_row_marks_scaffold_output_invalid() -> None:
    row = {
        "framework": "langgraph",
        "final_output": "scaffold completed a reproducible placeholder run for the task.",
        "latency_seconds": 0.25,
    }

    flags = classify_run_row(row)

    assert flags["is_scaffold"] is True
    assert flags["is_valid_analytical"] is False


def test_classify_run_row_heuristic_judge_only_stays_valid() -> None:
    row = {
        "framework": "autogen",
        "final_output": "Release plan: milestones, owner map, and rollback checks completed.",
        "mast_summary": "Heuristic fallback only: keyword match in trace text.",
        "raw_log_path": "results/trace.json",
        "latency_seconds": 12.0,
    }

    flags = classify_run_row(row)

    assert flags["is_heuristic_judge"] is True
    assert flags["is_scaffold"] is False
    assert flags["is_valid_analytical"] is True


def test_classify_run_row_placeholder_output_marks_scaffold() -> None:
    row = {
        "framework": "autogen",
        "final_output": "autogen scaffold completed a reproducible placeholder run for the task",
        "raw_log_path": "results/trace.json",
        "latency_seconds": 0.3,
    }

    flags = classify_run_row(row)

    assert flags["is_scaffold"] is True
    assert flags["is_valid_analytical"] is False


def test_classify_run_row_marks_metagpt_ultra_low_latency_as_scaffold() -> None:
    row = {
        "framework": "metagpt",
        "final_output": "Completed task.",
        "latency_seconds": 0.003,
        "raw_log_path": "results/trace.json",
    }

    flags = classify_run_row(row)

    assert flags["is_scaffold"] is True
    assert flags["is_valid_analytical"] is False


def test_classify_run_row_marks_normal_run_as_valid() -> None:
    row = {
        "framework": "autogen",
        "final_output": "Produced concrete plan, checks, and reconciled output.",
        "latency_seconds": 20.0,
        "raw_log_path": "results/trace.json",
    }

    flags = classify_run_row(row)

    assert flags["is_scaffold"] is False
    assert flags["is_valid_analytical"] is True


def test_annotate_validity_adds_expected_columns() -> None:
    df = pd.DataFrame([{"framework": "autogen", "final_output": "ok", "latency_seconds": 1.0}])

    out = annotate_validity(df)

    assert "is_scaffold" in out.columns
    assert "is_valid_analytical" in out.columns
