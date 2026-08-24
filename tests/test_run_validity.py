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


def test_classify_run_row_marks_metagpt_ultra_low_latency_as_scaffold() -> None:
    row = {
        "framework": "metagpt",
        "final_output": "Completed task.",
        "latency_seconds": 0.003,
    }

    flags = classify_run_row(row)

    assert flags["is_scaffold"] is True
    assert flags["is_valid_analytical"] is False


def test_classify_run_row_marks_normal_run_as_valid() -> None:
    row = {
        "framework": "autogen",
        "final_output": "Produced concrete plan, checks, and reconciled output.",
        "latency_seconds": 20.0,
    }

    flags = classify_run_row(row)

    assert flags["is_scaffold"] is False
    assert flags["is_valid_analytical"] is True


def test_annotate_validity_adds_expected_columns() -> None:
    df = pd.DataFrame([{"framework": "autogen", "final_output": "ok", "latency_seconds": 1.0}])

    out = annotate_validity(df)

    assert "is_scaffold" in out.columns
    assert "is_valid_analytical" in out.columns
