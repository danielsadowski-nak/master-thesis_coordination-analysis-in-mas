"""Tests for human-vs-judge agreement utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.judge_alignment import (
    build_judge_validation_report,
    compute_cohens_kappa,
    filter_annotation_candidates,
    load_annotation_reference,
    parse_mode_set,
    sample_annotation_rows,
    write_annotation_template,
)


def test_parse_mode_set_supports_lists_and_semicolon_strings() -> None:
    assert parse_mode_set(["3.2 Weak Verification", "2.4 Information Withholding"]) == {
        "3.2 Weak Verification",
        "2.4 Information Withholding",
    }
    assert parse_mode_set("3.2 Weak Verification; 2.4 Information Withholding") == {
        "3.2 Weak Verification",
        "2.4 Information Withholding",
    }


def test_compute_cohens_kappa_returns_one_for_identical_labels() -> None:
    kappa = compute_cohens_kappa([True, False, True, False], [True, False, True, False])

    assert kappa == 1.0


def test_sample_annotation_rows_preserves_group_coverage() -> None:
    df = pd.DataFrame(
        [
            {"framework": "langgraph", "benchmark": "coord/task-a", "run_id": "a"},
            {"framework": "langgraph", "benchmark": "coord/task-b", "run_id": "b"},
            {"framework": "autogen", "benchmark": "coord/task-a", "run_id": "c"},
            {"framework": "autogen", "benchmark": "coord/task-b", "run_id": "d"},
            {"framework": "crewai", "benchmark": "coord/task-a", "run_id": "e"},
        ]
    )

    sampled = sample_annotation_rows(df, sample_size=4, seed=42)

    assert len(sampled) == 4
    assert {"langgraph", "autogen"}.issubset(set(sampled["framework"]))


def test_filter_annotation_candidates_excludes_scaffold_and_fallback_runs() -> None:
    df = pd.DataFrame(
        [
            {"final_output": "No external model configured. This LangGraph runner is operating in fallback mode."},
            {"final_output": "autogen scaffold completed a reproducible placeholder run for the task."},
            {"final_output": "A genuine model response with substantive task content."},
        ]
    )

    filtered = filter_annotation_candidates(df, real_model_only=True)

    assert len(filtered) == 1
    assert filtered.iloc[0]["final_output"] == "A genuine model response with substantive task content."


def test_filter_annotation_candidates_real_model_only_keeps_failure_rows() -> None:
    df = pd.DataFrame(
        [
            {"success": True, "final_output": "A genuine model response with substantive task content."},
            {"success": False, "final_output": "A genuine model response but timed out before completion."},
        ]
    )

    filtered = filter_annotation_candidates(df, real_model_only=True)

    assert len(filtered) == 2
    assert set(filtered["success"].astype(bool).tolist()) == {False, True}


def test_filter_annotation_candidates_real_model_only_drops_scaffold_placeholder_text() -> None:
    df = pd.DataFrame(
        [
            {"success": False, "final_output": "metagpt scaffold completed a reproducible placeholder run for the task."},
            {"success": False, "final_output": "A genuine failure output with unresolved dependency conflict."},
        ]
    )

    filtered = filter_annotation_candidates(df, real_model_only=True)

    assert len(filtered) == 1
    assert "placeholder" not in filtered.iloc[0]["final_output"].lower()


def test_load_annotation_reference_prefers_adjudicated_labels(tmp_path: Path) -> None:
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {
                "run_id": "r-1",
                "manual_task_successful": True,
                "manual_primary_failure_modes": "3.2 Weak Verification",
                "adjudicated_task_successful": False,
                "adjudicated_primary_failure_modes": "2.4 Information Withholding",
            }
        ]
    ).to_csv(csv_path, index=False)

    loaded = load_annotation_reference(csv_path)

    assert bool(loaded.loc[0, "reference_task_successful"]) is False
    assert loaded.loc[0, "reference_primary_failure_modes"] == "2.4 Information Withholding"


def test_build_judge_validation_report_computes_summary_and_disagreements() -> None:
    annotations = pd.DataFrame(
        [
            {
                "framework": "langgraph",
                "benchmark": "coord/task-a",
                "run_index": 0,
                "run_id": "run-1",
                "raw_log_path": "trace-1.jsonl",
                "judge_task_successful": True,
                "reference_task_successful": True,
                "judge_primary_failure_modes": "3.2 Weak Verification",
                "reference_primary_failure_modes": "3.2 Weak Verification",
                "manual_summary": "match",
                "adjudicated_summary": "",
                "notes": "",
            },
            {
                "framework": "langgraph",
                "benchmark": "coord/task-b",
                "run_index": 1,
                "run_id": "run-2",
                "raw_log_path": "trace-2.jsonl",
                "judge_task_successful": False,
                "reference_task_successful": True,
                "judge_primary_failure_modes": "2.4 Information Withholding",
                "reference_primary_failure_modes": "3.2 Weak Verification",
                "manual_summary": "mismatch",
                "adjudicated_summary": "",
                "notes": "check",
            },
        ]
    )

    report = build_judge_validation_report(annotations)

    assert report["summary"]["n_runs"] == 2
    assert report["summary"]["primary_mode_exact_match_rate"] == 0.5
    assert not report["per_mode"].empty
    assert len(report["disagreements"]) == 1


def test_write_annotation_template_from_existing_results(tmp_path: Path, monkeypatch) -> None:
    results_root = tmp_path / "synthetic_results"
    output_csv = tmp_path / "annotation_template.csv"

    synthetic = pd.DataFrame(
        [
            {
                "framework": "langgraph",
                "benchmark": "coordination_suite/coord-info-asymmetry-constraint",
                "run_index": 0,
                "run_id": "r1",
                "raw_log_path": "trace-1.jsonl",
                "success": True,
                "final_output": "substantive response",
                "judge_task_successful": True,
                "judge_primary_failure_modes": "3.2 Weak Verification",
                "judge_summary": "ok",
            },
            {
                "framework": "autogen",
                "benchmark": "coordination_suite/coord-clarification-before-execution",
                "run_index": 1,
                "run_id": "r2",
                "raw_log_path": "trace-2.jsonl",
                "success": False,
                "final_output": "genuine failure explanation",
                "judge_task_successful": False,
                "judge_primary_failure_modes": "2.2 Fail to Ask for Clarification",
                "judge_summary": "fail",
            },
        ]
    )
    monkeypatch.setattr("evaluation.judge_alignment.load_runs_for_annotation", lambda _: synthetic)

    written = write_annotation_template(results_root, output_csv, sample_size=2, seed=42, real_model_only=True)

    assert len(written) == 2
    assert output_csv.exists()
    loaded = pd.read_csv(output_csv)
    assert {"judge_task_successful", "judge_primary_failure_modes", "run_id"}.issubset(loaded.columns)