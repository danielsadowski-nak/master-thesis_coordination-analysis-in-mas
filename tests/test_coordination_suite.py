"""Tests for custom coordination-suite benchmark loading."""

from __future__ import annotations

from pathlib import Path

from benchmarks.coordination_suite import load_coordination_suite_records, load_coordination_suite_tasks


def test_load_coordination_suite_records_has_expected_fields() -> None:
    records = load_coordination_suite_records()

    assert len(records) >= 6
    assert records[0].task_id
    assert records[0].coordination_pressure
    assert records[0].expected_behavior
    assert records[0].success_criteria


def test_load_coordination_suite_tasks_converts_to_benchmark_tasks() -> None:
    tasks = load_coordination_suite_tasks()

    assert len(tasks) >= 6
    assert tasks[0].prompt
    assert tasks[0].metadata["benchmark"] == "coordination_suite"
    assert "coordination_pressure" in tasks[0].metadata


def test_load_coordination_suite_records_from_foreign_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    records = load_coordination_suite_records()

    assert records
    assert Path(records[0].metadata["source"]).name == "coordination_suite_v1.jsonl"
