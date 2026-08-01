"""Tests for benchmark task resolution."""

from __future__ import annotations

import json
from pathlib import Path

from utils.benchmark_loader import load_benchmark_tasks


def test_load_benchmark_tasks_for_gaia(tmp_path: Path) -> None:
    source = tmp_path / "gaia.jsonl"
    source.write_text(json.dumps({"task_id": "g-1", "question": "Find the answer."}), encoding="utf-8")

    tasks = load_benchmark_tasks("gaia", source)

    assert len(tasks) == 1
    assert tasks[0].task_id == "g-1"
    assert tasks[0].prompt == "Find the answer."


def test_load_benchmark_tasks_for_unknown_returns_empty(tmp_path: Path) -> None:
    assert load_benchmark_tasks("unknown", tmp_path / "missing.jsonl") == []


def test_load_benchmark_tasks_for_coordination_suite() -> None:
    tasks = load_benchmark_tasks("coordination_suite")

    assert tasks
    assert tasks[0].metadata["benchmark"] == "coordination_suite"
