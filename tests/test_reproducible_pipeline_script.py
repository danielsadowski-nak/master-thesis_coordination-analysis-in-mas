"""Smoke test for the reproducible pipeline script helpers."""

from __future__ import annotations

from pathlib import Path

from experiments.run_reproducible_pipeline import sample_tasks
from benchmarks.base import BenchmarkTask


def test_sample_tasks_preserves_stable_seeded_selection() -> None:
    tasks = [
        BenchmarkTask(task_id=f"task-{index}", prompt=f"Prompt {index}", metadata={})
        for index in range(10)
    ]

    sampled = sample_tasks(tasks, 4, seed=42)

    assert len(sampled) == 4
    assert [task.task_id for task in sampled] == ["task-0", "task-1", "task-4", "task-9"]