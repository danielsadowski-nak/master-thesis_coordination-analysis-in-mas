"""Custom coordination-focused benchmark tasks for thesis experiments.

This module and the bundled tasks were generated with AI assistance and
reviewed/adapted for methodological transparency in the thesis workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from benchmarks.base import BenchmarkTask, ensure_path, load_records


DEFAULT_COORDINATION_TASKS_PATH = Path(__file__).resolve().parents[2] / "data/coordination_tasks/coordination_suite_v1.jsonl"


class CoordinationTask(BaseModel):
    """Schema for custom tasks designed to provoke coordination failures."""

    task_id: str
    title: str
    prompt: str
    roles_hint: list[str] | None = None
    coordination_pressure: list[str] = Field(default_factory=list)
    expected_behavior: str
    success_criteria: list[str] = Field(default_factory=list)
    mast_relevant_categories: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_benchmark_task(self) -> BenchmarkTask:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "title": self.title,
                "roles_hint": self.roles_hint or [],
                "coordination_pressure": self.coordination_pressure,
                "expected_behavior": self.expected_behavior,
                "success_criteria": self.success_criteria,
                "mast_relevant_categories": self.mast_relevant_categories,
                "difficulty": self.difficulty,
                "notes": self.notes,
                "benchmark": "coordination_suite",
            }
        )
        return BenchmarkTask(task_id=self.task_id, prompt=self.prompt, metadata=metadata)


def load_coordination_suite_records(source_path: Path | str | None = None) -> list[CoordinationTask]:
    """Load custom coordination tasks from JSON, JSONL, or CSV files."""

    path = ensure_path(source_path) or DEFAULT_COORDINATION_TASKS_PATH
    if path.is_dir():
        for candidate_name in ("coordination_suite_v1.jsonl", "coordination_suite.jsonl", "tasks.jsonl", "tasks.json"):
            candidate = path / candidate_name
            if candidate.exists():
                path = candidate
                break
        else:
            return []

    records = load_records(path)
    tasks: list[CoordinationTask] = []
    for index, record in enumerate(records):
        if "task_id" not in record:
            record = {**record, "task_id": f"coord-{index:03d}"}
        if "metadata" not in record:
            record["metadata"] = {}
        record["metadata"] = {
            **record["metadata"],
            "source": str(path),
        }
        tasks.append(CoordinationTask.model_validate(record))
    return tasks


def load_coordination_suite_tasks(source_path: Path | str | None = None) -> list[BenchmarkTask]:
    """Load benchmark-ready tasks for the experiment harness interface."""

    return [task.to_benchmark_task() for task in load_coordination_suite_records(source_path)]
