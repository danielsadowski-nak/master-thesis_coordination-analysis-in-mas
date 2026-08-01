"""GAIA benchmark helpers.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.base import BenchmarkTask, ensure_path, first_non_empty, load_records, metadata_without


def load_gaia_tasks(source_path: Path | str | None = None) -> list[BenchmarkTask]:
    """Load and normalize GAIA tasks from common dataset file formats.

    The loader supports JSON, JSONL, CSV, and plain text sources. When the input is a directory,
    it searches for likely GAIA dataset artifacts.
    """

    path = ensure_path(source_path)
    if path is None:
        return []
    if path.is_dir():
        for candidate_name in ("gaia.jsonl", "gaia.json", "tasks.jsonl", "tasks.json", "gaia.csv"):
            candidate = path / candidate_name
            if candidate.exists():
                path = candidate
                break
        else:
            return []

    records = load_records(path)
    tasks: list[BenchmarkTask] = []
    for index, record in enumerate(records):
        task_id = first_non_empty(record, ("task_id", "id", "question_id", "instance_id"), default=f"gaia-{index:05d}")
        prompt = first_non_empty(
            record,
            ("question", "prompt", "task", "instruction", "query", "text"),
            default="",
        )
        if not prompt:
            continue
        metadata = metadata_without(
            record,
            ("task_id", "id", "question_id", "instance_id", "question", "prompt", "task", "instruction", "query", "text"),
        )
        metadata.setdefault("source", str(path))
        metadata.setdefault("benchmark", "gaia")
        answer = record.get("answer")
        if answer is not None:
            metadata.setdefault("answer", answer)
        tasks.append(BenchmarkTask(task_id=task_id, prompt=prompt, metadata=metadata))
    return tasks


def render_gaia_prompt(task: BenchmarkTask) -> str:
    """Create a thesis-friendly prompt for a GAIA task."""

    context_lines = [f"Task ID: {task.task_id}", task.prompt]
    if task.metadata.get("level"):
        context_lines.append(f"Difficulty level: {task.metadata['level']}")
    if task.metadata.get("answer"):
        context_lines.append("Expected answer is withheld from the agent during execution.")
    return "\n\n".join(context_lines)
