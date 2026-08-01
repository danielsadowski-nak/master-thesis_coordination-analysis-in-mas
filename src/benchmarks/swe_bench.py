"""SWE-Bench benchmark helpers.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.base import BenchmarkTask, ensure_path, first_non_empty, load_records, metadata_without


def load_swe_bench_verified_subset(source_path: Path | str | None = None) -> list[BenchmarkTask]:
    """Load and normalize a SWE-Bench Verified subset from common file formats.

    Supported sources include JSON, JSONL, CSV, or plain text. When the source is a directory,
    the loader scans for common dataset filenames and combines their records.
    """

    path = ensure_path(source_path)
    if path is None:
        return []
    if path.is_dir():
        for candidate_name in ("swe_bench_verified.jsonl", "swe_bench_verified.json", "verified.jsonl", "verified.json"):
            candidate = path / candidate_name
            if candidate.exists():
                path = candidate
                break
        else:
            return []

    records = load_records(path)
    tasks: list[BenchmarkTask] = []
    for index, record in enumerate(records):
        task_id = first_non_empty(record, ("instance_id", "task_id", "id", "problem_id"), default=f"swe-{index:05d}")
        prompt = first_non_empty(
            record,
            ("problem_statement", "prompt", "question", "instruction", "text"),
            default="",
        )
        if not prompt:
            continue
        metadata = metadata_without(
            record,
            ("instance_id", "task_id", "id", "problem_id", "problem_statement", "prompt", "question", "instruction", "text"),
        )
        metadata.setdefault("source", str(path))
        metadata.setdefault("benchmark", "swe_bench_verified")
        tasks.append(BenchmarkTask(task_id=task_id, prompt=prompt, metadata=metadata))
    return tasks


def render_swe_bench_prompt(task: BenchmarkTask) -> str:
    """Create a thesis-friendly prompt for a SWE-Bench task."""

    context_lines = [f"Task ID: {task.task_id}", task.prompt]
    repository = task.metadata.get("repo")
    if repository:
        context_lines.append(f"Repository: {repository}")
    base_commit = task.metadata.get("base_commit")
    if base_commit:
        context_lines.append(f"Base commit: {base_commit}")
    if task.metadata.get("hints_text"):
        context_lines.append(f"Hints: {task.metadata['hints_text']}")
    return "\n\n".join(context_lines)
