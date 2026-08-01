"""Tests for benchmark loaders."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.gaia import load_gaia_tasks, render_gaia_prompt
from benchmarks.swe_bench import load_swe_bench_verified_subset, render_swe_bench_prompt


def test_load_swe_bench_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "swe_bench_verified.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "swe-001",
                        "problem_statement": "Fix the failing parser.",
                        "repo": "example/repo",
                        "base_commit": "abc123",
                        "hints_text": "Look at the tokenizer.",
                    }
                ),
                json.dumps({"instance_id": "swe-002", "problem_statement": "Add a missing test."}),
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_swe_bench_verified_subset(source)

    assert [task.task_id for task in tasks] == ["swe-001", "swe-002"]
    assert "Fix the failing parser." in render_swe_bench_prompt(tasks[0])
    assert tasks[0].metadata["benchmark"] == "swe_bench_verified"


def test_load_gaia_plain_text(tmp_path: Path) -> None:
    source = tmp_path / "gaia.txt"
    source.write_text("Question one\n\nQuestion two", encoding="utf-8")

    tasks = load_gaia_tasks(source)

    assert len(tasks) == 2
    assert tasks[0].task_id == "gaia-00000"
    assert "Question one" in render_gaia_prompt(tasks[0])
    assert tasks[0].metadata["benchmark"] == "gaia"
