"""Smoke tests for framework adapters."""

from __future__ import annotations

from frameworks.autogen_runner import AutoGenRunner
from frameworks.crewai_runner import CrewAIRunner
from frameworks.metagpt_runner import MetaGptRunner


def test_autogen_runner_produces_trace(tmp_path) -> None:
    runner = AutoGenRunner(seed=7)
    runner.trace_logger.base_dir = tmp_path / "autogen"
    result = runner.run_task("Solve the coordination task.", max_steps=3)

    assert result.success is True
    assert result.full_trace
    assert result.raw_log_path.endswith(".jsonl")


def test_crewai_runner_produces_trace(tmp_path) -> None:
    runner = CrewAIRunner(seed=7)
    runner.trace_logger.base_dir = tmp_path / "crewai"
    result = runner.run_task("Solve the coordination task.", max_steps=3)

    assert result.success is True
    assert result.full_trace
    assert result.raw_log_path.endswith(".jsonl")


def test_metagpt_runner_produces_trace(tmp_path) -> None:
    runner = MetaGptRunner(seed=7)
    runner.trace_logger.base_dir = tmp_path / "metagpt"
    result = runner.run_task("Solve the coordination task.", max_steps=3)

    assert result.success is True
    assert result.full_trace
    assert result.raw_log_path.endswith(".jsonl")
