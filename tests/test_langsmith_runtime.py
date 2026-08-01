"""Tests for LangSmith runtime helpers."""

from __future__ import annotations

import os

from utils.langsmith import LangSmithRuntime, build_trace_config


def test_build_trace_config_includes_metadata() -> None:
    config = build_trace_config(
        project="mas-coordination-analysis",
        run_id="run-123",
        framework="langgraph",
        benchmark="gaia",
        task_description="Solve the task.",
        tags=("thesis",),
        extra_metadata={"seed": 42},
    )

    assert config["configurable"]["thread_id"] == "run-123"
    assert config["metadata"]["benchmark"] == "gaia"
    assert config["metadata"]["seed"] == 42
    assert "langgraph" in config["tags"]
    assert "gaia" in config["tags"]


def test_langsmith_runtime_apply_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    runtime = LangSmithRuntime(enabled=True, project="thesis-project", api_key="test-key")
    runtime.apply()

    assert os.getenv("LANGCHAIN_TRACING_V2") == "true"
    assert os.getenv("LANGSMITH_PROJECT") == "thesis-project"
    assert os.getenv("LANGSMITH_API_KEY") == "test-key"
