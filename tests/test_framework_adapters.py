"""Smoke tests for framework adapters."""

from __future__ import annotations

from frameworks.autogen_runner import AutoGenRunner
from frameworks.base_runner import RunMetrics, TraceResult
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


def test_autogen_runner_closes_owned_model_client(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    fake_client = _FakeClient()
    runner = AutoGenRunner(seed=7)

    monkeypatch.setattr(runner, "_load_autogen_components", lambda: (object(), object(), object(), object()))
    monkeypatch.setattr(runner, "_build_model_client", lambda: fake_client)

    async def _fake_native(**_: object) -> TraceResult:
        return TraceResult(
            success=True,
            final_output="ok",
            full_trace=[],
            metrics=RunMetrics(),
            raw_log_path="fake.jsonl",
        )

    monkeypatch.setattr(runner, "_run_native_autogen", _fake_native)

    result = runner.run_task("test", max_steps=1)

    assert result.success is True
    assert fake_client.closed is True


def test_autogen_runner_does_not_close_injected_model_client(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    fake_client = _FakeClient()
    runner = AutoGenRunner(seed=7, model_client=fake_client)

    monkeypatch.setattr(runner, "_load_autogen_components", lambda: (object(), object(), object(), object()))

    async def _fake_native(**_: object) -> TraceResult:
        return TraceResult(
            success=True,
            final_output="ok",
            full_trace=[],
            metrics=RunMetrics(),
            raw_log_path="fake.jsonl",
        )

    monkeypatch.setattr(runner, "_run_native_autogen", _fake_native)

    result = runner.run_task("test", max_steps=1)

    assert result.success is True
    assert fake_client.closed is False
