"""Tests for the experiment harness."""

from __future__ import annotations

from evaluation.experiment_harness import ExperimentHarness
from frameworks.base_runner import RunMetrics, TraceResult, TraceStep, StepKind


class DummyRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def run_task_with_context(self, *, task_description: str, benchmark_name: str, max_steps: int = 25) -> TraceResult:
        self.calls.append((task_description, benchmark_name, max_steps))
        return TraceResult(
            success=True,
            final_output="ok",
            full_trace=[TraceStep(step_index=0, kind=StepKind.FINAL, content="ok")],
            metrics=RunMetrics(latency_seconds=0.01, total_tokens=1),
            raw_log_path="results/traces/dummy.jsonl",
            run_id="dummy-run",
        )


def test_harness_uses_context_aware_runner(tmp_path) -> None:
    dummy_runner = DummyRunner()
    harness = ExperimentHarness(runner_factory=lambda: dummy_runner, output_dir=tmp_path)

    result = harness.run(
        task_description="Solve the task.",
        framework_name="langgraph",
        benchmark_name="gaia",
        num_runs=1,
        max_steps=7,
    )

    assert dummy_runner.calls == [("Solve the task.", "gaia", 7)]
    assert result["summary"]["framework"] == "langgraph"


def test_harness_reports_heuristic_judge_runtime_by_default(tmp_path) -> None:
    dummy_runner = DummyRunner()
    harness = ExperimentHarness(runner_factory=lambda: dummy_runner, output_dir=tmp_path)

    result = harness.run(
        task_description="Solve the task.",
        framework_name="langgraph",
        benchmark_name="gaia",
        num_runs=1,
        max_steps=7,
    )

    assert result["summary"]["mast_judge_runtime"] == "heuristic_fallback"


def test_harness_passes_run_index_when_runner_factory_supports_it(tmp_path) -> None:
    class IndexedRunner(DummyRunner):
        def __init__(self, seed: int) -> None:
            super().__init__()
            self.seed = seed

    created_seeds: list[int] = []

    def runner_factory(*, run_index: int) -> IndexedRunner:
        seed = 42 + run_index
        created_seeds.append(seed)
        return IndexedRunner(seed=seed)

    harness = ExperimentHarness(runner_factory=runner_factory, output_dir=tmp_path)
    harness.run(
        task_description="Solve the task.",
        framework_name="langgraph",
        benchmark_name="gaia",
        num_runs=3,
        max_steps=3,
    )

    assert created_seeds == [42, 43, 44]