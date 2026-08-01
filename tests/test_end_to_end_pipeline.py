"""End-to-end smoke test for the experiment pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.experiment_harness import ExperimentHarness
from evaluation.plots import export_batch_summary, load_batch_artifacts, render_experiment_reports
from frameworks.base_runner import RunMetrics, StepKind, TraceResult, TraceStep


class SmokeRunner:
    def run_task_with_context(self, *, task_description: str, benchmark_name: str, max_steps: int = 25) -> TraceResult:
        return TraceResult(
            success=True,
            final_output=f"completed: {task_description}",
            full_trace=[TraceStep(step_index=0, kind=StepKind.FINAL, content=task_description)],
            metrics=RunMetrics(latency_seconds=0.02, total_tokens=4),
            raw_log_path=f"results/traces/{benchmark_name}.jsonl",
            run_id="smoke-run",
        )


def test_end_to_end_pipeline_runs_and_renders(tmp_path: Path) -> None:
    harness = ExperimentHarness(runner_factory=lambda: SmokeRunner(), output_dir=tmp_path)
    output = harness.run(
        task_description="Solve the task.",
        framework_name="langgraph",
        benchmark_name="gaia",
        num_runs=1,
        max_steps=5,
    )

    batch_dir = Path(output["batch_dir"])
    artifacts = load_batch_artifacts(batch_dir)
    figures_dir = batch_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_files = render_experiment_reports(batch_dir, figures_dir)
    summary_path = export_batch_summary(artifacts, figures_dir)

    assert (batch_dir / "summary.json").exists()
    assert (batch_dir / "records.csv").exists()
    assert len(generated_files) == 3
    assert all(path.exists() for path in generated_files)
    assert summary_path.exists()