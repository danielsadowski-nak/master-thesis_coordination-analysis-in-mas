"""Minimal low-cost smoke run for runtime readiness checks.

This script was generated with AI assistance and reviewed/adapted for
transparent use in the thesis setup.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.experiment_harness import ExperimentHarness
from frameworks.langgraph_runner import LangGraphRunner
from utils.benchmark_loader import load_benchmark_tasks
from utils.config import resolve_repo_path


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = resolve_repo_path(Path("results")) / f"smoke_{timestamp}"
    experiments_dir = run_root / "experiments"
    traces_dir = run_root / "traces"
    checkpoints_dir = run_root / "checkpoints"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_benchmark_tasks("coordination_suite")
    if tasks:
        task = tasks[0]
        task_description = task.prompt
        benchmark_name = f"coordination_suite/{task.task_id}"
    else:
        task_description = "Summarize the task and explicitly state one verification step."
        benchmark_name = "smoke/default"

    harness = ExperimentHarness(
        runner_factory=lambda: LangGraphRunner(
            model=None,  # Explicit fallback mode: no paid API call.
            trace_dir=traces_dir,
            checkpoint_dir=checkpoints_dir,
            seed=42,
            langsmith_enabled=False,
        ),
        output_dir=experiments_dir,
        seed=42,
    )
    output = harness.run(
        task_description=task_description,
        framework_name="langgraph",
        benchmark_name=benchmark_name,
        num_runs=1,
        max_steps=5,
    )
    (run_root / "smoke_manifest.json").write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "benchmark_name": benchmark_name,
                "summary_path": output["summary_path"],
                "records_path": output["records_path"],
                "notes": "Smoke run used LangGraph fallback mode (no external API calls).",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Smoke run completed.")
    print(f"Artifacts: {run_root}")


if __name__ == "__main__":
    main()
