"""Experiment harness for repeated, reproducible benchmark runs.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from benchmarks.base import BenchmarkTask
from evaluation.mast_classifier import MASTClassifier, MASTJudgement
from evaluation.metrics import results_to_frame, summarize_results
from frameworks.base_runner import TraceResult


RunnerFactory = Callable[[], Any]


@dataclass(slots=True)
class ExperimentOutput:
    """Persisted files for one experiment batch."""

    summary_path: Path
    records_path: Path
    traces_dir: Path


class ExperimentHarness:
    """Run repeated experiments and persist all results in a thesis-friendly format."""

    def __init__(
        self,
        *,
        runner_factory: RunnerFactory,
        output_dir: Path | str = Path("results/experiments"),
        judge_model_name: str | None = None,
        mast_judge_enabled: bool = False,
        mast_judge_model: Any | None = None,
        mast_judge_temperature: float = 0.0,
        mast_judge_max_retries: int = 2,
        seed: int | None = 42,
    ) -> None:
        self.runner_factory = runner_factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.judge_model_name = judge_model_name
        self.mast_judge_enabled = mast_judge_enabled
        self.mast_judge_temperature = mast_judge_temperature
        self.mast_judge_max_retries = mast_judge_max_retries
        self.mast_judge_model = mast_judge_model or self._build_judge_model()

    def run(
        self,
        *,
        task_description: str,
        framework_name: str,
        benchmark_name: str,
        num_runs: int = 50,
        max_steps: int = 25,
    ) -> dict[str, Any]:
        """Execute N runs and store trace-level as well as aggregate outputs."""

        traces: list[TraceResult] = []
        judge = MASTClassifier(
            model=self.mast_judge_model if self.mast_judge_enabled else None,
            temperature=self.mast_judge_temperature,
            max_retries=self.mast_judge_max_retries,
        )

        batch_dir = self.output_dir / benchmark_name / framework_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = batch_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        for run_index in range(num_runs):
            runner = self.runner_factory()
            run_with_context = getattr(runner, "run_task_with_context", None)
            if callable(run_with_context):
                result = run_with_context(
                    task_description=task_description,
                    benchmark_name=benchmark_name,
                    max_steps=max_steps,
                )
            else:
                result = runner.run_task(task_description=task_description, max_steps=max_steps)
            traces.append(result)
            trace_text = self._serialize_trace(result)
            judgement = judge.classify(trace_text)
            self._write_single_run(batch_dir=batch_dir, run_index=run_index, result=result, judgement=judgement)

        summary = summarize_results(traces)
        summary.update(
            {
                "framework": framework_name,
                "benchmark": benchmark_name,
                "num_runs": num_runs,
                "max_steps": max_steps,
                "judge_model_name": self.judge_model_name,
                "mast_judge_enabled": self.mast_judge_enabled and self.mast_judge_model is not None,
                "mast_judge_runtime": "model" if self.mast_judge_enabled and self.mast_judge_model is not None else "heuristic_fallback",
            }
        )

        summary_path = batch_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        records_frame = results_to_frame(traces)
        records_path = batch_dir / "records.csv"
        records_frame.to_csv(records_path, index=False)

        return {
            "summary": summary,
            "summary_path": str(summary_path),
            "records_path": str(records_path),
            "batch_dir": str(batch_dir),
        }

    def _build_judge_model(self) -> Any | None:
        """Build an optional LLM-as-Judge model when enabled by config."""

        if not self.mast_judge_enabled:
            return None
        if not self.judge_model_name:
            return None
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return None
        try:
            return ChatOpenAI(
                model=self.judge_model_name,
                temperature=self.mast_judge_temperature,
            )
        except Exception:
            return None

    def run_tasks(
        self,
        *,
        tasks: list[BenchmarkTask],
        framework_name: str,
        benchmark_name: str,
        num_runs: int = 1,
        max_steps: int = 25,
    ) -> dict[str, Any]:
        """Run a list of benchmark tasks and persist a separate batch for each task."""

        outputs: list[dict[str, Any]] = []
        for task in tasks:
            task_output = self.run(
                task_description=task.prompt,
                framework_name=framework_name,
                benchmark_name=f"{benchmark_name}/{task.task_id}",
                num_runs=num_runs,
                max_steps=max_steps,
            )
            outputs.append({"task_id": task.task_id, "prompt": task.prompt, **task_output})
        return {"count": len(outputs), "tasks": outputs}

    def _serialize_trace(self, result: TraceResult) -> str:
        return "\n".join(step.model_dump_json() for step in result.full_trace)

    def _write_single_run(
        self,
        *,
        batch_dir: Path,
        run_index: int,
        result: TraceResult,
        judgement: MASTJudgement,
    ) -> None:
        run_payload = {
            "run_index": run_index,
            "success": result.success,
            "final_output": result.final_output,
            "metrics": result.metrics.model_dump(),
            "raw_log_path": result.raw_log_path,
            "run_id": result.run_id,
            "judgement": judgement.model_dump(),
        }
        (batch_dir / f"run_{run_index:03d}.json").write_text(
            json.dumps(run_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Read a YAML config file for convenience scripts."""

    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
