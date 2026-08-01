"""Run a small, low-cost validation experiment for thesis readiness.

This script and helper functions were generated with AI assistance and then
reviewed/adapted for transparent use in the thesis methodology.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.experiment_harness import ExperimentHarness
from evaluation.plots import load_batch_artifacts
from frameworks.langgraph_runner import LangGraphRunner
from utils.benchmark_loader import load_benchmark_tasks
from utils.config import apply_llm_runtime_environment, load_experiment_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small validation experiment for thesis runtime readiness.")
    parser.add_argument("--config", type=Path, default=Path("experiments/config.yaml"))
    parser.add_argument("--coordination-source", type=Path, default=None, help="Optional custom source for coordination tasks.")
    parser.add_argument("--tasks", type=int, default=3, help="How many coordination-suite tasks to run.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per selected task (recommended 3-5).")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini", help="Model for cost-sensitive validation.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--include-external-benchmark", type=str, default=None, choices=["gaia", "swe_bench_verified"])
    parser.add_argument("--external-source", type=Path, default=None, help="Required if include-external-benchmark is set.")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    return parser.parse_args()


def build_langgraph_model(*, model_name: str, temperature: float) -> Any | None:
    """Return a real ChatOpenAI model when configured, otherwise fallback to scaffold mode."""

    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    try:
        return ChatOpenAI(model=model_name, temperature=temperature)
    except Exception:
        return None


def summarize_primary_modes(run_payloads: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for payload in run_payloads:
        judgement = payload.get("judgement", {}) if isinstance(payload, dict) else {}
        for mode in judgement.get("primary_failure_modes", []) or []:
            counter[str(mode)] += 1
    return counter.most_common(5)


def collect_run_data(batch_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_records: list[dict[str, Any]] = []
    all_run_payloads: list[dict[str, Any]] = []
    for batch_dir in batch_dirs:
        artifacts = load_batch_artifacts(batch_dir)
        all_records.extend(artifacts.records.to_dict(orient="records") if not artifacts.records.empty else [])
        all_run_payloads.extend(artifacts.run_payloads)
    return all_records, all_run_payloads


def build_summary(*, task_runs: list[dict[str, Any]], all_records: list[dict[str, Any]], all_run_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    success_rate = 0.0
    mean_latency_seconds = 0.0
    mean_total_tokens = 0.0
    mean_cost_usd = 0.0
    total_runs = len(all_records)
    if all_records:
        frame = pd.DataFrame(all_records)
        success_rate = float(frame["success"].mean()) if "success" in frame.columns else 0.0
        mean_latency_seconds = float(frame["latency_seconds"].mean()) if "latency_seconds" in frame.columns else 0.0
        mean_total_tokens = float(frame["total_tokens"].mean()) if "total_tokens" in frame.columns else 0.0
        mean_cost_usd = float(frame["cost_usd"].mean()) if "cost_usd" in frame.columns else 0.0

    top_mast_modes = summarize_primary_modes(all_run_payloads)
    return {
        "tasks": len(task_runs),
        "runs_per_task": task_runs[0]["runs_per_task"] if task_runs else 0,
        "total_runs": total_runs,
        "success_rate": success_rate,
        "mean_latency_seconds": mean_latency_seconds,
        "mean_total_tokens": mean_total_tokens,
        "mean_cost_usd": mean_cost_usd,
        "top_mast_modes": top_mast_modes,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Validation Summary ===")
    print(f"Tasks: {summary['tasks']}")
    print(f"Runs per task: {summary['runs_per_task']}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Mean latency: {summary['mean_latency_seconds']:.2f}s")
    print(f"Mean tokens: {summary['mean_total_tokens']:.1f}")
    print(f"Mean cost (reported): ${summary['mean_cost_usd']:.4f}")
    print("Top MAST modes:")
    if not summary["top_mast_modes"]:
        print("- None detected")
    for mode, count in summary["top_mast_modes"]:
        print(f"- {mode}: {count}")


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    apply_llm_runtime_environment(api_key=config.llm_api_key, base_url=config.llm_base_url)

    coordination_source = resolve_repo_path(args.coordination_source) if args.coordination_source is not None else None
    selected_tasks = load_benchmark_tasks("coordination_suite", coordination_source)[: max(args.tasks, 1)]
    if not selected_tasks:
        raise RuntimeError("No coordination-suite tasks found. Check data/coordination_tasks or --coordination-source.")

    if args.include_external_benchmark:
        if args.external_source is None:
            raise RuntimeError("--external-source is required when --include-external-benchmark is set.")
        external_tasks = load_benchmark_tasks(args.include_external_benchmark, resolve_repo_path(args.external_source))
        if external_tasks:
            selected_tasks.append(external_tasks[0])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = resolve_repo_path(args.output_root) / f"validation_{timestamp}"
    experiments_dir = run_root / "experiments"
    traces_dir = run_root / "traces"
    checkpoints_dir = run_root / "checkpoints"
    reports_dir = run_root / "reports"
    for path in (experiments_dir, traces_dir, checkpoints_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    model = build_langgraph_model(model_name=args.model_name, temperature=args.temperature)
    harness = ExperimentHarness(
        runner_factory=lambda: LangGraphRunner(
            model=model,
            trace_dir=traces_dir,
            checkpoint_dir=checkpoints_dir,
            seed=config.seed,
            langsmith_project=config.langsmith_project,
            langsmith_enabled=config.langsmith_enabled,
            langsmith_api_key=config.langsmith_api_key,
            langsmith_endpoint=config.langsmith_endpoint,
            langsmith_tags=config.langsmith_tags,
            llm_api_key=config.llm_api_key,
            llm_base_url=config.llm_base_url,
        ),
        output_dir=experiments_dir,
        judge_model_name=config.judge_model_name,
        mast_judge_enabled=config.mast_judge_enabled,
        mast_judge_temperature=config.mast_judge_temperature,
        mast_judge_max_retries=config.mast_judge_max_retries,
        seed=config.seed,
    )

    task_runs: list[dict[str, Any]] = []
    batch_dirs: list[Path] = []
    max_steps = args.max_steps if args.max_steps is not None else config.max_steps
    for task in selected_tasks:
        benchmark_name = f"{task.metadata.get('benchmark', 'coordination_suite')}/{task.task_id}"
        output = harness.run(
            task_description=task.prompt,
            framework_name="langgraph",
            benchmark_name=benchmark_name,
            num_runs=args.runs,
            max_steps=max_steps,
        )
        batch_dir = Path(output["batch_dir"])
        batch_dirs.append(batch_dir)
        task_runs.append(
            {
                "task_id": task.task_id,
                "benchmark": task.metadata.get("benchmark", "coordination_suite"),
                "title": task.metadata.get("title"),
                "runs_per_task": args.runs,
                "batch_dir": str(batch_dir),
            }
        )

    all_records, all_run_payloads = collect_run_data(batch_dirs)
    summary = build_summary(task_runs=task_runs, all_records=all_records, all_run_payloads=all_run_payloads)
    print_summary(summary)

    manifest = {
        "timestamp_utc": timestamp,
        "runner": "langgraph",
        "model_name": args.model_name,
        "temperature": args.temperature,
        "fallback_mode": model is None,
        "tasks_requested": args.tasks,
        "tasks_executed": len(selected_tasks),
        "runs_per_task": args.runs,
        "task_runs": task_runs,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if all_records:
        pd.DataFrame(all_records).to_csv(reports_dir / "all_runs.csv", index=False)

    print(f"\nValidation artifacts written to: {run_root}")


if __name__ == "__main__":
    main()
