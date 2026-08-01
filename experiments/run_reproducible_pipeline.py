"""Small reproducible end-to-end experiment runner for thesis validation.

Generated with GitHub Copilot assistance - reviewed and adapted by author.

This script is intentionally lightweight and easy to extend toward larger batches
or multiple frameworks. It reuses the existing LangGraph runner, LangSmith tracing,
the MAST classifier, and the experiment harness.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.experiment_harness import ExperimentHarness
from evaluation.plots import load_batch_artifacts
from frameworks.langgraph_runner import LangGraphRunner
from langchain_openai import ChatOpenAI
from utils.benchmark_loader import load_benchmark_tasks
from utils.config import apply_llm_runtime_environment, load_experiment_config, resolve_repo_path


DEFAULT_TASK_COUNT = 4
DEFAULT_RUN_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small reproducible end-to-end MAS coordination experiment."
    )
    parser.add_argument("--config", type=Path, default=Path("experiments/config.yaml"))
    parser.add_argument("--benchmark", type=str, default=None, help="Override benchmark name from config.")
    parser.add_argument("--benchmark-source", type=Path, default=None, help="Override benchmark source path.")
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASK_COUNT, help="Number of tasks to sample.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUN_COUNT, help="Runs per task.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name for the strong runner.")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai"], help="Model provider.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Root directory under which timestamped experiment folders are created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for reproducibility across task sampling and runs.",
    )
    return parser.parse_args()


def build_llm(model_name: str) -> Any:
    """Build the strong model used by the LangGraph runner.

    Current implementation uses OpenAI Chat models via langchain-openai.
    Extend this helper later if you want to support Anthropic or other providers.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")
    return ChatOpenAI(model=model_name, temperature=0)


def sample_tasks(tasks: list[Any], sample_size: int, seed: int | None = None) -> list[Any]:
    """Take a stable sample of tasks while preserving the benchmark order when possible."""

    if sample_size >= len(tasks):
        return tasks
    if seed is None:
        return tasks[:sample_size]
    import random

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(tasks)), sample_size))
    return [tasks[index] for index in indices]


def summarize_primary_modes(run_payloads: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Summarize the most common primary MAST failure modes across runs."""

    counter: Counter[str] = Counter()
    for payload in run_payloads:
        judgement = payload.get("judgement", {})
        for mode in judgement.get("primary_failure_modes", []) or []:
            counter[str(mode)] += 1
    return counter.most_common(5)


def print_summary(*, task_results: list[dict[str, Any]], all_records: list[dict[str, Any]], all_run_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Print a compact thesis-friendly summary to stdout."""

    total_runs = len(all_records)
    success_rate = 0.0
    mean_latency_seconds = 0.0
    mean_total_tokens = 0.0
    mean_cost_usd = 0.0
    if all_records:
        import pandas as pd

        dataframe = pd.DataFrame(all_records)
        if "success" in dataframe.columns:
            success_rate = float(dataframe["success"].mean())
        if "latency_seconds" in dataframe.columns:
            mean_latency_seconds = float(dataframe["latency_seconds"].mean())
        if "total_tokens" in dataframe.columns:
            mean_total_tokens = float(dataframe["total_tokens"].mean())
        if "cost_usd" in dataframe.columns:
            mean_cost_usd = float(dataframe["cost_usd"].mean())

    primary_modes = summarize_primary_modes(all_run_payloads)

    print("\n=== End-to-End Experiment Summary ===")
    print(f"Tasks: {len(task_results)}")
    print(f"Runs per task: {task_results[0]['runs_per_task'] if task_results else 0}")
    print(f"Total runs: {total_runs}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Latency: {mean_latency_seconds:.2f}s")
    print(f"Average Tokens: {mean_total_tokens:.1f}")
    print(f"Average Cost: ${mean_cost_usd:.4f}")
    print("Most common MAST failure modes:")
    if not primary_modes:
        print("- None detected")
    for mode, count in primary_modes:
        print(f"- {mode}: {count}")

    return {
        "tasks": len(task_results),
        "runs_per_task": task_results[0]["runs_per_task"] if task_results else 0,
        "total_runs": total_runs,
        "success_rate": success_rate,
        "mean_latency_seconds": mean_latency_seconds,
        "mean_total_tokens": mean_total_tokens,
        "mean_cost_usd": mean_cost_usd,
        "primary_failure_modes": primary_modes,
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    benchmark_name = args.benchmark or config.benchmark
    benchmark_source = resolve_repo_path(args.benchmark_source) if args.benchmark_source is not None else config.benchmark_source
    apply_llm_runtime_environment(api_key=config.llm_api_key, base_url=config.llm_base_url)
    tasks = load_benchmark_tasks(benchmark_name, benchmark_source)
    if not tasks:
        raise RuntimeError(
            f"No benchmark tasks found for '{benchmark_name}'. Provide --benchmark-source with a GAIA/SWE subset file."
        )

    sampled_tasks = sample_tasks(tasks, args.tasks, seed=args.seed or config.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = resolve_repo_path(args.output_root) / f"reproducible_pipeline_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    results_dir = run_root / "results"
    traces_dir = run_root / "traces"
    checkpoints_dir = run_root / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    llm = build_llm(args.model)
    harness = ExperimentHarness(
        runner_factory=lambda: LangGraphRunner(
            model=llm,
            trace_dir=traces_dir,
            checkpoint_dir=checkpoints_dir,
            seed=args.seed or config.seed,
            langsmith_project=config.langsmith_project,
            langsmith_enabled=False,
            llm_api_key=config.llm_api_key,
            llm_base_url=config.llm_base_url,
            mitigation_strategies=(),
        ),
        output_dir=results_dir,
        judge_model_name=config.judge_model_name,
        mast_judge_enabled=config.mast_judge_enabled,
        mast_judge_temperature=config.mast_judge_temperature,
        mast_judge_max_retries=config.mast_judge_max_retries,
        seed=args.seed or config.seed,
    )

    task_runs: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_run_payloads: list[dict[str, Any]] = []
    for task in sampled_tasks:
        task_output = harness.run(
            task_description=task.prompt,
            framework_name="langgraph",
            benchmark_name=f"{benchmark_name}/{task.task_id}",
            num_runs=args.runs,
            max_steps=config.max_steps,
        )
        artifacts = load_batch_artifacts(task_output["batch_dir"])
        all_records.extend(artifacts.records.to_dict(orient="records") if not artifacts.records.empty else [])
        all_run_payloads.extend(artifacts.run_payloads)
        task_runs.append(
            {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "metadata": task.metadata,
                "batch_dir": task_output["batch_dir"],
                "runs_per_task": args.runs,
            }
        )

    summary_payload = {
        "timestamp_utc": timestamp,
        "benchmark": benchmark_name,
        "model": args.model,
        "provider": args.provider,
        "tasks_sampled": len(sampled_tasks),
        "runs_per_task": args.runs,
        "output_root": str(run_root),
        "tasks": task_runs,
    }
    (run_root / "experiment_manifest.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    aggregate_summary = print_summary(
        task_results=task_runs,
        all_records=all_records,
        all_run_payloads=all_run_payloads,
    )

    (run_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if all_records:
        import pandas as pd

        pd.DataFrame(all_records).to_csv(run_root / "aggregate_records.csv", index=False)

    (run_root / "task_results.json").write_text(
        json.dumps(task_runs, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
