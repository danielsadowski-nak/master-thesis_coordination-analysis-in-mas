"""Command-line entry point for baseline experiments.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.experiment_harness import ExperimentHarness
from frameworks.langgraph_runner import LangGraphRunner
from utils.benchmark_loader import load_benchmark_tasks
from utils.config import apply_llm_runtime_environment, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline coordination experiment.")
    parser.add_argument("--config", type=Path, default=Path("experiments/config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    apply_llm_runtime_environment(api_key=config.llm_api_key, base_url=config.llm_base_url)
    harness = ExperimentHarness(
        runner_factory=lambda: LangGraphRunner(
            checkpoint_dir=config.checkpoint_dir,
            trace_dir=config.trace_dir,
            seed=config.seed,
            langsmith_project=config.langsmith_project,
            langsmith_enabled=False,
            llm_api_key=config.llm_api_key,
            llm_base_url=config.llm_base_url,
        ),
        output_dir=config.output_dir,
        judge_model_name=config.judge_model_name,
        mast_judge_enabled=config.mast_judge_enabled,
        mast_judge_temperature=config.mast_judge_temperature,
        mast_judge_max_retries=config.mast_judge_max_retries,
        seed=config.seed,
    )
    tasks = load_benchmark_tasks(config.benchmark, config.benchmark_source)
    if tasks:
        harness.run_tasks(
            tasks=tasks,
            framework_name=config.framework,
            benchmark_name=config.benchmark,
            num_runs=config.num_runs,
            max_steps=config.max_steps,
        )
    else:
        harness.run(
            task_description=config.task_description,
            framework_name=config.framework,
            benchmark_name=config.benchmark,
            num_runs=config.num_runs,
            max_steps=config.max_steps,
        )


if __name__ == "__main__":
    main()
