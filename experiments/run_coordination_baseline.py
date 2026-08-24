"""Run a thesis-grade baseline study on the Coordination Suite primary benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.experiment_harness import ExperimentHarness
from evaluation.metrics import build_statistical_report, pairwise_condition_comparisons
from evaluation.plots import load_experiment_dataframe, render_thesis_report
from frameworks.autogen_runner import AutoGenRunner
from frameworks.crewai_runner import CrewAIRunner
from frameworks.langgraph_runner import LangGraphRunner
from frameworks.metagpt_runner import MetaGptRunner
from utils.benchmark_loader import load_benchmark_tasks
from utils.config import apply_llm_runtime_environment, resolve_repo_path
from utils.runtime_checks import assert_framework_runtime_ready, assert_metagpt_native_runtime_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Coordination Suite baseline study across selected frameworks.")
    parser.add_argument(
        "--coordination-source",
        type=Path,
        default=Path("data/coordination_tasks/coordination_suite_v1.jsonl"),
        help="Coordination Suite JSONL source file.",
    )
    parser.add_argument("--task-ids", nargs="*", default=None, help="Optional fixed task ids.")
    parser.add_argument("--task-limit", type=int, default=None, help="Optional cap on the number of selected tasks.")
    parser.add_argument("--num-runs", type=int, default=30, help="Runs per framework x task cell.")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--judge-model-name", type=str, default="gpt-4o")
    parser.add_argument("--mast-judge-enabled", action="store_true")
    parser.add_argument("--mast-judge-temperature", type=float, default=0.0)
    parser.add_argument("--mast-judge-max-retries", type=int, default=2)
    parser.add_argument("--langsmith-project", type=str, default="mas-coordination-analysis")
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--comparison-test", choices=["welch", "mannwhitney"], default="welch")
    parser.add_argument(
        "--frameworks",
        nargs="*",
        default=["langgraph", "autogen", "crewai", "metagpt"],
        choices=["langgraph", "autogen", "crewai", "metagpt"],
    )
    parser.add_argument("--require-native-frameworks", action="store_true")
    return parser.parse_args()


def select_coordination_tasks(
    coordination_source: Path,
    *,
    task_ids: list[str] | None = None,
    task_limit: int | None = None,
) -> list[Any]:
    tasks = load_benchmark_tasks("coordination_suite", coordination_source)
    if not tasks:
        raise ValueError(f"No Coordination Suite tasks could be loaded from {coordination_source}")

    task_lookup = {task.task_id: task for task in tasks}
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in task_lookup]
        if missing:
            raise ValueError(f"Unknown Coordination Suite task ids: {missing}")
        selected = [task_lookup[task_id] for task_id in task_ids]
    else:
        selected = tasks

    if task_limit is not None:
        selected = selected[: max(task_limit, 0)]
    if not selected:
        raise ValueError("Task selection produced an empty Coordination Suite baseline.")
    return selected


def build_runner_factory(
    *,
    framework: str,
    trace_root: Path,
    checkpoint_root: Path,
    model_name: str,
    temperature: float,
    llm_api_key: str | None,
    llm_base_url: str | None,
    seed: int,
    langsmith_project: str,
) -> Any:
    framework_trace_dir = trace_root / framework
    if framework == "langgraph":
        model = _build_langgraph_model(model_name=model_name, temperature=temperature, api_key=llm_api_key, base_url=llm_base_url)
        return lambda: LangGraphRunner(
            model=model,
            trace_dir=framework_trace_dir,
            checkpoint_dir=checkpoint_root / framework,
            seed=seed,
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("coordination-baseline", framework),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
    if framework == "autogen":
        return lambda: AutoGenRunner(
            trace_dir=framework_trace_dir,
            seed=seed,
            model_name=model_name,
            temperature=temperature,
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("coordination-baseline", framework),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
    if framework == "crewai":
        return lambda: CrewAIRunner(
            trace_dir=framework_trace_dir,
            seed=seed,
            model_name=model_name,
            temperature=temperature,
            process="sequential",
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("coordination-baseline", framework),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
    if framework == "metagpt":
        return lambda: MetaGptRunner(
            trace_dir=framework_trace_dir,
            seed=seed,
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("coordination-baseline", framework),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
    raise ValueError(f"Unsupported framework: {framework}")


def _build_langgraph_model(*, model_name: str, temperature: float, api_key: str | None = None, base_url: str | None = None) -> Any | None:
    apply_llm_runtime_environment(api_key=api_key, base_url=base_url)
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    try:
        return ChatOpenAI(model=model_name, temperature=temperature)
    except Exception:
        return None


def _normalize_frameworks(frameworks: list[str]) -> list[str]:
    ordered_unique: list[str] = []
    for framework in frameworks:
        if framework not in ordered_unique:
            ordered_unique.append(framework)
    return ordered_unique


def run_coordination_baseline(args: argparse.Namespace, tasks: list[Any]) -> Path:
    results_root = resolve_repo_path(args.results_root)
    run_root = results_root / f"coordination_baseline_{date.today().isoformat()}"
    experiments_root = run_root / "experiments"
    traces_root = run_root / "traces"
    checkpoints_root = run_root / "checkpoints"
    reports_root = run_root / "reports"
    for directory in (experiments_root, traces_root, checkpoints_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)

    apply_llm_runtime_environment(api_key=args.llm_api_key, base_url=args.llm_base_url)
    frameworks = _normalize_frameworks(args.frameworks)
    run_manifest: list[dict[str, Any]] = []

    for task in tasks:
        benchmark_name = f"coordination_suite/{task.task_id}"
        for framework in frameworks:
            harness = ExperimentHarness(
                runner_factory=build_runner_factory(
                    framework=framework,
                    trace_root=traces_root,
                    checkpoint_root=checkpoints_root,
                    model_name=args.model_name,
                    temperature=args.temperature,
                    llm_api_key=args.llm_api_key,
                    llm_base_url=args.llm_base_url,
                    seed=args.seed,
                    langsmith_project=args.langsmith_project,
                ),
                output_dir=experiments_root,
                judge_model_name=args.judge_model_name,
                mast_judge_enabled=args.mast_judge_enabled,
                mast_judge_temperature=args.mast_judge_temperature,
                mast_judge_max_retries=args.mast_judge_max_retries,
                seed=args.seed,
            )
            output = harness.run(
                task_description=task.prompt,
                framework_name=framework,
                benchmark_name=benchmark_name,
                num_runs=args.num_runs,
                max_steps=args.max_steps,
            )
            run_manifest.append(
                {
                    "framework": framework,
                    "task_id": task.task_id,
                    "benchmark": "coordination_suite",
                    "coordination_pressure": task.metadata.get("coordination_pressure", []),
                    "expected_behavior": task.metadata.get("expected_behavior"),
                    "num_runs": args.num_runs,
                    "batch_dir": output["batch_dir"],
                }
            )

    _write_reports(run_root=run_root, experiments_root=experiments_root, reports_root=reports_root, run_manifest=run_manifest, comparison_test=args.comparison_test)
    _write_metadata(run_root=run_root, args=args, tasks=tasks)
    return run_root


def _write_reports(*, run_root: Path, experiments_root: Path, reports_root: Path, run_manifest: list[dict[str, Any]], comparison_test: str) -> None:
    results_df = load_experiment_dataframe(experiments_root)
    if results_df.empty:
        raise RuntimeError("No Coordination Suite baseline records were found after execution.")

    report = build_statistical_report(results_df, condition_column="framework", failure_modes_column="primary_failure_modes")
    report["summary"].to_csv(reports_root / "summary_statistics.csv", index=False)
    report["mast_modes"].to_csv(reports_root / "mast_modes.csv", index=False)
    report["mast_categories"].to_csv(reports_root / "mast_categories.csv", index=False)

    for metric_name in ("success", "latency_seconds", "cost_usd"):
        if metric_name in results_df.columns:
            comparisons = pairwise_condition_comparisons(
                results_df,
                metric_column=metric_name,
                condition_column="framework",
                test=comparison_test,
            )
            comparisons.to_csv(reports_root / f"pairwise_{metric_name}.csv", index=False)

    figures_dir = reports_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    render_thesis_report(results_df, figures_dir, condition_column="framework", failure_modes_column="primary_failure_modes")
    (run_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_metadata(*, run_root: Path, args: argparse.Namespace, tasks: list[Any]) -> None:
    metadata = {
        "benchmark": "coordination_suite",
        "design": {
            "frameworks": _normalize_frameworks(args.frameworks),
            "num_runs": args.num_runs,
            "max_steps": args.max_steps,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "comparison_test": args.comparison_test,
            "mast_judge_enabled": args.mast_judge_enabled,
            "study_protocol": "docs/study_protocol.md",
            "codebook": "docs/coordination_suite_codebook.md",
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "title": task.metadata.get("title"),
                "coordination_pressure": task.metadata.get("coordination_pressure", []),
                "expected_behavior": task.metadata.get("expected_behavior"),
                "source": task.metadata.get("source"),
            }
            for task in tasks
        ],
    }
    (run_root / "baseline_design.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_frameworks = _normalize_frameworks(args.frameworks)
    if not selected_frameworks:
        raise ValueError("At least one framework must be selected.")
    if args.require_native_frameworks:
        assert_framework_runtime_ready(selected_frameworks)
        if "metagpt" in selected_frameworks:
            assert_metagpt_native_runtime_ready()

    tasks = select_coordination_tasks(
        resolve_repo_path(args.coordination_source),
        task_ids=args.task_ids,
        task_limit=args.task_limit,
    )
    run_root = run_coordination_baseline(args, tasks)
    print(f"Coordination baseline finished. Artifacts saved to: {run_root}")


if __name__ == "__main__":
    main()