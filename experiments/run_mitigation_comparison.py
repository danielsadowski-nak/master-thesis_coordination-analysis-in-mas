"""Run controlled mitigation experiments across all four frameworks.

Thesis-oriented design:
- Uses the same fixed tasks as the baseline setup:
  - one SWE-Bench task
  - one or two GAIA tasks
- Controlled comparison per framework and task:
  - without mitigation (baseline condition: "none")
  - with each selected mitigation strategy
- Repetitions:
  - recommended N=50 per framework x task x condition
  - N=30 accepted under compute constraints (must be justified in thesis)
- Fairness:
  - same model name and temperature arguments across frameworks where supported
  - same max_steps and seed schedule
- Outputs:
  - per-run traces + MAST judgements via ExperimentHarness
  - aggregate statistical tables + pairwise tests + comparative figures
  - thesis-ready markdown summary

Artifacts are stored under results/mitigations_YYYY-MM-DD/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmarks.base import BenchmarkTask
from benchmarks.gaia import load_gaia_tasks
from benchmarks.swe_bench import load_swe_bench_verified_subset
from evaluation.experiment_harness import ExperimentHarness
from evaluation.metrics import (
    build_statistical_report,
    pairwise_condition_comparisons,
)
from evaluation.plots import load_experiment_dataframe, render_thesis_report
from frameworks.autogen_runner import AutoGenRunner
from frameworks.crewai_runner import CrewAIRunner
from frameworks.langgraph_runner import LangGraphRunner
from frameworks.metagpt_runner import MetaGptRunner
from utils.config import apply_llm_runtime_environment
from utils.mitigations import build_mitigation_strategies
from utils.runtime_checks import assert_framework_runtime_ready


@dataclass(slots=True)
class FixedTask:
    benchmark: str
    task_id: str
    prompt: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mitigation comparison experiments for the thesis.")
    parser.add_argument("--swe-source", type=Path, required=True)
    parser.add_argument("--gaia-source", type=Path, required=True)
    parser.add_argument("--num-gaia-tasks", type=int, default=2, choices=[1, 2])
    parser.add_argument("--swe-task-id", type=str, default=None)
    parser.add_argument("--gaia-task-ids", nargs="*", default=None)
    parser.add_argument("--frameworks", nargs="*", default=["langgraph", "autogen", "crewai", "metagpt"])
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=[
            "structured_output_validation",
            "supervisor_orchestrator",
            "reflection_independent_verification",
        ],
        help="Mitigation strategy names from utils.mitigations.MITIGATION_REGISTRY",
    )
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--judge-model-name", type=str, default="gpt-4o")
    parser.add_argument("--mast-judge-enabled", action="store_true", help="Enable model-based MAST judge instead of heuristic fallback.")
    parser.add_argument("--mast-judge-temperature", type=float, default=0.0)
    parser.add_argument("--mast-judge-max-retries", type=int, default=2)
    parser.add_argument("--langsmith-project", type=str, default="mas-coordination-analysis")
    parser.add_argument("--llm-api-key", type=str, default=None, help="Optional OpenAI-compatible API key used by the model clients.")
    parser.add_argument("--llm-base-url", type=str, default=None, help="Optional OpenAI-compatible base URL for local or hosted endpoints.")
    parser.add_argument("--comparison-test", choices=["welch", "mannwhitney"], default="welch")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--require-native-frameworks",
        action="store_true",
        help="Fail fast if optional framework dependencies are missing.",
    )
    return parser.parse_args()


def _pick_task(tasks: list[BenchmarkTask], task_id: str | None) -> BenchmarkTask:
    if not tasks:
        raise ValueError("No tasks loaded.")
    if task_id is None:
        return sorted(tasks, key=lambda task: task.task_id)[0]
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Task id '{task_id}' not found in provided benchmark source")


def select_fixed_tasks(args: argparse.Namespace) -> list[FixedTask]:
    swe_tasks = load_swe_bench_verified_subset(args.swe_source)
    gaia_tasks = load_gaia_tasks(args.gaia_source)
    if not swe_tasks:
        raise ValueError(f"No SWE-Bench tasks found at {args.swe_source}")
    if not gaia_tasks:
        raise ValueError(f"No GAIA tasks found at {args.gaia_source}")

    selected_swe = _pick_task(swe_tasks, args.swe_task_id)
    sorted_gaia = sorted(gaia_tasks, key=lambda task: task.task_id)

    selected_gaia: list[BenchmarkTask]
    if args.gaia_task_ids:
        selected_gaia = [_pick_task(sorted_gaia, task_id) for task_id in args.gaia_task_ids[: args.num_gaia_tasks]]
    else:
        selected_gaia = sorted_gaia[: args.num_gaia_tasks]

    fixed_tasks = [
        FixedTask(
            benchmark="swe_bench_verified",
            task_id=selected_swe.task_id,
            prompt=selected_swe.prompt,
            metadata=selected_swe.metadata,
        )
    ]
    fixed_tasks.extend(
        FixedTask(benchmark="gaia", task_id=task.task_id, prompt=task.prompt, metadata=task.metadata)
        for task in selected_gaia
    )
    return fixed_tasks


def _build_langgraph_model(
    model_name: str,
    temperature: float,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any | None:
    apply_llm_runtime_environment(api_key=api_key, base_url=base_url)
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    try:
        return ChatOpenAI(model=model_name, temperature=temperature)
    except Exception:
        return None


def build_runner_factory(
    *,
    framework: str,
    mitigation_names: tuple[str, ...],
    model_name: str,
    temperature: float,
    llm_api_key: str | None,
    llm_base_url: str | None,
    seed: int,
    trace_dir: Path,
    checkpoint_dir: Path,
    langsmith_project: str,
):
    mitigation_strategies = build_mitigation_strategies(mitigation_names)

    if framework == "langgraph":
        model = _build_langgraph_model(model_name, temperature, api_key=llm_api_key, base_url=llm_base_url)
        return lambda: LangGraphRunner(
            model=model,
            mitigation_strategies=mitigation_strategies,
            trace_dir=trace_dir,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("mitigation", framework, *(mitigation_names or ("none",))),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )

    if framework == "autogen":
        return lambda: AutoGenRunner(
            mitigation_strategies=mitigation_strategies,
            trace_dir=trace_dir,
            seed=seed,
            model_name=model_name,
            temperature=temperature,
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("mitigation", framework, *(mitigation_names or ("none",))),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )

    if framework == "crewai":
        return lambda: CrewAIRunner(
            mitigation_strategies=mitigation_strategies,
            trace_dir=trace_dir,
            seed=seed,
            model_name=model_name,
            temperature=temperature,
            process="sequential",
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("mitigation", framework, *(mitigation_names or ("none",))),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )

    if framework == "metagpt":
        return lambda: MetaGptRunner(
            mitigation_strategies=mitigation_strategies,
            trace_dir=trace_dir,
            seed=seed,
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("mitigation", framework, *(mitigation_names or ("none",))),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )

    raise ValueError(f"Unsupported framework '{framework}'")


def run_experiments(args: argparse.Namespace, fixed_tasks: list[FixedTask]) -> Path:
    run_root = args.results_root / f"mitigations_{date.today().isoformat()}"
    experiments_root = run_root / "experiments"
    traces_root = run_root / "traces"
    checkpoints_root = run_root / "checkpoints"
    reports_root = run_root / "reports"
    for directory in (experiments_root, traces_root, checkpoints_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)
    apply_llm_runtime_environment(api_key=args.llm_api_key, base_url=args.llm_base_url)

    conditions: list[tuple[str, tuple[str, ...]]] = [("none", tuple())]
    conditions.extend((strategy_name, (strategy_name,)) for strategy_name in args.strategies)

    run_manifest: list[dict[str, Any]] = []
    for task in fixed_tasks:
        task_key = f"{task.benchmark}/{task.task_id}"
        for framework in args.frameworks:
            for condition_name, mitigation_names in conditions:
                runner_factory = build_runner_factory(
                    framework=framework,
                    mitigation_names=mitigation_names,
                    model_name=args.model_name,
                    temperature=args.temperature,
                    llm_api_key=args.llm_api_key,
                    llm_base_url=args.llm_base_url,
                    seed=args.seed,
                    trace_dir=traces_root / framework / condition_name,
                    checkpoint_dir=checkpoints_root / framework / condition_name,
                    langsmith_project=args.langsmith_project,
                )
                harness = ExperimentHarness(
                    runner_factory=runner_factory,
                    output_dir=experiments_root,
                    judge_model_name=args.judge_model_name,
                    mast_judge_enabled=args.mast_judge_enabled,
                    mast_judge_temperature=args.mast_judge_temperature,
                    mast_judge_max_retries=args.mast_judge_max_retries,
                    seed=args.seed,
                )
                benchmark_name = f"{task_key}/{condition_name}"
                run_output = harness.run(
                    task_description=task.prompt,
                    framework_name=framework,
                    benchmark_name=benchmark_name,
                    num_runs=args.num_runs,
                    max_steps=args.max_steps,
                )
                run_manifest.append(
                    {
                        "framework": framework,
                        "task": task_key,
                        "condition": condition_name,
                        "mitigation_plugins": list(mitigation_names),
                        "num_runs": args.num_runs,
                        "batch_dir": run_output["batch_dir"],
                    }
                )

    (run_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_analysis_reports(run_root, experiments_root, reports_root, args.comparison_test)
    _write_design_note(run_root, args, fixed_tasks)
    return run_root


def _parse_condition_columns(results_df):
    benchmark_parts = results_df["benchmark"].astype(str).str.split("/")
    results_df = results_df.copy()
    results_df["benchmark_family"] = benchmark_parts.str[0]
    results_df["task_id"] = benchmark_parts.str[1]
    results_df["condition"] = benchmark_parts.str[-1]
    results_df["framework_condition"] = results_df["framework"].astype(str) + " | " + results_df["condition"].astype(str)
    return results_df


def _write_analysis_reports(run_root: Path, experiments_root: Path, reports_root: Path, comparison_test: str) -> None:
    results_df = load_experiment_dataframe(experiments_root)
    if results_df.empty:
        raise RuntimeError("No records found for mitigation comparison.")

    results_df = _parse_condition_columns(results_df)
    results_df.to_csv(reports_root / "all_runs.csv", index=False)

    overall_stats = build_statistical_report(
        results_df,
        condition_column="framework_condition",
        failure_modes_column="primary_failure_modes",
    )
    overall_stats["summary"].to_csv(reports_root / "overall_summary_framework_condition.csv", index=False)
    overall_stats["mast_modes"].to_csv(reports_root / "overall_mast_modes_framework_condition.csv", index=False)
    overall_stats["mast_categories"].to_csv(reports_root / "overall_mast_categories_framework_condition.csv", index=False)

    figures_root = reports_root / "figures_overall"
    figures_root.mkdir(parents=True, exist_ok=True)
    render_thesis_report(
        results_df,
        figures_root,
        condition_column="framework_condition",
        failure_modes_column="primary_failure_modes",
    )

    comparison_rows: list[dict[str, Any]] = []
    for framework_name, framework_df in results_df.groupby("framework"):
        available_conditions = set(framework_df["condition"].astype(str).unique())
        if "none" not in available_conditions:
            continue

        for condition_name in sorted(condition for condition in available_conditions if condition != "none"):
            subset = framework_df[framework_df["condition"].isin(["none", condition_name])]
            for metric_name in ("success", "latency_seconds", "cost_usd"):
                if metric_name not in subset.columns:
                    continue
                comparisons = pairwise_condition_comparisons(
                    subset,
                    metric_column=metric_name,
                    condition_column="condition",
                    test=comparison_test,
                )
                if comparisons.empty:
                    continue
                comparison = comparisons.iloc[0].to_dict()
                comparison_rows.append(
                    {
                        "framework": framework_name,
                        "strategy": condition_name,
                        "metric": metric_name,
                        **comparison,
                    }
                )

    comparisons_df = (
        pd.DataFrame(comparison_rows)
        if comparison_rows
        else pd.DataFrame(
            columns=["framework", "strategy", "metric", "test_name", "statistic", "p_value", "effect_size", "effect_size_name"]
        )
    )
    comparisons_df.to_csv(reports_root / "with_vs_without_mitigation_tests.csv", index=False)

    markdown = _build_markdown_summary(results_df, overall_stats["summary"], comparisons_df)
    (reports_root / "mitigation_chapter_summary.md").write_text(markdown, encoding="utf-8")


def _build_markdown_summary(results_df, summary_df, comparisons_df) -> str:
    lines: list[str] = []
    lines.append("# Mitigation Development and Evaluation")
    lines.append("")
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append(f"- Frameworks: {', '.join(sorted(results_df['framework'].astype(str).unique()))}")
    lines.append(f"- Conditions: {', '.join(sorted(results_df['condition'].astype(str).unique()))}")
    lines.append(f"- Tasks: {', '.join(sorted(results_df['task_id'].astype(str).unique()))}")
    lines.append(f"- Total runs: {len(results_df)}")
    lines.append("")

    lines.append("## Aggregate Summary (Framework x Condition)")
    lines.append("")
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("## Pairwise Statistical Tests (with vs. without mitigation)")
    lines.append("")
    if comparisons_df.empty:
        lines.append("No pairwise comparison results available.")
    else:
        lines.append(comparisons_df.to_string(index=False))
    lines.append("")

    lines.append("## Interpretation Guidance for Thesis")
    lines.append("")
    lines.append("- Discuss success-rate gains together with latency/cost trade-offs.")
    lines.append("- Map MAST category shifts to expected mechanism of each strategy.")
    lines.append("- Report effect sizes in addition to p-values.")
    lines.append("- If N=30 was used, include a compute-constraint justification and mention reduced statistical power.")
    lines.append("")

    return "\n".join(lines)


def _write_design_note(run_root: Path, args: argparse.Namespace, fixed_tasks: list[FixedTask]) -> None:
    if args.num_runs >= 50:
        n_note = "N=50 repetitions were used per framework x task x condition."
    elif args.num_runs == 30:
        n_note = "N=30 repetitions were used due to compute constraints; this must be justified in the thesis methods section."
    else:
        n_note = f"N={args.num_runs} repetitions were used; document the deviation from N=50 in the thesis methods section."

    payload = {
        "design": {
            "frameworks": args.frameworks,
            "strategies": args.strategies,
            "num_runs": args.num_runs,
            "max_steps": args.max_steps,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "comparison_test": args.comparison_test,
            "mast_judge_enabled": args.mast_judge_enabled,
            "mast_judge_temperature": args.mast_judge_temperature,
            "mast_judge_max_retries": args.mast_judge_max_retries,
            "note": n_note,
        },
        "tasks": [
            {
                "benchmark": task.benchmark,
                "task_id": task.task_id,
                "source": task.metadata.get("source"),
            }
            for task in fixed_tasks
        ],
    }
    (run_root / "mitigation_design.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.require_native_frameworks:
        assert_framework_runtime_ready(args.frameworks)
    fixed_tasks = select_fixed_tasks(args)
    run_root = run_experiments(args, fixed_tasks)
    print(f"Mitigation comparison completed. Artifacts saved to: {run_root}")


if __name__ == "__main__":
    main()
