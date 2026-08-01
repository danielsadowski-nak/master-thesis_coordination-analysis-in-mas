"""Run the full thesis baseline comparison across four MAS frameworks.

Experimental design (thesis-ready):
- Fixed benchmark tasks:
  - exactly one task from SWE-Bench Verified
  - one or two tasks from GAIA
- Repetitions:
  - default N=50 runs per framework x task
  - optional N=30 when compute limits require it (document this decision in the thesis)
- Frameworks compared:
  - LangGraph, AutoGen (AG2), CrewAI, MetaGPT
- Fairness controls:
  - same model name and temperature arguments are propagated to framework runners where supported
  - same max_steps and seed family across all runs
- Observability:
  - LangSmith tracing enabled for LangGraph
  - structured trace logging enabled for all adapters/runners
- Outputs:
  - per-run traces, MAST judgements, per-batch summaries and CSV records
  - aggregate statistical report + comparative thesis plots (PNG + PDF)
  - all artifacts saved under results/baseline_YYYY-MM-DD/

Usage example:
    python experiments/run_full_baseline_comparison.py \
      --swe-source /path/to/swe_bench_verified.jsonl \
      --gaia-source /path/to/gaia.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmarks.base import BenchmarkTask
from benchmarks.gaia import load_gaia_tasks
from benchmarks.swe_bench import load_swe_bench_verified_subset
from evaluation.experiment_harness import ExperimentHarness
from evaluation.metrics import build_statistical_report, pairwise_condition_comparisons
from evaluation.plots import load_experiment_dataframe, render_thesis_report
from frameworks.autogen_runner import AutoGenRunner
from frameworks.crewai_runner import CrewAIRunner
from frameworks.langgraph_runner import LangGraphRunner
from frameworks.metagpt_runner import MetaGptRunner
from utils.config import apply_llm_runtime_environment
from utils.runtime_checks import assert_framework_runtime_ready


@dataclass(slots=True)
class FixedBaselineTask:
    benchmark: str
    task_id: str
    prompt: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full baseline comparison across four frameworks.")
    parser.add_argument("--swe-source", type=Path, required=True, help="Path to SWE-Bench Verified dataset file or directory")
    parser.add_argument("--gaia-source", type=Path, required=True, help="Path to GAIA dataset file or directory")
    parser.add_argument("--num-gaia-tasks", type=int, default=2, choices=[1, 2], help="Number of fixed GAIA tasks (1 or 2)")
    parser.add_argument("--swe-task-id", type=str, default=None, help="Optional fixed SWE task id override")
    parser.add_argument("--gaia-task-ids", nargs="*", default=None, help="Optional fixed GAIA task id overrides")
    parser.add_argument("--num-runs", type=int, default=50, help="Repetitions per framework x task (50 recommended; 30 acceptable for compute limits)")
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
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--comparison-test", choices=["welch", "mannwhitney"], default="welch")
    parser.add_argument(
        "--frameworks",
        nargs="*",
        default=["langgraph", "autogen", "crewai", "metagpt"],
        choices=["langgraph", "autogen", "crewai", "metagpt"],
        help="Framework subset to execute (default: all four).",
    )
    parser.add_argument(
        "--require-native-frameworks",
        action="store_true",
        help="Fail fast if optional framework dependencies are missing.",
    )
    return parser.parse_args()


def select_fixed_tasks(
    *,
    swe_source: Path,
    gaia_source: Path,
    num_gaia_tasks: int,
    swe_task_id: str | None,
    gaia_task_ids: list[str] | None,
) -> list[FixedBaselineTask]:
    swe_tasks = sorted(load_swe_bench_verified_subset(swe_source), key=lambda task: task.task_id)
    gaia_tasks = sorted(load_gaia_tasks(gaia_source), key=lambda task: task.task_id)
    if not swe_tasks:
        raise ValueError(f"No SWE-Bench tasks could be loaded from {swe_source}")
    if not gaia_tasks:
        raise ValueError(f"No GAIA tasks could be loaded from {gaia_source}")

    selected_swe = _pick_task_by_id_or_first(swe_tasks, swe_task_id)

    if gaia_task_ids:
        selected_gaia = [_pick_task_by_id_or_first(gaia_tasks, task_id) for task_id in gaia_task_ids[:num_gaia_tasks]]
    else:
        selected_gaia = gaia_tasks[:num_gaia_tasks]

    fixed_tasks = [
        FixedBaselineTask(
            benchmark="swe_bench_verified",
            task_id=selected_swe.task_id,
            prompt=selected_swe.prompt,
            metadata=selected_swe.metadata,
        )
    ]
    fixed_tasks.extend(
        FixedBaselineTask(
            benchmark="gaia",
            task_id=task.task_id,
            prompt=task.prompt,
            metadata=task.metadata,
        )
        for task in selected_gaia
    )
    return fixed_tasks


def _pick_task_by_id_or_first(tasks: list[BenchmarkTask], task_id: str | None) -> BenchmarkTask:
    if task_id is None:
        return tasks[0]
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Task id '{task_id}' not found in provided benchmark source")


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
            langsmith_tags=("baseline", framework),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )

    if framework == "autogen":
        return lambda: AutoGenRunner(
            trace_dir=framework_trace_dir,
            seed=seed,
            model_name=model_name,
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("baseline", framework),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )

    if framework == "crewai":
        return lambda: CrewAIRunner(
            trace_dir=framework_trace_dir,
            seed=seed,
            model_name=model_name,
            process="sequential",
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("baseline", framework),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )

    if framework == "metagpt":
        return lambda: MetaGptRunner(
            trace_dir=framework_trace_dir,
            seed=seed,
            langsmith_enabled=False,
            langsmith_project=langsmith_project,
            langsmith_tags=("baseline", framework),
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


def run_baseline(args: argparse.Namespace, fixed_tasks: list[FixedBaselineTask]) -> Path:
    baseline_root = args.results_root / f"baseline_{date.today().isoformat()}"
    experiments_root = baseline_root / "experiments"
    traces_root = baseline_root / "traces"
    checkpoints_root = baseline_root / "checkpoints"
    reports_root = baseline_root / "reports"

    experiments_root.mkdir(parents=True, exist_ok=True)
    traces_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    apply_llm_runtime_environment(api_key=args.llm_api_key, base_url=args.llm_base_url)

    frameworks = _normalize_frameworks(args.frameworks)
    run_manifest: list[dict[str, Any]] = []

    for task in fixed_tasks:
        benchmark_key = f"{task.benchmark}/{task.task_id}"
        for framework in frameworks:
            runner_factory = build_runner_factory(
                framework=framework,
                trace_root=traces_root,
                checkpoint_root=checkpoints_root,
                model_name=args.model_name,
                temperature=args.temperature,
                llm_api_key=args.llm_api_key,
                llm_base_url=args.llm_base_url,
                seed=args.seed,
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
            output = harness.run(
                task_description=task.prompt,
                framework_name=framework,
                benchmark_name=benchmark_key,
                num_runs=args.num_runs,
                max_steps=args.max_steps,
            )
            run_manifest.append(
                {
                    "framework": framework,
                    "benchmark": task.benchmark,
                    "task_id": task.task_id,
                    "num_runs": args.num_runs,
                    "batch_dir": output["batch_dir"],
                }
            )

    _write_report_artifacts(
        baseline_root=baseline_root,
        experiments_root=experiments_root,
        reports_root=reports_root,
        run_manifest=run_manifest,
        comparison_test=args.comparison_test,
    )

    return baseline_root


def _write_report_artifacts(
    *,
    baseline_root: Path,
    experiments_root: Path,
    reports_root: Path,
    run_manifest: list[dict[str, Any]],
    comparison_test: str,
) -> None:
    results_df = load_experiment_dataframe(experiments_root)
    if results_df.empty:
        raise RuntimeError("No experiment records were found after baseline execution.")

    stats = build_statistical_report(results_df, condition_column="framework", failure_modes_column="primary_failure_modes")

    summary_df = stats["summary"]
    mast_modes_df = stats["mast_modes"]
    mast_categories_df = stats["mast_categories"]

    summary_df.to_csv(reports_root / "summary_statistics.csv", index=False)
    mast_modes_df.to_csv(reports_root / "mast_modes.csv", index=False)
    mast_categories_df.to_csv(reports_root / "mast_categories.csv", index=False)

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

    run_manifest_path = baseline_root / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _design_note(num_runs: int) -> str:
    if num_runs >= 50:
        return "N=50 repetitions were used per framework x task as planned for the baseline design."
    if num_runs == 30:
        return "N=30 repetitions were used due to compute constraints; this should be explicitly justified in the thesis methods section."
    return f"N={num_runs} repetitions were used. For thesis comparability, document this deviation from N=50 in the methods section."


def _normalize_frameworks(frameworks: list[str]) -> list[str]:
    ordered_unique: list[str] = []
    for framework in frameworks:
        if framework not in ordered_unique:
            ordered_unique.append(framework)
    return ordered_unique


def _write_metadata(baseline_root: Path, args: argparse.Namespace, fixed_tasks: list[FixedBaselineTask]) -> None:
    selected_frameworks = _normalize_frameworks(args.frameworks)
    metadata = {
        "design": {
            "frameworks": selected_frameworks,
            "num_runs": args.num_runs,
            "max_steps": args.max_steps,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "comparison_test": args.comparison_test,
            "mast_judge_enabled": args.mast_judge_enabled,
            "mast_judge_temperature": args.mast_judge_temperature,
            "mast_judge_max_retries": args.mast_judge_max_retries,
            "design_note": _design_note(args.num_runs),
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
    (baseline_root / "baseline_design.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_frameworks = _normalize_frameworks(args.frameworks)
    if not selected_frameworks:
        raise ValueError("At least one framework must be selected via --frameworks.")

    fixed_tasks = select_fixed_tasks(
        swe_source=args.swe_source,
        gaia_source=args.gaia_source,
        num_gaia_tasks=args.num_gaia_tasks,
        swe_task_id=args.swe_task_id,
        gaia_task_ids=args.gaia_task_ids,
    )
    if args.require_native_frameworks:
        assert_framework_runtime_ready(selected_frameworks)

    baseline_root = run_baseline(args, fixed_tasks)
    _write_metadata(baseline_root, args, fixed_tasks)

    print(f"Baseline comparison finished. Artifacts saved to: {baseline_root}")


if __name__ == "__main__":
    main()
