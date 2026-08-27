"""Run mitigation comparisons on the Coordination Suite primary benchmark.

Smoke runs remain possible for debugging, but thesis-grade runs should use
--thesis-strict (or equivalent protocol checks) to block underpowered designs.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.experiment_harness import ExperimentHarness
from evaluation.metrics import build_statistical_report, pairwise_condition_comparisons
from evaluation.plots import load_experiment_dataframe, render_thesis_report
from evaluation.task_success import score_output_against_criteria
from frameworks.autogen_runner import AutoGenRunner
from frameworks.crewai_runner import CrewAIRunner
from frameworks.langgraph_runner import LangGraphRunner
from frameworks.metagpt_runner import MetaGptRunner
from utils.benchmark_loader import load_benchmark_tasks
from utils.config import apply_llm_runtime_environment, resolve_repo_path
from utils.mitigations import build_mitigation_strategies
from utils.runtime_checks import assert_framework_runtime_ready, assert_metagpt_native_runtime_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Coordination Suite mitigation study across selected frameworks.")
    parser.add_argument(
        "--coordination-source",
        type=Path,
        default=Path("data/coordination_tasks/coordination_suite_v1.jsonl"),
        help="Coordination Suite JSONL source file.",
    )
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=[
            "structured_output_validation",
            "supervisor_orchestrator",
            "reflection_independent_verification",
        ],
    )
    parser.add_argument("--num-runs", type=int, default=30)
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
    parser.add_argument("--comparison-test", choices=["welch", "mannwhitney"], default="welch")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--frameworks",
        nargs="*",
        default=["langgraph", "autogen", "crewai"],
        choices=["langgraph", "autogen", "crewai", "metagpt"],
    )
    parser.add_argument("--require-native-frameworks", action="store_true")
    parser.add_argument("--thesis-strict", action="store_true", help="Abort underpowered runs (num_runs < 30 or degenerate design cells).")
    parser.add_argument("--allow-smoke", action="store_true", help="Allow smoke-sized runs when --thesis-strict is enabled.")
    parser.add_argument(
        "--cell-timeout-seconds",
        type=float,
        default=None,
        help="Optional wall-clock timeout per framework/task/condition cell. On timeout, a runtime_failure artifact is written and execution continues.",
    )
    parser.add_argument(
        "--repair-missing-only",
        action="store_true",
        help="Only execute cells that do not already have run artifacts for the requested num-runs.",
    )
    parser.add_argument(
        "--repair-label",
        type=str,
        default="",
        help="Optional label recorded in manifest metadata to mark repair/incomplete runs.",
    )
    return parser.parse_args()


class CellTimeoutError(TimeoutError):
    """Raised when a mitigation cell exceeds the configured wall-clock timeout."""


@contextmanager
def _cell_timeout(seconds: float | None):
    if seconds is None:
        yield
        return
    if seconds <= 0:
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise CellTimeoutError(f"Cell exceeded wall-clock timeout ({seconds}s)")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _cell_batch_dir(experiments_root: Path, *, task_key: str, condition_name: str, framework: str) -> Path:
    benchmark_name = f"{task_key}/{condition_name}"
    return experiments_root / benchmark_name / framework


def _cell_completed(batch_dir: Path, num_runs: int) -> bool:
    if num_runs <= 0:
        return False
    target_run = batch_dir / f"run_{num_runs - 1:03d}.json"
    records = batch_dir / "records.csv"
    summary = batch_dir / "summary.json"
    return target_run.exists() and records.exists() and summary.exists()


def _write_runtime_failure_artifacts(
    *,
    experiments_root: Path,
    traces_root: Path,
    task_id: str,
    task_key: str,
    condition_name: str,
    framework: str,
    timeout_seconds: float | None,
    reason: str,
) -> dict[str, Any]:
    benchmark_name = f"{task_key}/{condition_name}"
    batch_dir = _cell_batch_dir(experiments_root, task_key=task_key, condition_name=condition_name, framework=framework)
    batch_dir.mkdir(parents=True, exist_ok=True)

    trace_framework_dir = traces_root / framework / condition_name / framework
    trace_framework_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_framework_dir / f"{uuid.uuid4()}.jsonl"

    timeout_text = (
        f"Runtime failure: mitigation cell timed out before completion. "
        f"framework={framework}; condition={condition_name}; timeout_seconds={timeout_seconds}"
    )
    criteria = score_output_against_criteria(task_id, timeout_text)
    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    trace_event = {
        "step_index": 0,
        "kind": "error",
        "role": "system",
        "content": timeout_text,
        "timestamp": now_iso,
        "metadata": {"reason": reason, "timeout_seconds": timeout_seconds},
    }
    trace_path.write_text(json.dumps(trace_event, ensure_ascii=False) + "\n", encoding="utf-8")

    run_payload = {
        "run_index": 0,
        "success": False,
        "final_output": timeout_text,
        "metrics": {
            "latency_seconds": float(timeout_seconds or 0.0),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tool_calls": 0,
            "steps_executed": 1,
            "cost_usd": 0.0,
        },
        "raw_log_path": str(trace_path),
        "run_id": str(uuid.uuid4()),
        "runtime_mode": "runtime_failure",
        "is_valid_analytical": False,
        "judgement": {
            "task_successful": False,
            "primary_failure_modes": [],
            "summary": timeout_text,
            "reasoning": "runtime timeout",
            "raw_response": "",
        },
        "criteria_success": criteria["criteria_success"],
        "criteria_matched": criteria["criteria_matched"],
        "criteria_total": criteria["criteria_total"],
        "criteria_scorer": criteria["scorer"],
        "task_id": task_id,
        "is_scaffold": False,
        "is_heuristic_judge": True,
        "is_runtime_failure": True,
        "validity_reason": reason,
        "derived_is_valid_analytical": False,
        "mast_judge_enabled": False,
        "mast_judge_runtime": "heuristic_fallback",
    }
    (batch_dir / "run_000.json").write_text(json.dumps(run_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with (batch_dir / "records.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_index",
                "success",
                "final_output",
                "latency_seconds",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "tool_calls",
                "steps_executed",
                "cost_usd",
                "raw_log_path",
                "run_id",
                "runtime_mode",
                "is_valid_analytical",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_index": 0,
                "success": False,
                "final_output": timeout_text,
                "latency_seconds": float(timeout_seconds or 0.0),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "tool_calls": 0,
                "steps_executed": 1,
                "cost_usd": 0.0,
                "raw_log_path": str(trace_path),
                "run_id": run_payload["run_id"],
                "runtime_mode": "runtime_failure",
                "is_valid_analytical": False,
            }
        )

    summary = {
        "num_runs": 1,
        "success_rate": 0.0,
        "mean_latency_seconds": float(timeout_seconds or 0.0),
        "framework": framework,
        "benchmark": benchmark_name,
        "runtime_failure": True,
        "runtime_failure_reason": reason,
    }
    (batch_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "batch_dir": str(batch_dir),
        "run_json": str(batch_dir / "run_000.json"),
        "trace_path": str(trace_path),
    }


def select_coordination_tasks(source: Path, *, task_ids: list[str] | None = None, task_limit: int | None = None) -> list[Any]:
    tasks = load_benchmark_tasks("coordination_suite", source)
    if not tasks:
        raise ValueError(f"No Coordination Suite tasks could be loaded from {source}")

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
        raise ValueError("Task selection produced an empty Coordination Suite mitigation study.")
    return selected


def _normalize_frameworks(frameworks: list[str]) -> list[str]:
    ordered_unique: list[str] = []
    for framework in frameworks:
        if framework not in ordered_unique:
            ordered_unique.append(framework)
    return ordered_unique


def _build_langgraph_model(model_name: str, temperature: float, *, api_key: str | None = None, base_url: str | None = None) -> Any | None:
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
) -> Any:
    mitigation_strategies = build_mitigation_strategies(mitigation_names)
    if framework == "langgraph":
        model = _build_langgraph_model(model_name, temperature, api_key=llm_api_key, base_url=llm_base_url)

        def _factory(run_index: int = 0) -> LangGraphRunner:
            derived_seed = seed + int(run_index)
            return LangGraphRunner(
                model=model,
                mitigation_strategies=mitigation_strategies,
                trace_dir=trace_dir,
                checkpoint_dir=checkpoint_dir,
                seed=derived_seed,
                langsmith_enabled=False,
                langsmith_project=langsmith_project,
                langsmith_tags=("coordination-mitigation", framework, *(mitigation_names or ("none",))),
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )

        return _factory
    if framework == "autogen":

        def _factory(run_index: int = 0) -> AutoGenRunner:
            derived_seed = seed + int(run_index)
            return AutoGenRunner(
                mitigation_strategies=mitigation_strategies,
                trace_dir=trace_dir,
                seed=derived_seed,
                model_name=model_name,
                temperature=temperature,
                langsmith_enabled=False,
                langsmith_project=langsmith_project,
                langsmith_tags=("coordination-mitigation", framework, *(mitigation_names or ("none",))),
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )

        return _factory
    if framework == "crewai":

        def _factory(run_index: int = 0) -> CrewAIRunner:
            derived_seed = seed + int(run_index)
            return CrewAIRunner(
                mitigation_strategies=mitigation_strategies,
                trace_dir=trace_dir,
                seed=derived_seed,
                model_name=model_name,
                temperature=temperature,
                process="sequential",
                langsmith_enabled=False,
                langsmith_project=langsmith_project,
                langsmith_tags=("coordination-mitigation", framework, *(mitigation_names or ("none",))),
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )

        return _factory
    if framework == "metagpt":

        def _factory(run_index: int = 0) -> MetaGptRunner:
            derived_seed = seed + int(run_index)
            return MetaGptRunner(
                mitigation_strategies=mitigation_strategies,
                trace_dir=trace_dir,
                seed=derived_seed,
                langsmith_enabled=False,
                langsmith_project=langsmith_project,
                langsmith_tags=("coordination-mitigation", framework, *(mitigation_names or ("none",))),
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )

        return _factory
    raise ValueError(f"Unsupported framework '{framework}'")


def run_coordination_mitigation(args: argparse.Namespace, tasks: list[Any]) -> Path:
    results_root = resolve_repo_path(args.results_root)
    run_root = results_root / f"coordination_mitigations_{date.today().isoformat()}"
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

    for task in tasks:
        task_key = f"coordination_suite/{task.task_id}"
        for framework in _normalize_frameworks(args.frameworks):
            for condition_name, mitigation_names in conditions:
                cell_start = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                print(
                    f"START cell framework={framework} condition={condition_name} task_id={task.task_id} time={cell_start}",
                    flush=True,
                )
                batch_dir = _cell_batch_dir(
                    experiments_root,
                    task_key=task_key,
                    condition_name=condition_name,
                    framework=framework,
                )
                if args.repair_missing_only and _cell_completed(batch_dir, args.num_runs):
                    run_manifest.append(
                        {
                            "framework": framework,
                            "task_id": task.task_id,
                            "condition": condition_name,
                            "mitigation_plugins": list(mitigation_names),
                            "coordination_pressure": task.metadata.get("coordination_pressure", []),
                            "num_runs": args.num_runs,
                            "batch_dir": str(batch_dir),
                            "status": "skipped_existing",
                            "started_at": cell_start,
                            "ended_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        }
                    )
                    print(
                        f"SKIP cell framework={framework} condition={condition_name} task_id={task.task_id} reason=existing-artifacts",
                        flush=True,
                    )
                    continue

                harness = ExperimentHarness(
                    runner_factory=build_runner_factory(
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
                    ),
                    output_dir=experiments_root,
                    judge_model_name=args.judge_model_name,
                    mast_judge_enabled=args.mast_judge_enabled,
                    mast_judge_temperature=args.mast_judge_temperature,
                    mast_judge_max_retries=args.mast_judge_max_retries,
                    seed=args.seed,
                )
                benchmark_name = f"{task_key}/{condition_name}"
                try:
                    timeout_seconds = args.cell_timeout_seconds
                    start_perf = time.perf_counter()
                    with _cell_timeout(timeout_seconds):
                        output = harness.run(
                            task_description=task.prompt,
                            framework_name=framework,
                            benchmark_name=benchmark_name,
                            num_runs=args.num_runs,
                            max_steps=args.max_steps,
                        )
                    elapsed = time.perf_counter() - start_perf
                    run_manifest.append(
                        {
                            "framework": framework,
                            "task_id": task.task_id,
                            "condition": condition_name,
                            "mitigation_plugins": list(mitigation_names),
                            "coordination_pressure": task.metadata.get("coordination_pressure", []),
                            "num_runs": args.num_runs,
                            "batch_dir": output["batch_dir"],
                            "status": "completed",
                            "started_at": cell_start,
                            "ended_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            "elapsed_seconds": elapsed,
                            "repair_label": args.repair_label,
                        }
                    )
                    print(
                        f"END cell framework={framework} condition={condition_name} task_id={task.task_id} "
                        f"run_json={Path(output['batch_dir']) / 'run_000.json'} elapsed_seconds={elapsed:.2f}",
                        flush=True,
                    )
                except CellTimeoutError:
                    timeout_written = _write_runtime_failure_artifacts(
                        experiments_root=experiments_root,
                        traces_root=traces_root,
                        task_id=task.task_id,
                        task_key=task_key,
                        condition_name=condition_name,
                        framework=framework,
                        timeout_seconds=args.cell_timeout_seconds,
                        reason="cell_timeout",
                    )
                    run_manifest.append(
                        {
                            "framework": framework,
                            "task_id": task.task_id,
                            "condition": condition_name,
                            "mitigation_plugins": list(mitigation_names),
                            "coordination_pressure": task.metadata.get("coordination_pressure", []),
                            "num_runs": args.num_runs,
                            "batch_dir": timeout_written["batch_dir"],
                            "status": "runtime_failure",
                            "failure_reason": "cell_timeout",
                            "timeout_seconds": args.cell_timeout_seconds,
                            "started_at": cell_start,
                            "ended_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            "repair_label": args.repair_label,
                        }
                    )
                    print(
                        f"END cell framework={framework} condition={condition_name} task_id={task.task_id} "
                        f"status=runtime_failure reason=cell_timeout run_json={timeout_written['run_json']}",
                        flush=True,
                    )
                except Exception as exc:
                    failure_written = _write_runtime_failure_artifacts(
                        experiments_root=experiments_root,
                        traces_root=traces_root,
                        task_id=task.task_id,
                        task_key=task_key,
                        condition_name=condition_name,
                        framework=framework,
                        timeout_seconds=args.cell_timeout_seconds,
                        reason=f"cell_exception:{type(exc).__name__}",
                    )
                    run_manifest.append(
                        {
                            "framework": framework,
                            "task_id": task.task_id,
                            "condition": condition_name,
                            "mitigation_plugins": list(mitigation_names),
                            "coordination_pressure": task.metadata.get("coordination_pressure", []),
                            "num_runs": args.num_runs,
                            "batch_dir": failure_written["batch_dir"],
                            "status": "runtime_failure",
                            "failure_reason": f"cell_exception:{type(exc).__name__}",
                            "exception": str(exc),
                            "started_at": cell_start,
                            "ended_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            "repair_label": args.repair_label,
                        }
                    )
                    print(
                        f"END cell framework={framework} condition={condition_name} task_id={task.task_id} "
                        f"status=runtime_failure reason=cell_exception:{type(exc).__name__} run_json={failure_written['run_json']}",
                        flush=True,
                    )

    (run_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_analysis_reports(run_root, experiments_root, reports_root, args.comparison_test)
    _write_design_metadata(run_root, args, tasks)
    return run_root


def _parse_condition_columns(results_df: pd.DataFrame) -> pd.DataFrame:
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
        raise RuntimeError("No Coordination Suite mitigation records were found after execution.")

    results_df = _parse_condition_columns(results_df)
    results_df.to_csv(reports_root / "all_runs.csv", index=False)

    overall_stats = build_statistical_report(results_df, condition_column="framework_condition", failure_modes_column="primary_failure_modes")
    overall_stats["summary"].to_csv(reports_root / "overall_summary_framework_condition.csv", index=False)
    overall_stats["mast_modes"].to_csv(reports_root / "overall_mast_modes_framework_condition.csv", index=False)
    overall_stats["mast_categories"].to_csv(reports_root / "overall_mast_categories_framework_condition.csv", index=False)

    figures_root = reports_root / "figures_overall"
    figures_root.mkdir(parents=True, exist_ok=True)
    render_thesis_report(results_df, figures_root, condition_column="framework_condition", failure_modes_column="primary_failure_modes")

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
                comparison_rows.append({"framework": framework_name, "strategy": condition_name, "metric": metric_name, **comparison})

    comparisons_df = pd.DataFrame(comparison_rows)
    comparisons_df.to_csv(reports_root / "with_vs_without_mitigation_tests.csv", index=False)

    legacy_notice = {
        "status": "non_confirmatory_legacy",
        "reason": "Inline mitigation report is legacy convenience output and not the confirmatory Phase D H4 analysis.",
        "confirmatory_primary_outcome": "criteria_success",
        "confirmatory_latency_test": "mannwhitney",
        "confirmatory_reanalysis_script": "experiments/reanalyze_phase_d.py",
        "recommended_command": (
            "uv run python experiments/reanalyze_phase_d.py <phase_d_results_root> "
            "--output-dir <phase_d_results_root>/reports/thesis_clean_v1"
        ),
    }
    (reports_root / "non_confirmatory_legacy_notice.json").write_text(
        json.dumps(legacy_notice, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_design_metadata(run_root: Path, args: argparse.Namespace, tasks: list[Any]) -> None:
    metadata = {
        "benchmark": "coordination_suite",
        "design": {
            "frameworks": _normalize_frameworks(args.frameworks),
            "strategies": list(args.strategies),
            "num_runs": args.num_runs,
            "max_steps": args.max_steps,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "comparison_test": args.comparison_test,
            "mast_judge_enabled": args.mast_judge_enabled,
            "require_native_frameworks": args.require_native_frameworks,
            "thesis_strict": args.thesis_strict,
            "allow_smoke": args.allow_smoke,
            "cell_timeout_seconds": args.cell_timeout_seconds,
            "repair_missing_only": args.repair_missing_only,
            "repair_label": args.repair_label,
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
    (run_root / "mitigation_design.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.allow_smoke and args.cell_timeout_seconds is None:
        args.cell_timeout_seconds = 180.0
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

    if args.thesis_strict and not args.allow_smoke:
        if "metagpt" in selected_frameworks:
            raise ValueError("--thesis-strict does not allow metagpt. Use only langgraph/autogen/crewai for confirmatory Phase D.")
        if args.num_runs < 30:
            raise ValueError("--thesis-strict requires --num-runs >= 30 unless --allow-smoke is set.")
        if len(selected_frameworks) <= 1:
            raise ValueError("--thesis-strict requires more than one framework unless --allow-smoke is set.")
        if len(tasks) <= 1:
            raise ValueError("--thesis-strict requires more than one task unless --allow-smoke is set.")
        if len(args.strategies) <= 1:
            raise ValueError("--thesis-strict requires more than one mitigation strategy unless --allow-smoke is set.")

    run_root = run_coordination_mitigation(args, tasks)
    print(f"Coordination mitigation study finished. Artifacts saved to: {run_root}")


if __name__ == "__main__":
    main()