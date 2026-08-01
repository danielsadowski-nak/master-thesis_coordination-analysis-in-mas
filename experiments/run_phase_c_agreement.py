"""Run a reproducible Phase C agreement workflow from adjudication-ready inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.judge_alignment import build_judge_validation_report, write_judge_validation_report


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _normalize_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _normalize_modes(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    if not parts:
        return ""
    return "; ".join(sorted(set(parts)))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _resolve_reference_row(row: pd.Series) -> tuple[bool | None, str, str]:
    """Return reference task success, modes, and resolution source for one row."""

    adjudicated_success = _normalize_bool(row.get("adjudicated_task_successful"))
    adjudicated_modes = _normalize_modes(row.get("adjudicated_primary_failure_modes"))
    if adjudicated_success is not None and adjudicated_modes:
        return adjudicated_success, adjudicated_modes, "adjudicated"

    r1_success = _normalize_bool(row.get("manual_task_successful_r1"))
    r2_success = _normalize_bool(row.get("manual_task_successful_r2"))
    r1_modes = _normalize_modes(row.get("manual_primary_failure_modes_r1"))
    r2_modes = _normalize_modes(row.get("manual_primary_failure_modes_r2"))

    if r1_success is None or r2_success is None or not r1_modes or not r2_modes:
        return None, "", "missing_annotations"

    if r1_success == r2_success and r1_modes == r2_modes:
        return r1_success, r1_modes, "reviewer_consensus"

    return None, "", "unresolved_disagreement"


def _build_resolved_frame(adjudication_df: pd.DataFrame, master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    join_keys = ["framework", "benchmark", "run_index", "run_id", "raw_log_path"]
    needed_master_cols = join_keys + ["judge_task_successful", "judge_primary_failure_modes", "judge_summary"]
    missing_master = [col for col in needed_master_cols if col not in master_df.columns]
    if missing_master:
        raise KeyError(f"Master file is missing columns: {missing_master}")

    merged = adjudication_df.merge(master_df[needed_master_cols], on=join_keys, how="left", validate="one_to_one")

    resolved_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    status_counter = {
        "adjudicated": 0,
        "reviewer_consensus": 0,
        "missing_annotations": 0,
        "unresolved_disagreement": 0,
    }

    for _, row in merged.iterrows():
        reference_success, reference_modes, source = _resolve_reference_row(row)
        status_counter[source] += 1

        base = {
            "framework": row.get("framework"),
            "benchmark": row.get("benchmark"),
            "run_index": row.get("run_index"),
            "run_id": row.get("run_id"),
            "raw_log_path": row.get("raw_log_path"),
            "judge_task_successful": row.get("judge_task_successful"),
            "judge_primary_failure_modes": row.get("judge_primary_failure_modes"),
            "manual_task_successful": reference_success,
            "manual_primary_failure_modes": reference_modes,
            "manual_summary": row.get("adjudicated_summary", "") if source == "adjudicated" else "",
            "adjudicated_task_successful": row.get("adjudicated_task_successful", ""),
            "adjudicated_primary_failure_modes": row.get("adjudicated_primary_failure_modes", ""),
            "adjudicated_summary": row.get("adjudicated_summary", ""),
            "notes": row.get("notes_r1", ""),
            "reference_source": source,
        }

        if reference_success is None or not reference_modes:
            unresolved_rows.append(base)
        else:
            resolved_rows.append(base)

    return pd.DataFrame(resolved_rows), pd.DataFrame(unresolved_rows), status_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Phase C judge agreement from adjudication tables.")
    parser.add_argument(
        "--reviewer1-csv",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded.csv"),
        help="Reviewer 1 blinded annotation CSV.",
    )
    parser.add_argument(
        "--reviewer2-csv",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded_reviewer2.csv"),
        help="Reviewer 2 blinded annotation CSV.",
    )
    parser.add_argument(
        "--adjudication-csv",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/annotation_adjudication_balanced_k2.csv"),
        help="Adjudication-ready CSV produced from two reviewer sheets.",
    )
    parser.add_argument(
        "--master-csv",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/annotation_master_balanced_k2.csv"),
        help="Master CSV with judge columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/agreement_balanced_k2"),
        help="Where to write intermediate and final agreement artifacts.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="If set, compute a preliminary agreement report from resolved rows even if unresolved rows remain.",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip annotation QC preflight (not recommended for final reporting).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reviewer1_csv = _resolve_repo_path(args.reviewer1_csv)
    args.reviewer2_csv = _resolve_repo_path(args.reviewer2_csv)
    args.adjudication_csv = _resolve_repo_path(args.adjudication_csv)
    args.master_csv = _resolve_repo_path(args.master_csv)
    args.output_dir = _resolve_repo_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_qc:
        qc_script = REPO_ROOT / "experiments" / "check_phase_c_annotation_quality.py"
        qc_cmd = [
            sys.executable,
            str(qc_script),
            "--reviewer1-csv",
            str(args.reviewer1_csv),
            "--reviewer2-csv",
            str(args.reviewer2_csv),
            "--adjudication-csv",
            str(args.adjudication_csv),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.allow_partial:
            qc_cmd.append("--allow-incomplete")

        qc_result = subprocess.run(qc_cmd, capture_output=True, text=True)
        if qc_result.stdout:
            print(qc_result.stdout.strip())
        if qc_result.stderr:
            print(qc_result.stderr.strip(), file=sys.stderr)
        if qc_result.returncode != 0:
            raise SystemExit("QC preflight failed. Resolve annotation issues before agreement analysis.")

    adjudication_df = _read_csv(args.adjudication_csv)
    master_df = _read_csv(args.master_csv)

    resolved_df, unresolved_df, status_counter = _build_resolved_frame(adjudication_df, master_df)

    resolved_path = args.output_dir / "resolved_annotations.csv"
    unresolved_path = args.output_dir / "unresolved_annotations.csv"
    status_path = args.output_dir / "agreement_workflow_status.json"

    resolved_df.to_csv(resolved_path, index=False)
    unresolved_df.to_csv(unresolved_path, index=False)

    status_payload = {
        "n_total": int(len(adjudication_df)),
        "n_resolved": int(len(resolved_df)),
        "n_unresolved": int(len(unresolved_df)),
        **{f"n_{key}": int(value) for key, value in status_counter.items()},
    }
    status_path.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Status written to: {status_path}")
    print(json.dumps(status_payload, indent=2))
    print(f"Resolved rows written to: {resolved_path}")
    print(f"Unresolved rows written to: {unresolved_path}")

    if unresolved_df.empty or args.allow_partial:
        if resolved_df.empty:
            raise SystemExit("No resolved rows available for agreement analysis.")

        report = build_judge_validation_report(
            resolved_df.assign(
                reference_task_successful=resolved_df["manual_task_successful"],
                reference_primary_failure_modes=resolved_df["manual_primary_failure_modes"],
            )
        )
        written = write_judge_validation_report(report, args.output_dir)
        print(f"Agreement summary written to: {written['summary']}")
        return

    raise SystemExit(
        "Unresolved rows remain. Complete missing annotations and adjudication first, "
        "or rerun with --allow-partial for a preliminary report."
    )


if __name__ == "__main__":
    main()
