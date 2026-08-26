"""Quality-control checks for Phase C annotation files."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.mast_classifier import MASTFailureMode


REPO_ROOT = Path(__file__).resolve().parents[1]


ALLOWED_MODES = {mode.value for mode in MASTFailureMode}
NO_FAILURE_MODE_LABEL = "NO_FAILURE_MODE"


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@dataclass
class QCIssue:
    severity: str
    file: str
    row_index: int | None
    annotation_item_id: str | None
    column: str
    code: str
    message: str
    value: str


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


def _split_modes(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _is_nonempty(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(str(value).strip())


def _add_issue(issues: list[QCIssue], *, severity: str, file: Path, row_index: int | None, annotation_item_id: str | None, column: str, code: str, message: str, value: Any) -> None:
    issues.append(
        QCIssue(
            severity=severity,
            file=str(file),
            row_index=row_index,
            annotation_item_id=annotation_item_id,
            column=column,
            code=code,
            message=message,
            value="" if value is None else str(value),
        )
    )


def _validate_reviewer_sheet(path: Path, issues: list[QCIssue], reviewer_suffix: str | None = None) -> dict[str, int]:
    df = pd.read_csv(path)

    task_col = "manual_task_successful" if reviewer_suffix is None else f"manual_task_successful_{reviewer_suffix}"
    mode_col = "manual_primary_failure_modes" if reviewer_suffix is None else f"manual_primary_failure_modes_{reviewer_suffix}"
    summary_col = "manual_summary" if reviewer_suffix is None else f"manual_summary_{reviewer_suffix}"

    required = ["annotation_item_id", "framework", "benchmark", "run_id", task_col, mode_col, summary_col]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        _add_issue(
            issues,
            severity="error",
            file=path,
            row_index=None,
            annotation_item_id=None,
            column=",".join(missing_cols),
            code="missing_columns",
            message="Required columns are missing.",
            value=missing_cols,
        )
        return {"rows": int(len(df)), "complete_rows": 0}

    complete_rows = 0
    non_success_rows = 0
    for row_index, row in df.iterrows():
        item_id = str(row.get("annotation_item_id", ""))
        task_value = row.get(task_col)
        mode_value = row.get(mode_col)
        summary_value = row.get(summary_col)

        task_bool = _normalize_bool(task_value)
        modes = _split_modes(mode_value)
        has_summary = _is_nonempty(summary_value)

        no_mode_selected = len(modes) == 1 and modes[0] == NO_FAILURE_MODE_LABEL
        if NO_FAILURE_MODE_LABEL in modes and not no_mode_selected:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column=mode_col,
                code="invalid_no_mode_combination",
                message="NO_FAILURE_MODE must not be combined with other mode labels.",
                value=mode_value,
            )
            no_mode_selected = False

        if "success" in df.columns:
            success_value = _normalize_bool(row.get("success"))
            if success_value is not True:
                non_success_rows += 1

        if task_bool is None and not modes and not has_summary:
            continue

        if task_bool is None:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column=task_col,
                code="invalid_boolean",
                message="Task-success label must be true or false.",
                value=task_value,
            )

        if not modes and not no_mode_selected:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column=mode_col,
                code="missing_modes",
                message="Primary failure mode list is required for completed rows.",
                value=mode_value,
            )
        else:
            modes_for_validation = [] if no_mode_selected else modes
            invalid_modes = [mode for mode in modes_for_validation if mode not in ALLOWED_MODES]
            if invalid_modes:
                _add_issue(
                    issues,
                    severity="error",
                    file=path,
                    row_index=int(row_index),
                    annotation_item_id=item_id,
                    column=mode_col,
                    code="invalid_mode_label",
                    message="Mode label is not part of the canonical MAST taxonomy.",
                    value="; ".join(invalid_modes),
                )

        if not has_summary:
            _add_issue(
                issues,
                severity="warning",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column=summary_col,
                code="missing_summary",
                message="Evidence summary is empty.",
                value=summary_value,
            )

        if task_bool is not None and (modes or no_mode_selected):
            complete_rows += 1

    return {"rows": int(len(df)), "complete_rows": int(complete_rows), "non_success_rows": int(non_success_rows)}


def _metadata_by_item_id(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    meta_cols = ["framework", "benchmark", "run_index", "run_id", "raw_log_path", "success"]
    available_meta_cols = [col for col in meta_cols if col in df.columns]

    metadata: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        item_id = str(row.get("annotation_item_id", ""))
        metadata[item_id] = {col: row.get(col) for col in available_meta_cols}
    return metadata


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _validate_cross_file_integrity(
    reviewer1_path: Path,
    reviewer2_path: Path,
    adjudication_path: Path,
    issues: list[QCIssue],
) -> dict[str, int]:
    r1 = pd.read_csv(reviewer1_path)
    r2 = pd.read_csv(reviewer2_path)
    adj = pd.read_csv(adjudication_path)

    required = ["annotation_item_id", "framework", "benchmark", "run_index", "run_id", "raw_log_path"]
    for path, df in [(reviewer1_path, r1), (reviewer2_path, r2), (adjudication_path, adj)]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=None,
                annotation_item_id=None,
                column=",".join(missing),
                code="missing_integrity_columns",
                message="Required integrity columns are missing.",
                value=missing,
            )

    duplicates_count = 0
    for path, df in [(reviewer1_path, r1), (reviewer2_path, r2), (adjudication_path, adj)]:
        if "annotation_item_id" not in df.columns:
            continue
        dupes = df[df["annotation_item_id"].duplicated(keep=False)]
        if dupes.empty:
            continue
        duplicates_count += int(len(dupes))
        for row_index, row in dupes.iterrows():
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=str(row.get("annotation_item_id", "")),
                column="annotation_item_id",
                code="duplicate_annotation_item_id",
                message="annotation_item_id must be unique within each file.",
                value=row.get("annotation_item_id"),
            )

    id_set_mismatch = 0
    if "annotation_item_id" in r1.columns and "annotation_item_id" in r2.columns and "annotation_item_id" in adj.columns:
        r1_ids = set(r1["annotation_item_id"].astype(str))
        r2_ids = set(r2["annotation_item_id"].astype(str))
        adj_ids = set(adj["annotation_item_id"].astype(str))

        for missing_id in sorted(r1_ids - r2_ids):
            id_set_mismatch += 1
            _add_issue(
                issues,
                severity="error",
                file=reviewer2_path,
                row_index=None,
                annotation_item_id=missing_id,
                column="annotation_item_id",
                code="id_set_mismatch",
                message="Item present in reviewer1 but missing in reviewer2.",
                value=missing_id,
            )
        for missing_id in sorted(r2_ids - r1_ids):
            id_set_mismatch += 1
            _add_issue(
                issues,
                severity="error",
                file=reviewer1_path,
                row_index=None,
                annotation_item_id=missing_id,
                column="annotation_item_id",
                code="id_set_mismatch",
                message="Item present in reviewer2 but missing in reviewer1.",
                value=missing_id,
            )
        for missing_id in sorted((r1_ids | r2_ids) - adj_ids):
            id_set_mismatch += 1
            _add_issue(
                issues,
                severity="error",
                file=adjudication_path,
                row_index=None,
                annotation_item_id=missing_id,
                column="annotation_item_id",
                code="id_set_mismatch",
                message="Item present in reviewer sheets but missing in adjudication file.",
                value=missing_id,
            )

    metadata_mismatches = 0
    if all(col in r1.columns for col in ["annotation_item_id", "framework", "benchmark", "run_index", "run_id", "raw_log_path"]) and all(
        col in r2.columns for col in ["annotation_item_id", "framework", "benchmark", "run_index", "run_id", "raw_log_path"]
    ):
        r1_meta = _metadata_by_item_id(r1)
        r2_meta = _metadata_by_item_id(r2)
        adj_meta = _metadata_by_item_id(adj) if "annotation_item_id" in adj.columns else {}

        common_ids = set(r1_meta.keys()) & set(r2_meta.keys())
        for item_id in sorted(common_ids):
            cols = sorted(set(r1_meta[item_id].keys()) & set(r2_meta[item_id].keys()))
            for col in cols:
                left = _normalize_scalar(r1_meta[item_id].get(col))
                right = _normalize_scalar(r2_meta[item_id].get(col))
                if left != right:
                    metadata_mismatches += 1
                    _add_issue(
                        issues,
                        severity="error",
                        file=reviewer2_path,
                        row_index=None,
                        annotation_item_id=item_id,
                        column=col,
                        code="metadata_mismatch",
                        message="Reviewer sheets disagree on immutable metadata.",
                        value=f"r1={left} | r2={right}",
                    )

        common_adj_ids = common_ids & set(adj_meta.keys())
        for item_id in sorted(common_adj_ids):
            cols = sorted(set(r1_meta[item_id].keys()) & set(adj_meta[item_id].keys()))
            for col in cols:
                left = _normalize_scalar(r1_meta[item_id].get(col))
                right = _normalize_scalar(adj_meta[item_id].get(col))
                if left != right:
                    metadata_mismatches += 1
                    _add_issue(
                        issues,
                        severity="error",
                        file=adjudication_path,
                        row_index=None,
                        annotation_item_id=item_id,
                        column=col,
                        code="metadata_mismatch",
                        message="Adjudication file differs from reviewer metadata.",
                        value=f"reviewer={left} | adjudication={right}",
                    )

    return {
        "duplicate_item_rows": int(duplicates_count),
        "id_set_mismatch_count": int(id_set_mismatch),
        "metadata_mismatch_count": int(metadata_mismatches),
    }


def _validate_adjudication_table(path: Path, issues: list[QCIssue]) -> dict[str, int]:
    df = pd.read_csv(path)
    required = [
        "annotation_item_id",
        "needs_annotation",
        "needs_adjudication",
        "adjudicated_task_successful",
        "adjudicated_primary_failure_modes",
    ]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        _add_issue(
            issues,
            severity="error",
            file=path,
            row_index=None,
            annotation_item_id=None,
            column=",".join(missing_cols),
            code="missing_columns",
            message="Required adjudication columns are missing.",
            value=missing_cols,
        )
        return {"rows": int(len(df)), "rows_needing_adjudication": 0}

    rows_needing_adjudication = 0
    for row_index, row in df.iterrows():
        item_id = str(row.get("annotation_item_id", ""))
        needs_annotation = bool(row.get("needs_annotation", False))
        needs_adjudication = bool(row.get("needs_adjudication", False))
        if needs_annotation:
            continue
        if not needs_adjudication:
            continue

        rows_needing_adjudication += 1
        adjudicated_success = _normalize_bool(row.get("adjudicated_task_successful"))
        adjudicated_modes = _split_modes(row.get("adjudicated_primary_failure_modes"))
        if adjudicated_success is None:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column="adjudicated_task_successful",
                code="missing_adjudicated_success",
                message="Adjudicated task-success label is required for disagreement rows.",
                value=row.get("adjudicated_task_successful"),
            )

        no_mode_selected = len(adjudicated_modes) == 1 and adjudicated_modes[0] == NO_FAILURE_MODE_LABEL
        if NO_FAILURE_MODE_LABEL in adjudicated_modes and not no_mode_selected:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column="adjudicated_primary_failure_modes",
                code="invalid_adjudicated_no_mode_combination",
                message="NO_FAILURE_MODE must not be combined with other adjudicated mode labels.",
                value=row.get("adjudicated_primary_failure_modes"),
            )

        if not adjudicated_modes:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column="adjudicated_primary_failure_modes",
                code="missing_adjudicated_modes",
                message="Adjudicated mode labels are required for disagreement rows.",
                value=row.get("adjudicated_primary_failure_modes"),
            )
            continue

        if adjudicated_success is True:
            if not no_mode_selected:
                _add_issue(
                    issues,
                    severity="error",
                    file=path,
                    row_index=int(row_index),
                    annotation_item_id=item_id,
                    column="adjudicated_primary_failure_modes",
                    code="adjudicated_success_requires_no_failure_mode",
                    message="Adjudicated success rows must use NO_FAILURE_MODE as the sole mode label.",
                    value=row.get("adjudicated_primary_failure_modes"),
                )
            continue

        if no_mode_selected:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column="adjudicated_primary_failure_modes",
                code="adjudicated_failure_cannot_use_no_failure_mode",
                message="Adjudicated failure rows must use canonical MAST mode labels, not NO_FAILURE_MODE.",
                value=row.get("adjudicated_primary_failure_modes"),
            )
            continue

        invalid_modes = [mode for mode in adjudicated_modes if mode not in ALLOWED_MODES]
        if invalid_modes:
            _add_issue(
                issues,
                severity="error",
                file=path,
                row_index=int(row_index),
                annotation_item_id=item_id,
                column="adjudicated_primary_failure_modes",
                code="invalid_adjudicated_mode_label",
                message="Adjudicated mode label is not part of the canonical MAST taxonomy.",
                value="; ".join(invalid_modes),
            )

    return {"rows": int(len(df)), "rows_needing_adjudication": int(rows_needing_adjudication)}


def _build_missing_manifest(path: Path, reviewer_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        task = _normalize_bool(row.get("manual_task_successful"))
        modes = _split_modes(row.get("manual_primary_failure_modes"))
        summary_ok = _is_nonempty(row.get("manual_summary"))

        if task is not None and modes and summary_ok:
            continue

        missing_fields: list[str] = []
        if task is None:
            missing_fields.append("manual_task_successful")
        if not modes:
            missing_fields.append("manual_primary_failure_modes")
        if not summary_ok:
            missing_fields.append("manual_summary")

        rows.append(
            {
                "reviewer": reviewer_tag,
                "annotation_item_id": row.get("annotation_item_id", ""),
                "framework": row.get("framework", ""),
                "benchmark": row.get("benchmark", ""),
                "run_id": row.get("run_id", ""),
                "raw_log_path": row.get("raw_log_path", ""),
                "missing_fields": "; ".join(missing_fields),
            }
        )

    return pd.DataFrame(rows)


def _build_adjudication_backlog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "needs_adjudication" not in df.columns:
        return pd.DataFrame(columns=["annotation_item_id", "framework", "benchmark", "run_id", "reason"])

    backlog = df[df["needs_adjudication"].astype(bool)].copy()
    if backlog.empty:
        return pd.DataFrame(columns=["annotation_item_id", "framework", "benchmark", "run_id", "reason"])

    return backlog.assign(reason="reviewer_disagreement")[["annotation_item_id", "framework", "benchmark", "run_id", "reason"]]


def _build_progress_matrix(path: Path, reviewer_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for (framework, benchmark), group in df.groupby(["framework", "benchmark"], dropna=False):
        completed = 0
        for _, row in group.iterrows():
            task = _normalize_bool(row.get("manual_task_successful"))
            modes = _split_modes(row.get("manual_primary_failure_modes"))
            summary_ok = _is_nonempty(row.get("manual_summary"))
            if task is not None and modes and summary_ok:
                completed += 1

        total = int(len(group))
        pending = int(total - completed)
        rows.append(
            {
                "reviewer": reviewer_tag,
                "framework": framework,
                "benchmark": benchmark,
                "total_rows": total,
                "completed_rows": int(completed),
                "pending_rows": pending,
                "completion_rate": round((completed / total) if total else 0.0, 6),
            }
        )

    return pd.DataFrame(rows).sort_values(["reviewer", "framework", "benchmark"]).reset_index(drop=True)


def _stable_score(seed: int, reviewer: str, annotation_item_id: str) -> int:
    key = f"{seed}|{reviewer}|{annotation_item_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:16], 16)


def _build_annotation_worklist(missing_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if missing_df.empty:
        return pd.DataFrame(
            columns=[
                "reviewer",
                "sequence",
                "annotation_item_id",
                "framework",
                "benchmark",
                "run_id",
                "raw_log_path",
                "missing_fields",
            ]
        )

    rows: list[dict[str, Any]] = []
    for reviewer, reviewer_df in missing_df.groupby("reviewer", dropna=False):
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for _, row in reviewer_df.iterrows():
            key = (str(row.get("framework", "")), str(row.get("benchmark", "")))
            groups.setdefault(key, []).append(row.to_dict())

        for key in groups:
            groups[key].sort(
                key=lambda r: _stable_score(
                    seed,
                    str(reviewer),
                    str(r.get("annotation_item_id", "")),
                )
            )

        ordered_keys = sorted(groups.keys())
        sequence = 1
        remaining = True
        while remaining:
            remaining = False
            for key in ordered_keys:
                bucket = groups[key]
                if not bucket:
                    continue
                remaining = True
                current = bucket.pop(0)
                rows.append(
                    {
                        "reviewer": reviewer,
                        "sequence": sequence,
                        "annotation_item_id": current.get("annotation_item_id", ""),
                        "framework": current.get("framework", ""),
                        "benchmark": current.get("benchmark", ""),
                        "run_id": current.get("run_id", ""),
                        "raw_log_path": current.get("raw_log_path", ""),
                        "missing_fields": current.get("missing_fields", ""),
                    }
                )
                sequence += 1

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quality-control checks for Phase C annotation assets.")
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=None,
        help="Optional root folder override (e.g., results/judge_validation_phase_c_v4). Overrides reviewer/adjudication/output defaults.",
    )
    parser.add_argument(
        "--reviewer1-csv",
        type=Path,
        # Kept on v3 paths for backward compatibility; prefer --sample-root for current v4 runs.
        default=Path("results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded.csv"),
    )
    parser.add_argument(
        "--reviewer2-csv",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/annotation_sheet_balanced_k2_blinded_reviewer2.csv"),
    )
    parser.add_argument(
        "--adjudication-csv",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/annotation_adjudication_balanced_k2.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/agreement_balanced_k2"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="If set, do not fail QC when reviewer sheets are only partially completed.",
    )
    parser.add_argument(
        "--worklist-seed",
        type=int,
        default=20260801,
        help="Deterministic seed for generated annotation worklists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success_only_manifest = False
    if args.sample_root is not None:
        sample_root = _resolve_repo_path(args.sample_root)
        args.reviewer1_csv = sample_root / "annotation_sheet_blinded_reviewer1.csv"
        args.reviewer2_csv = sample_root / "annotation_sheet_blinded_reviewer2.csv"
        args.adjudication_csv = sample_root / "annotation_adjudication.csv"
        args.output_dir = sample_root / "agreement"
        manifest_path = sample_root / "annotation_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                success_only_manifest = bool(manifest.get("success_only", False))
            except Exception:
                success_only_manifest = False

    args.reviewer1_csv = _resolve_repo_path(args.reviewer1_csv)
    args.reviewer2_csv = _resolve_repo_path(args.reviewer2_csv)
    args.adjudication_csv = _resolve_repo_path(args.adjudication_csv)
    args.output_dir = _resolve_repo_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    issues: list[QCIssue] = []

    reviewer1_stats = _validate_reviewer_sheet(args.reviewer1_csv, issues)
    reviewer2_stats = _validate_reviewer_sheet(args.reviewer2_csv, issues)

    if success_only_manifest:
        if reviewer1_stats.get("non_success_rows", 0) > 0:
            _add_issue(
                issues,
                severity="error",
                file=args.reviewer1_csv,
                row_index=None,
                annotation_item_id=None,
                column="success",
                code="non_success_row_in_success_only_sample",
                message="Manifest marks success_only=true, but reviewer sheet contains failure rows.",
                value=reviewer1_stats.get("non_success_rows"),
            )
        if reviewer2_stats.get("non_success_rows", 0) > 0:
            _add_issue(
                issues,
                severity="error",
                file=args.reviewer2_csv,
                row_index=None,
                annotation_item_id=None,
                column="success",
                code="non_success_row_in_success_only_sample",
                message="Manifest marks success_only=true, but reviewer sheet contains failure rows.",
                value=reviewer2_stats.get("non_success_rows"),
            )

    adjudication_exists = args.adjudication_csv.exists()
    if adjudication_exists:
        adjudication_stats = _validate_adjudication_table(args.adjudication_csv, issues)
        integrity_stats = _validate_cross_file_integrity(
            args.reviewer1_csv,
            args.reviewer2_csv,
            args.adjudication_csv,
            issues,
        )
    else:
        if args.allow_incomplete:
            _add_issue(
                issues,
                severity="warning",
                file=args.adjudication_csv,
                row_index=None,
                annotation_item_id=None,
                column="adjudication_csv",
                code="missing_adjudication_file",
                message="Adjudication CSV not found; cross-file adjudication checks were skipped in allow-incomplete mode.",
                value=args.adjudication_csv,
            )
            adjudication_stats = {
                "rows": 0,
                "complete_rows": 0,
                "resolved_rows": 0,
                "needs_annotation_rows": 0,
                "needs_adjudication_rows": 0,
            }
            integrity_stats = {
                "metadata_mismatches": 0,
                "duplicates": 0,
                "id_set_mismatch": 0,
                "source_hash_mismatches": 0,
                "missing_adjudication_file": 1,
            }
        else:
            raise FileNotFoundError(
                f"Adjudication CSV not found: {args.adjudication_csv}. "
                "Create it first with experiments/prepare_phase_c_adjudication.py or run with --allow-incomplete for smoke checks."
            )

    missing_r1_df = _build_missing_manifest(args.reviewer1_csv, "reviewer1")
    missing_r2_df = _build_missing_manifest(args.reviewer2_csv, "reviewer2")
    missing_annotations_df = pd.concat([missing_r1_df, missing_r2_df], ignore_index=True)
    missing_annotations_path = args.output_dir / "annotation_missing_manifest.csv"
    if missing_annotations_df.empty:
        missing_annotations_df = pd.DataFrame(
            columns=["reviewer", "annotation_item_id", "framework", "benchmark", "run_id", "raw_log_path", "missing_fields"]
        )
    missing_annotations_df.to_csv(missing_annotations_path, index=False)

    progress_matrix_df = pd.concat(
        [
            _build_progress_matrix(args.reviewer1_csv, "reviewer1"),
            _build_progress_matrix(args.reviewer2_csv, "reviewer2"),
        ],
        ignore_index=True,
    )
    progress_matrix_path = args.output_dir / "annotation_progress_matrix.csv"
    progress_matrix_df.to_csv(progress_matrix_path, index=False)

    worklist_df = _build_annotation_worklist(missing_annotations_df, args.worklist_seed)
    worklist_path = args.output_dir / "annotation_worklist.csv"
    worklist_df.to_csv(worklist_path, index=False)

    if adjudication_exists:
        adjudication_backlog_df = _build_adjudication_backlog(args.adjudication_csv)
    else:
        adjudication_backlog_df = pd.DataFrame(
            columns=[
                "annotation_item_id",
                "framework",
                "benchmark",
                "run_id",
                "raw_log_path",
                "needs_annotation",
                "needs_adjudication",
            ]
        )
    adjudication_backlog_path = args.output_dir / "adjudication_backlog_manifest.csv"
    adjudication_backlog_df.to_csv(adjudication_backlog_path, index=False)

    if not args.allow_incomplete:
        if reviewer1_stats["complete_rows"] < reviewer1_stats["rows"]:
            _add_issue(
                issues,
                severity="error",
                file=args.reviewer1_csv,
                row_index=None,
                annotation_item_id=None,
                column="manual_task_successful/manual_primary_failure_modes",
                code="incomplete_reviewer_sheet",
                message="Reviewer 1 sheet is incomplete.",
                value=f"{reviewer1_stats['complete_rows']}/{reviewer1_stats['rows']} completed",
            )
        if reviewer2_stats["complete_rows"] < reviewer2_stats["rows"]:
            _add_issue(
                issues,
                severity="error",
                file=args.reviewer2_csv,
                row_index=None,
                annotation_item_id=None,
                column="manual_task_successful/manual_primary_failure_modes",
                code="incomplete_reviewer_sheet",
                message="Reviewer 2 sheet is incomplete.",
                value=f"{reviewer2_stats['complete_rows']}/{reviewer2_stats['rows']} completed",
            )

    issues_df = pd.DataFrame([asdict(issue) for issue in issues])
    issues_path = args.output_dir / "annotation_qc_issues.csv"
    if issues_df.empty:
        issues_df = pd.DataFrame(
            columns=[
                "severity",
                "file",
                "row_index",
                "annotation_item_id",
                "column",
                "code",
                "message",
                "value",
            ]
        )
    issues_df.to_csv(issues_path, index=False)

    n_error = int((issues_df["severity"] == "error").sum()) if not issues_df.empty else 0
    summary = {
        "reviewer1": reviewer1_stats,
        "reviewer2": reviewer2_stats,
        "adjudication": adjudication_stats,
        "integrity": integrity_stats,
        "n_missing_manifest_rows": int(len(missing_annotations_df)),
        "n_worklist_rows": int(len(worklist_df)),
        "n_adjudication_backlog_rows": int(len(adjudication_backlog_df)),
        "n_issues_total": int(len(issues_df)),
        "n_issues_error": n_error,
        "n_issues_warning": int((issues_df["severity"] == "warning").sum()) if not issues_df.empty else 0,
        "qc_pass": bool(n_error == 0),
        "strict_complete_mode": bool(not args.allow_incomplete),
        "allowed_mast_modes": sorted(ALLOWED_MODES),
    }
    summary_path = args.output_dir / "annotation_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"QC summary written to: {summary_path}")
    print(json.dumps(summary, indent=2))
    print(f"QC issues written to: {issues_path}")
    print(f"Missing-annotation manifest written to: {missing_annotations_path}")
    print(f"Progress matrix written to: {progress_matrix_path}")
    print(f"Annotation worklist written to: {worklist_path}")
    print(f"Adjudication backlog manifest written to: {adjudication_backlog_path}")

    if not summary["qc_pass"]:
        raise SystemExit("QC failed: resolve annotation errors before running final agreement analysis.")


if __name__ == "__main__":
    main()
