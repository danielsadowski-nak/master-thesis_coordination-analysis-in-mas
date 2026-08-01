"""Print a compact readiness snapshot for the Phase C annotation workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show concise Phase C readiness metrics.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/judge_validation_phase_c_cli_v3/agreement_balanced_k2"),
        help="Directory that contains QC and agreement status artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_repo_path(args.output_dir)

    qc_summary = _safe_read_json(output_dir / "annotation_qc_summary.json")
    workflow_status = _safe_read_json(output_dir / "agreement_workflow_status.json")
    missing_manifest = _safe_read_csv(output_dir / "annotation_missing_manifest.csv")

    reviewer_open: dict[str, int] = {}
    if not missing_manifest.empty and "reviewer" in missing_manifest.columns:
        counts = missing_manifest.groupby("reviewer").size().to_dict()
        reviewer_open = {str(k): int(v) for k, v in counts.items()}

    qc_pass = bool(qc_summary.get("qc_pass", False))
    unresolved_rows = int(workflow_status.get("n_unresolved", 0)) if workflow_status else 0
    resolved_rows = int(workflow_status.get("n_resolved", 0)) if workflow_status else 0
    adjudication_backlog = int(qc_summary.get("n_adjudication_backlog_rows", 0)) if qc_summary else 0

    readiness = {
        "qc_pass": qc_pass,
        "reviewer_open_items": reviewer_open,
        "adjudication_backlog_rows": adjudication_backlog,
        "resolved_rows": resolved_rows,
        "unresolved_rows": unresolved_rows,
        "strict_complete_mode": qc_summary.get("strict_complete_mode", None),
    }

    print("Phase C Readiness Snapshot")
    print(json.dumps(readiness, indent=2, ensure_ascii=False))

    # Non-zero exit if final strict-readiness is not reached.
    final_ready = qc_pass and adjudication_backlog == 0 and unresolved_rows == 0 and resolved_rows > 0
    if not final_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
