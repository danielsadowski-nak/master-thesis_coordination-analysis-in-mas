"""Build an adjudication-ready table from two independently annotated reviewer sheets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
    "annotation_item_id",
    "framework",
    "benchmark",
    "run_index",
    "run_id",
    "raw_log_path",
    "success",
    "final_output",
]

LABEL_COLUMNS = [
    "manual_task_successful",
    "manual_primary_failure_modes",
    "manual_summary",
    "reviewer_id",
    "notes",
]


def _normalize_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
        .map({"true": True, "false": False})
    )


def _normalize_mode_string(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.split(";")
        .apply(lambda parts: sorted({p.strip() for p in parts if p.strip()}))
        .apply(lambda parts: "; ".join(parts))
    )


def _load_reviewer_sheet(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS + LABEL_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    keep = KEY_COLUMNS + LABEL_COLUMNS
    renamed = {col: f"{col}_{suffix}" for col in LABEL_COLUMNS}
    return df[keep].rename(columns=renamed)


def build_adjudication_table(reviewer1_csv: Path, reviewer2_csv: Path) -> pd.DataFrame:
    r1 = _load_reviewer_sheet(reviewer1_csv, "r1")
    r2 = _load_reviewer_sheet(reviewer2_csv, "r2")

    merged = r1.merge(r2, on=KEY_COLUMNS, how="inner", validate="one_to_one")

    r1_success = _normalize_bool(merged["manual_task_successful_r1"])
    r2_success = _normalize_bool(merged["manual_task_successful_r2"])
    r1_modes = _normalize_mode_string(merged["manual_primary_failure_modes_r1"])
    r2_modes = _normalize_mode_string(merged["manual_primary_failure_modes_r2"])

    r1_complete = r1_success.notna() & r1_modes.ne("")
    r2_complete = r2_success.notna() & r2_modes.ne("")
    merged["reviewer1_complete"] = r1_complete
    merged["reviewer2_complete"] = r2_complete
    merged["needs_annotation"] = ~(r1_complete & r2_complete)

    merged["task_success_disagreement"] = (r1_success.notna() & r2_success.notna() & (r1_success != r2_success))
    merged["mode_disagreement"] = (r1_modes != r2_modes) & ~(r1_modes.eq("") & r2_modes.eq(""))

    merged["needs_adjudication"] = (~merged["needs_annotation"]) & (
        merged["task_success_disagreement"] | merged["mode_disagreement"]
    )

    merged["adjudicated_task_successful"] = pd.NA
    merged["adjudicated_primary_failure_modes"] = ""
    merged["adjudicated_summary"] = ""

    return merged.sort_values(["framework", "benchmark", "run_index"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare adjudication table from two reviewer annotation sheets.")
    parser.add_argument("reviewer1_csv", type=Path, help="Path to reviewer 1 blinded annotation CSV.")
    parser.add_argument("reviewer2_csv", type=Path, help="Path to reviewer 2 blinded annotation CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Where to write the adjudication table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjudication = build_adjudication_table(args.reviewer1_csv, args.reviewer2_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    adjudication.to_csv(args.output_csv, index=False)

    print(f"Adjudication table written to: {args.output_csv}")
    print(f"Rows: {len(adjudication)}")
    print(f"Rows requiring annotation completion: {int(adjudication['needs_annotation'].sum())}")
    print(f"Rows requiring adjudication: {int(adjudication['needs_adjudication'].sum())}")


if __name__ == "__main__":
    main()
