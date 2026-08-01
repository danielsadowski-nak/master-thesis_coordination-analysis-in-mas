"""Generate annotation templates and evaluate human-vs-judge agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.judge_alignment import (
    build_judge_validation_report,
    load_annotation_reference,
    write_annotation_template,
    write_judge_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotation templates and validate MAST judge agreement.")
    parser.add_argument("results_root", type=Path, help="Experiment root directory containing batch artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/judge_validation"))
    parser.add_argument("--annotations-csv", type=Path, default=None, help="Completed manual annotation CSV.")
    parser.add_argument("--sample-size", type=int, default=40, help="How many runs to include in the annotation template.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-model-only", action="store_true", help="Exclude scaffold and fallback runs from the generated annotation template.")
    parser.add_argument("--template-only", action="store_true", help="Only generate the annotation template and stop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template_path = args.output_dir / "annotation_template.csv"
    template = write_annotation_template(
        args.results_root,
        template_path,
        sample_size=args.sample_size,
        seed=args.seed,
        real_model_only=args.real_model_only,
    )
    print(f"Annotation template written to: {template_path}")
    print(f"Rows in template: {len(template)}")

    if args.template_only:
        return
    if args.annotations_csv is None:
        raise SystemExit("--annotations-csv is required unless --template-only is used.")

    annotations = load_annotation_reference(args.annotations_csv)
    report = build_judge_validation_report(annotations)
    written = write_judge_validation_report(report, args.output_dir)

    summary_path = written["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Judge validation summary written to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()