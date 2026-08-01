"""Render plots and summary artifacts from an experiment batch.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.plots import export_batch_summary, load_batch_artifacts, render_experiment_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render plots for a completed experiment batch.")
    parser.add_argument("batch_dir", type=Path, help="Path to the benchmark/framework batch directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = load_batch_artifacts(args.batch_dir)
    output_dir = args.output_dir or args.batch_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_experiment_reports(args.batch_dir, output_dir)
    export_batch_summary(artifacts, output_dir)


if __name__ == "__main__":
    main()
