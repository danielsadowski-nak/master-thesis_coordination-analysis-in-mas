"""Tests for experiment plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.plots import export_batch_summary, load_batch_artifacts, render_experiment_reports


def test_render_experiment_reports_creates_figures(tmp_path: Path) -> None:
    batch_dir = tmp_path / "gaia" / "langgraph"
    batch_dir.mkdir(parents=True)
    (batch_dir / "summary.json").write_text(
        json.dumps(
            {
                "framework": "langgraph",
                "benchmark": "gaia",
                "num_runs": 2,
                "success_rate": 0.5,
                "mean_latency_seconds": 1.2,
                "mean_total_tokens": 120,
                "mean_cost_usd": 0.01,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"latency_seconds": 1.0},
            {"latency_seconds": 1.4},
        ]
    ).to_csv(batch_dir / "records.csv", index=False)
    (batch_dir / "run_000.json").write_text(
        json.dumps({"judgement": {"primary_failure_modes": ["3.2 Weak Verification"]}}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "figures"
    files = render_experiment_reports(batch_dir, output_dir)

    assert len(files) == 3
    assert all(path.exists() for path in files)


def test_export_batch_summary_creates_csv(tmp_path: Path) -> None:
    batch_dir = tmp_path / "swe" / "crewai"
    batch_dir.mkdir(parents=True)
    (batch_dir / "summary.json").write_text(
        json.dumps({"framework": "crewai", "benchmark": "swe_bench_verified"}),
        encoding="utf-8",
    )

    artifacts = load_batch_artifacts(batch_dir)
    output_dir = tmp_path / "summary"
    output_dir.mkdir()
    summary_path = export_batch_summary(artifacts, output_dir)

    assert summary_path.exists()