"""Tests for validation script helpers."""

from __future__ import annotations

from experiments.run_validation import build_summary, summarize_primary_modes


def test_summarize_primary_modes_counts_modes() -> None:
    payloads = [
        {"judgement": {"primary_failure_modes": ["2.4 Information Withholding", "3.2 Weak Verification"]}},
        {"judgement": {"primary_failure_modes": ["3.2 Weak Verification"]}},
    ]
    top_modes = summarize_primary_modes(payloads)

    assert top_modes[0][0] == "3.2 Weak Verification"
    assert top_modes[0][1] == 2


def test_build_summary_works_with_minimal_input() -> None:
    summary = build_summary(
        task_runs=[{"runs_per_task": 3}],
        all_records=[
            {"success": True, "latency_seconds": 1.2, "total_tokens": 20, "cost_usd": 0.0},
            {"success": False, "latency_seconds": 1.8, "total_tokens": 24, "cost_usd": 0.0},
        ],
        all_run_payloads=[{"judgement": {"primary_failure_modes": ["3.2 Weak Verification"]}}],
    )

    assert summary["tasks"] == 1
    assert summary["runs_per_task"] == 3
    assert summary["total_runs"] == 2
    assert summary["success_rate"] == 0.5
