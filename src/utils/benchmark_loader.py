"""Benchmark task resolution helpers.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.base import BenchmarkTask
from benchmarks.coordination_suite import load_coordination_suite_tasks
from benchmarks.gaia import load_gaia_tasks
from benchmarks.swe_bench import load_swe_bench_verified_subset


def load_benchmark_tasks(benchmark_name: str, source_path: Path | str | None = None) -> list[BenchmarkTask]:
    """Resolve benchmark-specific task loaders from a benchmark name."""

    normalized_name = benchmark_name.strip().lower()
    if normalized_name in {"swe_bench", "swe_bench_verified", "swe-bench", "swe-bench-verified"}:
        return load_swe_bench_verified_subset(source_path)
    if normalized_name == "gaia":
        return load_gaia_tasks(source_path)
    if normalized_name in {"coordination_suite", "coordination-suite", "coordination"}:
        return load_coordination_suite_tasks(source_path)
    return []
