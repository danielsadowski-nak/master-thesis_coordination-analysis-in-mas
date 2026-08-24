"""Multiple-comparison helpers for thesis statistical reporting."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def holm_adjust(p_values: Iterable[float | None]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""

    values = list(p_values)
    n = len(values)
    if n == 0:
        return []

    indexed = sorted(
        [(i, float(p) if p is not None and not np.isnan(p) else np.nan) for i, p in enumerate(values)],
        key=lambda item: (np.isnan(item[1]), item[1]),
    )
    adjusted = [np.nan] * n
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(indexed):
        if np.isnan(p_value):
            adjusted[original_index] = np.nan
            continue
        candidate = (n - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted


def add_holm_adjustment(df: pd.DataFrame, *, p_column: str = "p_value") -> pd.DataFrame:
    """Add Holm-adjusted p-values and significance labels to a comparison table."""

    if df.empty or p_column not in df.columns:
        return df.copy()

    out = df.copy()
    adjusted = holm_adjust(out[p_column].tolist())
    out["p_value_holm"] = adjusted
    out["significance_holm"] = [
        "***" if (isinstance(p, float) and p < 0.001) else "**" if (isinstance(p, float) and p < 0.01) else "*" if (isinstance(p, float) and p < 0.05) else "ns" if isinstance(p, float) and not np.isnan(p) else ""
        for p in adjusted
    ]
    return out
