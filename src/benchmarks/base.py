"""Shared benchmark abstractions and file loading helpers.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    """Canonical benchmark task representation."""

    task_id: str
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def ensure_path(value: Path | str | None) -> Path | None:
    """Normalize an optional path-like value."""

    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def load_records(source_path: Path) -> list[dict[str, Any]]:
    """Load records from JSON, JSONL, CSV, or plain text."""

    if not source_path.exists():
        return []

    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return [record for record in payload["data"] if isinstance(record, dict)]
            return [payload]
        return []
    if suffix == ".csv":
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    return [{"text": line} for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_non_empty(record: dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    """Return the first non-empty string value from a record."""

    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def metadata_without(record: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return a metadata dict without prompt fields."""

    excluded = set(keys)
    return {key: value for key, value in record.items() if key not in excluded}
