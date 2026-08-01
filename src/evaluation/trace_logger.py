"""Structured JSONL trace logging for multi-agent coordination experiments.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from frameworks.base_runner import RunMetrics, StepKind, TraceStep


class TraceEvent(BaseModel):
    """Canonical JSONL event structure stored on disk."""

    run_id: str
    step_index: int
    kind: StepKind
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: str | None = None
    content: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    token_count: int | None = None
    latency_ms: float | None = None
    state_snapshot: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class TraceContext:
    """Metadata for the active trace file."""

    run_id: str
    log_path: Path


class TraceLogger:
    """Append-only JSONL logger for all run artifacts."""

    def __init__(self, *, base_dir: Path | str = Path("results/traces")) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        *,
        framework_name: str,
        task_description: str,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> TraceContext:
        """Create the run log and write an initial metadata record."""

        active_run_id = run_id or str(uuid.uuid4())
        framework_dir = self.base_dir / framework_name
        framework_dir.mkdir(parents=True, exist_ok=True)
        log_path = framework_dir / f"{active_run_id}.jsonl"
        header = {
            "event_type": "run_start",
            "run_id": active_run_id,
            "framework": framework_name,
            "task_description": task_description,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return TraceContext(run_id=active_run_id, log_path=log_path)

    def log_event(
        self,
        *,
        run_id: str,
        step_index: int,
        kind: StepKind,
        role: str | None = None,
        content: str | None = None,
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_output: Any | None = None,
        token_count: int | None = None,
        latency_ms: float | None = None,
        state_snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Append a structured event to the active trace file."""

        event = TraceEvent(
            run_id=run_id,
            step_index=step_index,
            kind=kind,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            token_count=token_count,
            latency_ms=latency_ms,
            state_snapshot=state_snapshot,
            metadata=metadata or {},
        )
        self._append(run_id=run_id, event=event)
        return event

    def finish_run(
        self,
        *,
        run_id: str,
        success: bool,
        final_output: str,
        metrics: dict[str, Any],
    ) -> None:
        """Record a terminal event so downstream analysis can find the end state quickly."""

        terminal_event = {
            "event_type": "run_end",
            "run_id": run_id,
            "success": success,
            "final_output": final_output,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = self._resolve_path(run_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(terminal_event, ensure_ascii=False) + "\n")

    def read_trace(self, run_id: str) -> list[TraceStep]:
        """Load all step-level trace entries for a run."""

        path = self._resolve_path(run_id)
        if not path.exists():
            return []
        steps: list[TraceStep] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("event_type") in {"run_start", "run_end"}:
                    continue
                steps.append(TraceStep.model_validate(payload))
        return steps

    def _append(self, *, run_id: str, event: TraceEvent) -> None:
        path = self._resolve_path(run_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def _resolve_path(self, run_id: str) -> Path:
        for candidate in self.base_dir.glob(f"**/{run_id}.jsonl"):
            return candidate
        return self.base_dir / f"{run_id}.jsonl"
