"""Common runner interfaces and result models.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class StepKind(str, Enum):
    """Structured categories for trace entries."""

    PROMPT = "prompt"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE = "state"
    FINAL = "final"
    ERROR = "error"


class TraceStep(BaseModel):
    """Single structured event captured during an experiment run."""

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


class RunMetrics(BaseModel):
    """High-level metrics for a completed run."""

    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    steps_executed: int = 0
    cost_usd: float = 0.0


class TraceResult(BaseModel):
    """Return model for all runner implementations."""

    success: bool
    final_output: str
    full_trace: list[TraceStep]
    metrics: RunMetrics
    raw_log_path: str
    run_id: str | None = None
    runtime_mode: str | None = None
    is_valid_analytical: bool | None = None


class RunnerProtocol(Protocol):
    """Protocol for runner implementations."""

    def run_task(self, task_description: str, max_steps: int = 25) -> TraceResult:
        """Execute one task and return a structured trace result."""
        ...


class BaseMASRunner(ABC):
    """Abstract base class shared by all framework-specific runners."""

    def __init__(self, *, name: str, seed: int | None = None) -> None:
        self.name = name
        self.seed = seed

    @abstractmethod
    def run_task(self, task_description: str, max_steps: int = 25) -> TraceResult:
        """Execute a single benchmark task and return a TraceResult."""

        raise NotImplementedError

    def _build_empty_result(self, task_description: str, log_path: Path, reason: str) -> TraceResult:
        trace = [
            TraceStep(
                step_index=0,
                kind=StepKind.ERROR,
                content=reason,
                state_snapshot={"task_description": task_description},
            )
        ]
        return TraceResult(
            success=False,
            final_output=reason,
            full_trace=trace,
            metrics=RunMetrics(steps_executed=1),
            raw_log_path=str(log_path),
        )
