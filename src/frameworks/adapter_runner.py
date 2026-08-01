"""Shared scaffold for framework-specific adapters.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import random
import time
import uuid
from pathlib import Path
from typing import Any

from evaluation.trace_logger import TraceLogger
from frameworks.base_runner import BaseMASRunner, RunMetrics, StepKind, TraceResult, TraceStep
from utils.mitigations import MitigationStrategy, apply_mitigation_strategies


class FrameworkAdapterRunner(BaseMASRunner):
    """Reusable fallback runner for framework adapters.

    The class provides a deterministic, traceable scaffold that can later be
    replaced with native framework orchestration without changing downstream APIs.
    """

    def __init__(
        self,
        *,
        name: str,
        seed: int | None = None,
        trace_dir: Path | str = Path("results/traces"),
        trace_logger: TraceLogger | None = None,
        mitigation_strategies: tuple[MitigationStrategy, ...] = (),
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.trace_logger = trace_logger or TraceLogger(base_dir=trace_dir)
        self.mitigation_strategies = mitigation_strategies

    def run_task(self, task_description: str, max_steps: int = 25) -> TraceResult:
        """Execute a deterministic scaffold run with structured logging."""

        random.seed(self.seed)
        run_id = str(uuid.uuid4())
        context = self.trace_logger.start_run(
            framework_name=self.name,
            task_description=task_description,
            metadata={"max_steps": max_steps, "seed": self.seed, "adapter": self.name},
            run_id=run_id,
        )

        start_time = time.perf_counter()
        prompt = self._build_prompt(task_description=task_description, max_steps=max_steps)
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=0,
            kind=StepKind.PROMPT,
            role="system",
            content=prompt,
            state_snapshot={"task_description": task_description, "max_steps": max_steps},
        )

        response_text = self._generate_response(task_description=task_description, max_steps=max_steps)
        latency_seconds = time.perf_counter() - start_time
        token_estimate = max(1, len(prompt.split()) // 2)

        self.trace_logger.log_event(
            run_id=run_id,
            step_index=1,
            kind=StepKind.RESPONSE,
            role=self.name,
            content=response_text,
            latency_ms=latency_seconds * 1000.0,
            token_count=token_estimate,
            state_snapshot={"adapter": self.name, "latency_seconds": latency_seconds},
        )
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=2,
            kind=StepKind.FINAL,
            role="system",
            content=response_text,
            state_snapshot={"success": True, "adapter": self.name},
        )

        metrics = RunMetrics(
            latency_seconds=latency_seconds,
            prompt_tokens=token_estimate,
            completion_tokens=max(1, len(response_text.split()) // 2),
            total_tokens=token_estimate + max(1, len(response_text.split()) // 2),
            tool_calls=0,
            steps_executed=3,
            cost_usd=0.0,
        )
        full_trace = self.trace_logger.read_trace(run_id)
        self.trace_logger.finish_run(
            run_id=run_id,
            success=True,
            final_output=response_text,
            metrics=metrics.model_dump(),
        )
        return TraceResult(
            success=True,
            final_output=response_text,
            full_trace=full_trace,
            metrics=metrics,
            raw_log_path=str(context.log_path),
            run_id=run_id,
        )

    def _build_prompt(self, *, task_description: str, max_steps: int) -> str:
        return (
            f"Framework adapter scaffold for {self.name}.\n"
            f"Task: {task_description}\n"
            f"Max steps: {max_steps}\n"
            f"{apply_mitigation_strategies('', task_description, self.mitigation_strategies)}\n"
            "This adapter currently runs in scaffold mode until the framework-specific orchestration is connected."
        )

    def _generate_response(self, *, task_description: str, max_steps: int) -> str:
        return (
            f"{self.name} scaffold completed a reproducible placeholder run for the task. "
            f"The framework-specific integration can now be attached to this adapter without changing the API."
        )
