"""CrewAI runner with an optional native integration.

Generated with GitHub Copilot assistance - reviewed and adapted by author.

Optional dependency installation:
    pip install crewai
"""

from __future__ import annotations

import random
import time
import uuid
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any

from evaluation.trace_logger import TraceLogger
from frameworks.adapter_runner import FrameworkAdapterRunner
from frameworks.base_runner import RunMetrics, StepKind, TraceResult
from utils.config import apply_llm_runtime_environment
from utils.langsmith import LangSmithRuntime
from utils.mitigations import MitigationStrategy, apply_mitigation_strategies


class CrewAIRunner(FrameworkAdapterRunner):
    """CrewAI runner with a native CrewAI execution path."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        trace_dir: Path | str = Path("results/traces"),
        trace_logger: TraceLogger | None = None,
        mitigation_strategies: tuple[MitigationStrategy, ...] = (),
        langsmith_project: str = "mas-coordination-analysis",
        langsmith_enabled: bool = True,
        langsmith_api_key: str | None = None,
        langsmith_endpoint: str | None = None,
        langsmith_tags: tuple[str, ...] = (),
        llm: Any | None = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        process: str = "sequential",
        manager_llm: Any | None = None,
    ) -> None:
        super().__init__(
            name="crewai",
            seed=seed,
            trace_dir=trace_dir,
            trace_logger=trace_logger,
            mitigation_strategies=mitigation_strategies,
        )
        self.llm = llm
        self.model_name = model_name
        self.temperature = temperature
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.process = process
        self.manager_llm = manager_llm
        self.langsmith_runtime = LangSmithRuntime(
            enabled=langsmith_enabled,
            project=langsmith_project,
            api_key=langsmith_api_key,
            endpoint=langsmith_endpoint,
            tags=langsmith_tags,
        )
        apply_llm_runtime_environment(api_key=llm_api_key, base_url=llm_base_url)
        self.langsmith_runtime.apply()

    def run_task(self, task_description: str, max_steps: int = 25) -> TraceResult:
        """Run a task via native CrewAI and fall back to the scaffold when unavailable."""

        crew_components = self._load_crewai_components()
        if crew_components is None:
            return super().run_task(task_description, max_steps=max_steps)

        if self.llm is None:
            self.llm = self._build_llm()
        if self.llm is None:
            return super().run_task(task_description, max_steps=max_steps)

        Agent, Crew, Process, Task, LLM = crew_components

        random.seed(self.seed)
        run_id = str(uuid.uuid4())
        context = self.trace_logger.start_run(
            framework_name=self.name,
            task_description=task_description,
            metadata={"max_steps": max_steps, "seed": self.seed, "mode": "native", "framework": "CrewAI"},
            run_id=run_id,
        )
        start_time = time.perf_counter()

        system_prompt = self._build_system_prompt(task_description, max_steps)
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=0,
            kind=StepKind.PROMPT,
            role="system",
            content=system_prompt,
            state_snapshot={"task_description": task_description, "max_steps": max_steps, "mode": "native"},
        )

        planner = Agent(
            role="Planner",
            goal="Decompose the task and maintain coordination discipline.",
            backstory="An expert coordinator for reproducible multi-agent experiments.",
            llm=self.llm,
            verbose=False,
            allow_delegation=True,
        )
        executor = Agent(
            role="Executor",
            goal="Solve the task while following the planner's coordination constraints.",
            backstory="A disciplined implementation agent focused on reliable outputs.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

        planning_task = Task(
            description=(
                f"{system_prompt}\n\n"
                f"Task:\n{task_description}\n\n"
                "Produce a concise execution plan with explicit verification checkpoints."
            ),
            expected_output="A short plan with steps, risks, and verification points.",
            agent=planner,
        )
        execution_task = Task(
            description=(
                f"{system_prompt}\n\n"
                f"Task:\n{task_description}\n\n"
                "Execute the plan and provide the final answer in a structured, concise form."
            ),
            expected_output="A final answer and a short self-check.",
            agent=executor,
        )

        process_value = getattr(Process, self.process, Process.sequential)
        crew_kwargs: dict[str, Any] = {
            "agents": [planner, executor],
            "tasks": [planning_task, execution_task],
            "process": process_value,
            "verbose": False,
        }
        if process_value == getattr(Process, "hierarchical", None):
            crew_kwargs["manager_llm"] = self.manager_llm or self._build_llm_from_class(LLM)

        crew = Crew(**crew_kwargs)

        try:
            crew_output = crew.kickoff(inputs={"task": task_description})
        except Exception:
            return super().run_task(task_description, max_steps=max_steps)
        final_output, message_lines = self._extract_output_text(crew_output)
        latency_seconds = time.perf_counter() - start_time
        prompt_tokens = max(1, len(system_prompt.split()) // 2)
        completion_tokens = max(1, len(final_output.split()) // 2)
        metrics = RunMetrics(
            latency_seconds=latency_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tool_calls=0,
            steps_executed=max(2, len(message_lines) + 1),
            cost_usd=0.0,
        )

        for step_index, line in enumerate(message_lines, start=1):
            self.trace_logger.log_event(
                run_id=run_id,
                step_index=step_index,
                kind=StepKind.RESPONSE,
                role=line["role"],
                content=line["content"],
                state_snapshot={"mode": "native", "message_index": step_index, "source": line.get("source")},
            )

        self.trace_logger.log_event(
            run_id=run_id,
            step_index=metrics.steps_executed,
            kind=StepKind.FINAL,
            role="system",
            content=final_output,
            latency_ms=latency_seconds * 1000.0,
            token_count=metrics.total_tokens,
            state_snapshot={"success": True, "mode": "native", "latency_seconds": latency_seconds},
        )

        full_trace = self.trace_logger.read_trace(run_id)
        self.trace_logger.finish_run(
            run_id=run_id,
            success=True,
            final_output=final_output,
            metrics=metrics.model_dump(),
        )
        return TraceResult(
            success=True,
            final_output=final_output,
            full_trace=full_trace,
            metrics=metrics,
            raw_log_path=str(context.log_path),
            run_id=run_id,
        )

    def _load_crewai_components(self) -> tuple[Any, Any, Any, Any, Any] | None:
        """Load CrewAI components dynamically so optional dependency checks stay local."""

        original_warn = warnings.warn
        try:
            crewai_module = import_module("crewai")
        except ModuleNotFoundError:
            return None
        finally:
            warnings.warn = original_warn

        Agent = getattr(crewai_module, "Agent", None)
        Crew = getattr(crewai_module, "Crew", None)
        Process = getattr(crewai_module, "Process", None)
        Task = getattr(crewai_module, "Task", None)
        LLM = getattr(crewai_module, "LLM", None)
        if any(component is None for component in (Agent, Crew, Process, Task, LLM)):
            return None
        return Agent, Crew, Process, Task, LLM

    def _build_llm(self) -> Any | None:
        """Build a CrewAI LLM instance if the optional class is available."""

        apply_llm_runtime_environment(api_key=self.llm_api_key, base_url=self.llm_base_url)

        try:
            crewai_module = import_module("crewai")
        except ModuleNotFoundError:
            return None

        llm_cls = getattr(crewai_module, "LLM", None)
        if llm_cls is None:
            return None
        return llm_cls(model=self.model_name, temperature=self.temperature)

    def _build_llm_from_class(self, llm_cls: Any) -> Any | None:
        """Build a manager LLM using the class returned from the CrewAI module."""

        try:
            return llm_cls(model=self.model_name, temperature=self.temperature)
        except Exception:
            return None

    def _build_system_prompt(self, task_description: str, max_steps: int) -> str:
        mitigation_block = apply_mitigation_strategies("", task_description, self.mitigation_strategies)
        return (
            f"CrewAI coordinator for reproducible experiments.\n"
            f"Task: {task_description}\n"
            f"Max steps: {max_steps}\n"
            f"{mitigation_block}\n"
            "The agents must remain role-consistent, keep coordination explicit, and verify the result before termination."
        )

    def _extract_output_text(self, crew_output: Any) -> tuple[str, list[dict[str, str]]]:
        message_lines: list[dict[str, str]] = []

        messages = getattr(crew_output, "messages", None)
        if messages:
            for message in messages:
                role = getattr(message, "role", getattr(message, "agent", "crew"))
                content = getattr(message, "content", None)
                if content is None and hasattr(message, "to_dict"):
                    content = message.to_dict().get("content", "")
                message_lines.append({"role": str(role), "content": str(content or ""), "source": str(role)})

        if isinstance(crew_output, str):
            return crew_output, message_lines
        for attribute_name in ("raw", "output", "summary", "final_output"):
            value = getattr(crew_output, attribute_name, None)
            if value:
                return str(value), message_lines
        fallback = str(crew_output)
        if not message_lines:
            message_lines.append({"role": "crew", "content": fallback, "source": "crew"})
        return fallback, message_lines
