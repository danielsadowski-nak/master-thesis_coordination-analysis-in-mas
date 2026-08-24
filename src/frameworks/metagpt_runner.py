"""MetaGPT runner with native runtime detection.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import time
from typing import Any
import uuid

from evaluation.trace_logger import TraceLogger
from frameworks.adapter_runner import FrameworkAdapterRunner
from frameworks.base_runner import RunMetrics, StepKind, TraceResult
from utils.config import apply_llm_runtime_environment
from utils.langsmith import LangSmithRuntime
from utils.mitigations import MitigationStrategy


class MetaGptRunner(FrameworkAdapterRunner):
    """MetaGPT runner with an explicit native runtime path."""

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
        role_config: dict[str, Any] | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
    ) -> None:
        super().__init__(
            name="metagpt",
            seed=seed,
            trace_dir=trace_dir,
            trace_logger=trace_logger,
            mitigation_strategies=mitigation_strategies,
        )
        self.role_config = role_config or {}
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
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
        """Run MetaGPT natively.

        This adapter must never silently degrade to scaffold mode because such
        runs are analytically invalid for primary framework comparisons.
        """

        apply_llm_runtime_environment(api_key=self.llm_api_key, base_url=self.llm_base_url)

        try:
            import_module("metagpt")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MetaGPT native runtime is unavailable (package 'metagpt' is missing). "
                "Run via docker-compose.metagpt.yml and enable --require-native-frameworks "
                "to enforce native-only baseline execution."
            ) from exc

        try:
            metagpt_actions = __import__("metagpt.actions", fromlist=["Action"])
            metagpt_environment = __import__("metagpt.environment", fromlist=["Environment"])
            metagpt_roles = __import__("metagpt.roles", fromlist=["Role"])
            metagpt_team = __import__("metagpt.team", fromlist=["Team"])
            metagpt_schema = __import__("metagpt.schema", fromlist=["Message"])
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MetaGPT native integration is unavailable. Install 'metagpt' to run this adapter."
            ) from exc

        Action = getattr(metagpt_actions, "Action", None)
        Environment = getattr(metagpt_environment, "Environment", None)
        Role = getattr(metagpt_roles, "Role", None)
        Team = getattr(metagpt_team, "Team", None)
        Message = getattr(metagpt_schema, "Message", None)
        if any(component is None for component in (Action, Environment, Role, Team, Message)):
            raise RuntimeError(
                "MetaGPT native integration could not load required classes. "
                "Use docker-compose.metagpt.yml and run with --require-native-frameworks."
            )

        class CoordinationAction(Action):
            name: str = "CoordinationAction"

            async def run(self, context: str, **kwargs: Any) -> str:
                prompt = (
                    f"You are a MetaGPT-native coordinator.\n"
                    f"Task: {task_description}\n"
                    f"Context: {context}\n"
                    f"Max steps: {max_steps}\n"
                    "Provide a concise final answer and mention coordination issues explicitly if any are present."
                )
                response = await self._aask(prompt)
                return str(response)

        class CoordinationRole(Role):
            name: str = "Coordinator"
            profile: str = "Coordinator"

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.set_actions([CoordinationAction])
                self._set_react_mode(react_mode="by_order")

            async def _act(self) -> Any:
                todo = self.rc.todo
                memories = self.get_memories(k=1)
                context = memories[0].content if memories else task_description
                result = await todo.run(context)
                message = Message(content=result, role=self.profile, cause_by=type(todo))
                self.rc.memory.add(message)
                return message

        context_desc = f"Task coordination run for: {task_description}"
        start_time = time.perf_counter()
        environment = Environment(desc=context_desc)
        team = Team(investment=1.0, env=environment, roles=[CoordinationRole()])

        if hasattr(team, "run_project"):
            team.run_project(task_description, send_to="Coordinator")

        result = __import__("asyncio").run(team.run(n_round=max_steps))
        final_output = self._extract_output_text(result)
        latency_seconds = time.perf_counter() - start_time
        success = self._infer_native_success(final_output)

        metrics = RunMetrics(
            latency_seconds=latency_seconds,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tool_calls=0,
            steps_executed=max_steps,
            cost_usd=0.0,
        )
        run_id = str(uuid.uuid4())
        context = self.trace_logger.start_run(
            framework_name=self.name,
            task_description=task_description,
            metadata={
                "max_steps": max_steps,
                "seed": self.seed,
                "mode": "native",
                "framework": "MetaGPT",
                "runtime_mode": "native",
                "is_valid_analytical": True,
                "tokens_estimated": False,
            },
            run_id=run_id,
        )
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=0,
            kind=StepKind.PROMPT,
            role="system",
            content=task_description,
            state_snapshot={"task_description": task_description, "max_steps": max_steps, "mode": "native"},
        )
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=1,
            kind=StepKind.RESPONSE,
            role="metagpt",
            content=final_output,
            state_snapshot={
                "mode": "native",
                "runtime_mode": "native",
                "is_valid_analytical": True,
                "latency_seconds": latency_seconds,
                "tokens_estimated": False,
            },
        )
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=2,
            kind=StepKind.FINAL,
            role="system",
            content=final_output,
            state_snapshot={
                "success": success,
                "mode": "native",
                "runtime_mode": "native",
                "is_valid_analytical": True,
            },
        )
        full_trace = self.trace_logger.read_trace(run_id)
        self.trace_logger.finish_run(run_id=run_id, success=success, final_output=final_output, metrics=metrics.model_dump())
        return TraceResult(
            success=success,
            final_output=final_output,
            full_trace=full_trace,
            metrics=metrics,
            raw_log_path=str(context.log_path),
            run_id=run_id,
            runtime_mode="native",
            is_valid_analytical=True,
        )

    @staticmethod
    def _infer_native_success(final_output: str) -> bool:
        text = (final_output or "").strip().lower()
        if not text:
            return False
        failure_markers = (
            "cannot complete",
            "unable to complete",
            "failed",
            "error",
            "runtime_mode=scaffold",
            "placeholder run",
        )
        return not any(marker in text for marker in failure_markers)
