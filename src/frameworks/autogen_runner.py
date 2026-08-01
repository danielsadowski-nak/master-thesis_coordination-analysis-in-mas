"""AutoGen (AG2) runner with an optional native integration.

Generated with GitHub Copilot assistance - reviewed and adapted by author.

Optional dependency installation:
    pip install "autogen-agentchat" "autogen-ext[openai]"
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any

from evaluation.trace_logger import TraceLogger
from frameworks.adapter_runner import FrameworkAdapterRunner
from frameworks.base_runner import RunMetrics, StepKind, TraceResult
from utils.config import apply_llm_runtime_environment
from utils.langsmith import LangSmithRuntime
from utils.mitigations import MitigationStrategy, apply_mitigation_strategies


class AutoGenRunner(FrameworkAdapterRunner):
    """AutoGen runner with a native AG2 execution path."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        trace_dir: Path | str = Path("results/traces"),
        trace_logger: TraceLogger | None = None,
        mitigation_strategies: tuple[MitigationStrategy, ...] = (),
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        model_client: Any | None = None,
        langsmith_project: str = "mas-coordination-analysis",
        langsmith_enabled: bool = True,
        langsmith_api_key: str | None = None,
        langsmith_endpoint: str | None = None,
        langsmith_tags: tuple[str, ...] = (),
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
    ) -> None:
        super().__init__(
            name="autogen",
            seed=seed,
            trace_dir=trace_dir,
            trace_logger=trace_logger,
            mitigation_strategies=mitigation_strategies,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.model_client = model_client
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
        """Run a task via native AutoGen and fall back to the scaffold when unavailable."""

        autogen_components = self._load_autogen_components()
        if autogen_components is None:
            return super().run_task(task_description, max_steps=max_steps)

        if self.model_client is None:
            self.model_client = self._build_model_client()
        if self.model_client is None:
            return super().run_task(task_description, max_steps=max_steps)

        AssistantAgent, UserProxyAgent, RoundRobinGroupChat, MaxMessageTermination, TextMentionTermination = autogen_components

        try:
            native_result = asyncio.run(
                self._run_native_autogen(
                    task_description=task_description,
                    max_steps=max_steps,
                    AssistantAgent=AssistantAgent,
                    UserProxyAgent=UserProxyAgent,
                    RoundRobinGroupChat=RoundRobinGroupChat,
                    MaxMessageTermination=MaxMessageTermination,
                    TextMentionTermination=TextMentionTermination,
                )
            )
            return native_result
        except Exception:
            return super().run_task(task_description, max_steps=max_steps)

    async def _run_native_autogen(
        self,
        *,
        task_description: str,
        max_steps: int,
        AssistantAgent: Any,
        UserProxyAgent: Any,
        RoundRobinGroupChat: Any,
        MaxMessageTermination: Any,
        TextMentionTermination: Any,
    ) -> TraceResult:
        """Run the AG2-native chat with explicit termination conditions and message history."""

        random.seed(self.seed)
        run_id = str(uuid.uuid4())
        context = self.trace_logger.start_run(
            framework_name=self.name,
            task_description=task_description,
            metadata={"max_steps": max_steps, "seed": self.seed, "mode": "native", "framework": "AG2"},
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

        assistant = AssistantAgent(name="assistant", model_client=self.model_client, system_message=system_prompt)
        user_proxy = UserProxyAgent(name="user_proxy")
        termination_condition = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=max_steps)
        team = RoundRobinGroupChat(
            [user_proxy, assistant],
            termination_condition=termination_condition,
        )

        chat_result = await team.run(task=task_description)
        final_output, message_lines = self._extract_output_text(chat_result)
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

    def _load_autogen_components(self) -> tuple[Any, Any, Any, Any, Any] | None:
        """Load AG2/AutoGen components dynamically without hard dependencies."""

        try:
            autogen_agents = import_module("autogen_agentchat.agents")
            autogen_teams = import_module("autogen_agentchat.teams")
            autogen_conditions = import_module("autogen_agentchat.conditions")
        except ModuleNotFoundError:
            return None

        AssistantAgent = getattr(autogen_agents, "AssistantAgent", None)
        UserProxyAgent = getattr(autogen_agents, "UserProxyAgent", None)
        RoundRobinGroupChat = getattr(autogen_teams, "RoundRobinGroupChat", None)
        MaxMessageTermination = getattr(autogen_conditions, "MaxMessageTermination", None)
        TextMentionTermination = getattr(autogen_conditions, "TextMentionTermination", None)
        if any(
            component is None for component in (AssistantAgent, UserProxyAgent, RoundRobinGroupChat, MaxMessageTermination, TextMentionTermination)
        ):
            return None
        return AssistantAgent, UserProxyAgent, RoundRobinGroupChat, MaxMessageTermination, TextMentionTermination

    def _build_model_client(self) -> Any | None:
        """Build the current AG2 OpenAI model client when the optional dependency is installed."""

        apply_llm_runtime_environment(api_key=self.llm_api_key, base_url=self.llm_base_url)

        try:
            openai_module = import_module("autogen_ext.models.openai")
        except ModuleNotFoundError:
            return None

        openai_client_cls = getattr(openai_module, "OpenAIChatCompletionClient", None)
        if openai_client_cls is None:
            return None
        # Optional dependency note: install `autogen-ext[openai]` to use this path.
        try:
            return openai_client_cls(model=self.model_name, temperature=self.temperature)
        except Exception:
            return None

    def _build_system_prompt(self, task_description: str, max_steps: int) -> str:
        mitigation_block = apply_mitigation_strategies("", task_description, self.mitigation_strategies)
        return (
            f"AutoGen coordinator for reproducible experiments.\n"
            f"Task: {task_description}\n"
            f"Max steps: {max_steps}\n"
            f"{mitigation_block}\n"
            "The agents must remain role-consistent, keep coordination explicit, and verify the result before termination."
        )

    def _extract_output_text(self, chat_result: Any) -> tuple[str, list[dict[str, str]]]:
        """Extract a textual final answer plus a compact message history for logging."""

        message_lines: list[dict[str, str]] = []

        messages = getattr(chat_result, "messages", None)
        if messages:
            for message in messages:
                role = getattr(message, "source", getattr(message, "role", "agent"))
                content = getattr(message, "content", None)
                if content is None and hasattr(message, "to_text"):
                    content = message.to_text()
                message_lines.append({"role": str(role), "content": str(content or ""), "source": str(role)})

        if message_lines:
            final_output = message_lines[-1]["content"]
        elif isinstance(chat_result, str):
            final_output = chat_result
        else:
            for attribute_name in ("summary", "content", "raw", "output", "final_output"):
                value = getattr(chat_result, attribute_name, None)
                if value:
                    final_output = str(value)
                    break
            else:
                final_output = str(chat_result)
                message_lines.append({"role": "assistant", "content": final_output, "source": "assistant"})

        return final_output, message_lines
