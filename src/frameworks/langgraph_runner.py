"""LangGraph-based runner with checkpointing and structured tracing.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel

from evaluation.trace_logger import TraceLogger
from frameworks.base_runner import BaseMASRunner, RunMetrics, StepKind, TraceResult
from utils.config import apply_llm_runtime_environment
from utils.langsmith import LangSmithRuntime, build_trace_config
from utils.mitigations import MitigationStrategy, apply_mitigation_strategies

try:  # pragma: no cover - optional dependency guard for local scaffolding.
    checkpoint_module = import_module("langgraph.checkpoint.memory")
    graph_module = import_module("langgraph.graph")
    MemorySaver = getattr(checkpoint_module, "MemorySaver", None)
    StateGraph = getattr(graph_module, "StateGraph", None)
    END = getattr(graph_module, "END", "__end__")
    START = getattr(graph_module, "START", "__start__")
except ImportError:  # pragma: no cover - fallback when langgraph is unavailable locally.
    MemorySaver = None
    StateGraph = None
    START = "__start__"
    END = "__end__"


class LangGraphStateRequired(TypedDict):
    """Required state keys used in all execution paths."""

    run_id: str
    task_description: str
    max_steps: int


class LangGraphState(LangGraphStateRequired, total=False):
    """State carried through the LangGraph workflow."""

    step_index: int
    messages: list[dict[str, Any]]
    final_output: str
    success: bool
    done: bool
    error_message: str
    token_usage: dict[str, int | float]
    latency_seconds: float


class LLMMessage(BaseModel):
    """Lightweight message envelope for model interchange."""

    role: str
    content: str


class LangGraphRunner(BaseMASRunner):
    """Stateful LangGraph orchestration with trace logging and checkpointing."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        trace_logger: TraceLogger | None = None,
        trace_dir: Path | str = Path("results/traces"),
        checkpoint_dir: Path | str = Path("results/checkpoints"),
        system_prompt: str | None = None,
        langsmith_project: str = "mas-coordination-analysis",
        langsmith_enabled: bool = False,
        langsmith_api_key: str | None = None,
        langsmith_endpoint: str | None = None,
        langsmith_tags: tuple[str, ...] = (),
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        mitigation_strategies: tuple[MitigationStrategy, ...] = (),
        seed: int | None = 42,
    ) -> None:
        super().__init__(name="langgraph", seed=seed)
        self.model = model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.trace_logger = trace_logger or TraceLogger(base_dir=trace_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = system_prompt or (
            "You are a coordination-aware multi-agent system that must solve the task, "
            "record every important action, and terminate only when the task is complete."
        )
        self.langsmith_runtime = LangSmithRuntime(
            enabled=langsmith_enabled,
            project=langsmith_project,
            api_key=langsmith_api_key,
            endpoint=langsmith_endpoint,
            tags=langsmith_tags,
        )
        self.mitigation_strategies = mitigation_strategies
        apply_llm_runtime_environment(api_key=llm_api_key, base_url=llm_base_url)
        self.langsmith_runtime.apply()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Compile the LangGraph workflow when the dependency is available."""

        if StateGraph is None:
            return None

        graph = StateGraph(LangGraphState)
        graph.add_node("prepare", self._prepare_state)
        graph.add_node("act", self._agent_step)
        graph.add_node("finalize", self._finalize_state)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "act")
        graph.add_conditional_edges("act", self._route_after_act, {"continue": "act", "finalize": "finalize"})
        graph.add_edge("finalize", END)
        checkpointer = MemorySaver() if MemorySaver is not None else None
        return graph.compile(checkpointer=checkpointer)

    def run_task(self, task_description: str, max_steps: int = 25) -> TraceResult:
        """Run one benchmark task and return a structured trace result."""

        random.seed(self.seed)
        run_id = str(uuid.uuid4())
        context = self.trace_logger.start_run(
            framework_name=self.name,
            task_description=task_description,
            metadata={"max_steps": max_steps, "seed": self.seed},
            run_id=run_id,
        )
        start_time = time.perf_counter()

        initial_state: LangGraphState = {
            "run_id": run_id,
            "task_description": task_description,
            "max_steps": max_steps,
            "step_index": 0,
            "messages": [],
            "success": False,
            "done": False,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_seconds": 0.0,
        }

        try:
            if self._graph is not None:
                final_state = self._graph.invoke(
                    initial_state,
                    config=build_trace_config(
                        project=self.langsmith_runtime.project,
                        run_id=run_id,
                        framework=self.name,
                        benchmark="unknown",
                        task_description=task_description,
                        tags=self.langsmith_runtime.tags,
                        extra_metadata={"max_steps": max_steps, "seed": self.seed},
                    ),
                )
            else:
                final_state = self._run_without_langgraph(initial_state)
        except Exception as exc:  # pragma: no cover - explicit failure recording path.
            error_text = f"LangGraph runner failed: {exc!s}"
            self.trace_logger.log_event(
                run_id=run_id,
                step_index=0,
                kind=StepKind.ERROR,
                role="system",
                content=error_text,
                state_snapshot={"task_description": task_description},
                metadata={"exception_type": type(exc).__name__},
            )
            return self._build_empty_result(task_description, context.log_path, error_text)

        latency_seconds = time.perf_counter() - start_time
        full_trace = self.trace_logger.read_trace(run_id)
        token_usage = final_state.get("token_usage", {})
        metrics = RunMetrics(
            latency_seconds=latency_seconds,
            prompt_tokens=int(token_usage.get("prompt_tokens", 0)),
            completion_tokens=int(token_usage.get("completion_tokens", 0)),
            total_tokens=int(token_usage.get("total_tokens", 0)),
            tool_calls=int(token_usage.get("tool_calls", 0)),
            steps_executed=int(final_state.get("step_index", len(full_trace))),
            cost_usd=float(token_usage.get("cost_usd", 0.0)),
        )
        final_output = str(final_state.get("final_output", ""))
        success = bool(final_state.get("success", False))

        self.trace_logger.finish_run(
            run_id=run_id,
            success=success,
            final_output=final_output,
            metrics=metrics.model_dump(),
        )

        return TraceResult(
            success=success,
            final_output=final_output,
            full_trace=full_trace,
            metrics=metrics,
            raw_log_path=str(context.log_path),
            run_id=run_id,
        )

    def run_task_with_context(
        self,
        *,
        task_description: str,
        benchmark_name: str,
        max_steps: int = 25,
    ) -> TraceResult:
        """Run a task with an explicit benchmark context for LangSmith metadata."""

        random.seed(self.seed)
        run_id = str(uuid.uuid4())
        context = self.trace_logger.start_run(
            framework_name=self.name,
            task_description=task_description,
            metadata={"max_steps": max_steps, "seed": self.seed, "benchmark": benchmark_name},
            run_id=run_id,
        )
        start_time = time.perf_counter()

        initial_state: LangGraphState = {
            "run_id": run_id,
            "task_description": task_description,
            "max_steps": max_steps,
            "step_index": 0,
            "messages": [],
            "success": False,
            "done": False,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_seconds": 0.0,
        }

        try:
            if self._graph is not None:
                final_state = self._graph.invoke(
                    initial_state,
                    config=build_trace_config(
                        project=self.langsmith_runtime.project,
                        run_id=run_id,
                        framework=self.name,
                        benchmark=benchmark_name,
                        task_description=task_description,
                        tags=self.langsmith_runtime.tags,
                        extra_metadata={"max_steps": max_steps, "seed": self.seed},
                    ),
                )
            else:
                final_state = self._run_without_langgraph(initial_state)
        except Exception as exc:  # pragma: no cover - explicit failure recording path.
            error_text = f"LangGraph runner failed: {exc!s}"
            self.trace_logger.log_event(
                run_id=run_id,
                step_index=0,
                kind=StepKind.ERROR,
                role="system",
                content=error_text,
                state_snapshot={"task_description": task_description},
                metadata={"exception_type": type(exc).__name__},
            )
            return self._build_empty_result(task_description, context.log_path, error_text)

        latency_seconds = time.perf_counter() - start_time
        full_trace = self.trace_logger.read_trace(run_id)
        token_usage = final_state.get("token_usage", {})
        metrics = RunMetrics(
            latency_seconds=latency_seconds,
            prompt_tokens=int(token_usage.get("prompt_tokens", 0)),
            completion_tokens=int(token_usage.get("completion_tokens", 0)),
            total_tokens=int(token_usage.get("total_tokens", 0)),
            tool_calls=int(token_usage.get("tool_calls", 0)),
            steps_executed=int(final_state.get("step_index", len(full_trace))),
            cost_usd=float(token_usage.get("cost_usd", 0.0)),
        )
        final_output = str(final_state.get("final_output", ""))
        success = bool(final_state.get("success", False))

        self.trace_logger.finish_run(
            run_id=run_id,
            success=success,
            final_output=final_output,
            metrics=metrics.model_dump(),
        )

        return TraceResult(
            success=success,
            final_output=final_output,
            full_trace=full_trace,
            metrics=metrics,
            raw_log_path=str(context.log_path),
            run_id=run_id,
        )

    def _run_without_langgraph(self, state: LangGraphState) -> LangGraphState:
        """Fallback path that still produces a reproducible trace without LangGraph installed."""

        state = self._prepare_state(state)
        state = self._agent_step(state)
        state = self._finalize_state(state)
        return state

    def _prepare_state(self, state: LangGraphState) -> LangGraphState:
        run_id = state["run_id"]
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=0,
            kind=StepKind.PROMPT,
            role="system",
            content=self.system_prompt,
            state_snapshot={"task_description": state["task_description"], "step_index": 0},
        )
        return state

    def _agent_step(self, state: LangGraphState) -> LangGraphState:
        run_id = state["run_id"]
        step_index = int(state.get("step_index", 0)) + 1
        task_description = state["task_description"]
        messages = list(state.get("messages", []))
        prompt = self._build_prompt(task_description, messages)
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=step_index,
            kind=StepKind.PROMPT,
            role="assistant",
            content=prompt,
            state_snapshot={"step_index": step_index, "messages": messages},
        )

        start = time.perf_counter()
        response_text, token_usage = self._invoke_model(prompt=prompt, messages=messages)
        latency_ms = (time.perf_counter() - start) * 1000.0

        self.trace_logger.log_event(
            run_id=run_id,
            step_index=step_index,
            kind=StepKind.RESPONSE,
            role="assistant",
            content=response_text,
            latency_ms=latency_ms,
            token_count=int(token_usage.get("total_tokens", 0)) if token_usage.get("total_tokens") is not None else None,
            state_snapshot={"step_index": step_index, "prompt_length": len(prompt)},
            metadata={"prompt_tokens": token_usage.get("prompt_tokens", 0), "completion_tokens": token_usage.get("completion_tokens", 0)},
        )

        messages.append({"role": "assistant", "content": response_text})
        state.update(
            {
                "step_index": step_index,
                "messages": messages,
                "final_output": response_text,
                "success": self._is_success_response(response_text),
                "done": True,
                "token_usage": self._merge_token_usage(state.get("token_usage", {}), token_usage),
                "latency_seconds": state.get("latency_seconds", 0.0) + latency_ms / 1000.0,
            }
        )
        return state

    def _finalize_state(self, state: LangGraphState) -> LangGraphState:
        run_id = state["run_id"]
        self.trace_logger.log_event(
            run_id=run_id,
            step_index=int(state.get("step_index", 0)) + 1,
            kind=StepKind.FINAL,
            role="system",
            content=state.get("final_output", ""),
            state_snapshot={"success": state.get("success", False), "done": state.get("done", False)},
        )
        return state

    def _route_after_act(self, state: LangGraphState) -> str:
        if state.get("done", False) or int(state.get("step_index", 0)) >= int(state.get("max_steps", 1)):
            return "finalize"
        return "continue"

    def _build_prompt(self, task_description: str, messages: list[dict[str, Any]]) -> str:
        history = "\n".join(f"{message.get('role', 'unknown')}: {message.get('content', '')}" for message in messages)
        if history:
            history = f"\nPrevious turns:\n{history}\n"
        mitigation_block = apply_mitigation_strategies("", task_description, self.mitigation_strategies)
        if mitigation_block:
            mitigation_block = f"\n{mitigation_block}\n"
        return (
            f"{self.system_prompt}\n"
            f"Task:\n{task_description}\n"
            f"{mitigation_block}"
            f"{history}"
            "Return a concise answer and indicate when the task is complete."
        )

    def _invoke_model(self, prompt: str, messages: list[dict[str, Any]]) -> tuple[str, dict[str, int | float]]:
        if self.model is None:
            return self._fallback_response(prompt, messages), {
                "prompt_tokens": max(1, len(prompt.split()) // 3),
                "completion_tokens": max(1, len(prompt.split()) // 5),
                "total_tokens": max(2, len(prompt.split()) // 2),
                "tool_calls": 0,
                "cost_usd": 0.0,
            }

        payload = [
            LLMMessage(role="system", content=self.system_prompt).model_dump(),
            *messages,
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.model, "invoke"):
            response = self.model.invoke(payload)
        else:
            response = self.model(payload)

        response_text = self._extract_text(response)
        token_usage = self._extract_token_usage(response)
        if not token_usage:
            token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "tool_calls": 0,
                "cost_usd": 0.0,
            }
        return response_text, token_usage

    def _fallback_response(self, prompt: str, messages: list[dict[str, Any]]) -> str:
        preview = prompt.splitlines()[-1] if prompt.splitlines() else prompt
        return (
            "No external model configured. This LangGraph runner is operating in fallback mode. "
            f"Task preview: {preview[:180]}"
        )

    def _extract_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            content = getattr(response, "content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
        if isinstance(response, dict):
            for key in ("content", "text", "output"):
                if key in response:
                    return str(response[key])
        return json.dumps(response, default=str)

    def _extract_token_usage(self, response: Any) -> dict[str, int | float]:
        usage = getattr(response, "usage", None) or getattr(response, "response_metadata", None)
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tool_calls": int(usage.get("tool_calls", 0) or 0),
                "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
            }
        return {}

    def _merge_token_usage(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, int | float]:
        merged = {
            "prompt_tokens": int(left.get("prompt_tokens", 0)) + int(right.get("prompt_tokens", 0)),
            "completion_tokens": int(left.get("completion_tokens", 0)) + int(right.get("completion_tokens", 0)),
            "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
            "tool_calls": int(left.get("tool_calls", 0)) + int(right.get("tool_calls", 0)),
            "cost_usd": float(left.get("cost_usd", 0.0)) + float(right.get("cost_usd", 0.0)),
        }
        return merged

    def _is_success_response(self, response_text: str) -> bool:
        response_lower = response_text.lower()
        return any(token in response_lower for token in ("success", "completed", "done", "resolved"))
