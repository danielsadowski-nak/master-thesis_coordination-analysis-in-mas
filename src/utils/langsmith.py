"""LangSmith tracing helpers.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LangSmithRuntime:
    """Runtime tracing settings used by LangGraph and LangChain components."""

    enabled: bool = True
    project: str = "mas-coordination-analysis"
    api_key: str | None = None
    endpoint: str | None = None
    tags: tuple[str, ...] = ()

    def apply(self) -> dict[str, Any]:
        """Apply the runtime configuration to process environment variables."""

        if self.enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.project
            os.environ["LANGSMITH_PROJECT"] = self.project
            if self.api_key:
                os.environ["LANGSMITH_API_KEY"] = self.api_key
            if self.endpoint:
                os.environ["LANGSMITH_ENDPOINT"] = self.endpoint
        else:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
            os.environ.setdefault("LANGSMITH_TRACING", "false")

        return {
            "enabled": self.enabled,
            "project": self.project,
            "endpoint": self.endpoint,
            "tags": list(self.tags),
        }


def build_trace_config(
    *,
    project: str,
    run_id: str,
    framework: str,
    benchmark: str,
    task_description: str,
    tags: tuple[str, ...] = (),
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a LangChain/LangGraph config payload with tracing metadata."""

    metadata = {
        "project": project,
        "run_id": run_id,
        "framework": framework,
        "benchmark": benchmark,
        "task_description": task_description,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "configurable": {"thread_id": run_id},
        "metadata": metadata,
        "tags": list(tags) + [framework, benchmark],
        "run_name": f"{framework}:{benchmark}",
    }
