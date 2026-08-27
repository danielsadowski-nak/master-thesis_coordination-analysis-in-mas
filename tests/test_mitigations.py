"""Tests for mitigation plugins."""

from __future__ import annotations

from types import SimpleNamespace

from frameworks.adapter_runner import FrameworkAdapterRunner
from frameworks.autogen_runner import AutoGenRunner
from frameworks.crewai_runner import CrewAIRunner
from frameworks.langgraph_runner import LangGraphRunner
from utils.mitigations import (
    StructuredOutputValidationMitigation,
    SupervisorOrchestratorMitigation,
    ReflectionIndependentVerificationMitigation,
    CrossVerificationMitigation,
    ReflectionLoopMitigation,
    StructuredProtocolMitigation,
    SupervisorPatternMitigation,
    apply_mitigation_strategies,
    build_mitigation_strategies,
)


def test_apply_mitigation_strategies_composes_in_order() -> None:
    prompt = apply_mitigation_strategies(
        "Base prompt",
        "Solve the task.",
        (StructuredProtocolMitigation(), SupervisorPatternMitigation()),
    )

    assert "structured output" in prompt.lower()
    assert "supervisor/orchestrator pattern" in prompt.lower()


def test_adapter_runner_prompt_includes_mitigation() -> None:
    runner = FrameworkAdapterRunner(
        name="test-adapter",
        mitigation_strategies=(CrossVerificationMitigation(), ReflectionLoopMitigation()),
    )

    prompt = runner._build_prompt(task_description="Solve the task.", max_steps=3)

    assert "independent verification" in prompt.lower()
    assert "reflection" in prompt.lower()


def test_langgraph_prompt_includes_mitigation() -> None:
    runner = LangGraphRunner(
        mitigation_strategies=(StructuredProtocolMitigation(),),
    )

    prompt = runner._build_prompt("Solve the task.", [])

    assert "structured output" in prompt.lower()


def test_langgraph_extracts_usage_metadata_from_langchain_response() -> None:
    runner = LangGraphRunner()

    usage = runner._extract_token_usage(
        SimpleNamespace(
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            response_metadata={},
        )
    )

    assert usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "tool_calls": 0,
        "cost_usd": 0.0,
    }


def test_langgraph_extracts_nested_response_metadata_token_usage() -> None:
    runner = LangGraphRunner()

    usage = runner._extract_token_usage(
        SimpleNamespace(
            usage_metadata=None,
            response_metadata={"token_usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
        )
    )

    assert usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
        "tool_calls": 0,
        "cost_usd": 0.0,
    }


def test_new_mitigation_strategies_have_expected_prompt_behavior() -> None:
    prompt = apply_mitigation_strategies(
        "Base prompt",
        "Solve the task.",
        (
            StructuredOutputValidationMitigation(),
            SupervisorOrchestratorMitigation(),
            ReflectionIndependentVerificationMitigation(),
        ),
    )

    assert "structured output" in prompt.lower()
    assert "supervisor/orchestrator pattern" in prompt.lower()
    assert "independent verification" in prompt.lower()


def test_build_mitigation_strategies_resolves_and_validates_names() -> None:
    strategies = build_mitigation_strategies(
        [
            "structured_output_validation",
            "supervisor_orchestrator",
            "reflection_independent_verification",
        ]
    )

    assert [strategy.name for strategy in strategies] == [
        "structured_output_validation",
        "supervisor_orchestrator",
        "reflection_independent_verification",
    ]


def test_none_vs_plugin_changes_prompt_for_primary_runners() -> None:
    task = "Solve the task."

    autogen_none = AutoGenRunner(mitigation_strategies=())._build_system_prompt(task, 3)
    autogen_plugin = AutoGenRunner(mitigation_strategies=(StructuredOutputValidationMitigation(),))._build_system_prompt(task, 3)
    assert "Mitigation plugin:" not in autogen_none
    assert "Mitigation plugin:" in autogen_plugin
    assert autogen_none != autogen_plugin

    crewai_none = CrewAIRunner(mitigation_strategies=())._build_system_prompt(task, 3)
    crewai_plugin = CrewAIRunner(mitigation_strategies=(StructuredOutputValidationMitigation(),))._build_system_prompt(task, 3)
    assert "Mitigation plugin:" not in crewai_none
    assert "Mitigation plugin:" in crewai_plugin
    assert crewai_none != crewai_plugin

    langgraph_none = LangGraphRunner(mitigation_strategies=())._build_prompt(task, [])
    langgraph_plugin = LangGraphRunner(mitigation_strategies=(StructuredOutputValidationMitigation(),))._build_prompt(task, [])
    assert "Mitigation plugin:" not in langgraph_none
    assert "Mitigation plugin:" in langgraph_plugin
    assert langgraph_none != langgraph_plugin
