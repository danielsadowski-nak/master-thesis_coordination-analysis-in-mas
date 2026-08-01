"""Tests for mitigation plugins."""

from __future__ import annotations

from frameworks.adapter_runner import FrameworkAdapterRunner
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
