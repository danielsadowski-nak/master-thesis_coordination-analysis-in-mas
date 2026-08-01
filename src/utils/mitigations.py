"""Configurable mitigation plugins for coordination experiments.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol


class MitigationStrategy(Protocol):
    """Protocol for prompt-level mitigation plugins."""

    name: str

    def augment_system_prompt(self, base_prompt: str, task_description: str) -> str:
        """Return an augmented system prompt for the current task."""


@dataclass(slots=True)
class StructuredOutputValidationMitigation:
    """Require structured output and explicit schema-level self-validation."""

    name: str = "structured_output_validation"

    def augment_system_prompt(self, base_prompt: str, task_description: str) -> str:
        return (
            f"{base_prompt}\n\n"
            "Mitigation plugin: structured output + pydantic validation. "
            "Return the final answer as structured JSON with keys: plan, execution_summary, verification_checks, final_answer. "
            "Before finalizing, validate that all required keys are present, types are correct, and each verification check is grounded in evidence. "
            "If schema validation fails, repair output and re-validate before termination."
        )


@dataclass(slots=True)
class SupervisorOrchestratorMitigation:
    """Use a supervisor orchestration pattern with explicit stage gates."""

    name: str = "supervisor_orchestrator"

    def augment_system_prompt(self, base_prompt: str, task_description: str) -> str:
        return (
            f"{base_prompt}\n\n"
            "Mitigation plugin: supervisor/orchestrator pattern. "
            "Follow staged execution: (1) planning, (2) delegated execution, (3) supervisor review, (4) finalization. "
            "No stage transition without an explicit supervisor gate approval. "
            "If unresolved ambiguity remains, request clarification rather than continuing with assumptions."
        )


@dataclass(slots=True)
class ReflectionIndependentVerificationMitigation:
    """Require a reflection pass and an independent judge-style verification pass."""

    name: str = "reflection_independent_verification"

    def augment_system_prompt(self, base_prompt: str, task_description: str) -> str:
        return (
            f"{base_prompt}\n\n"
            "Mitigation plugin: reflection + independent verification. "
            "Run two separate checks before final answer: "
            "(A) reflection pass that lists potential mistakes or omissions; "
            "(B) independent judge pass that verifies task compliance, coordination consistency, and termination correctness. "
            "If either check fails, revise once and re-run both checks."
        )


@dataclass(slots=True)
class StructuredProtocolMitigation(StructuredOutputValidationMitigation):
    """Backward-compatible alias for legacy configuration names."""

    name: str = "structured_protocol"


@dataclass(slots=True)
class SupervisorPatternMitigation(SupervisorOrchestratorMitigation):
    """Backward-compatible alias for legacy configuration names."""

    name: str = "supervisor_pattern"


@dataclass(slots=True)
class CrossVerificationMitigation(ReflectionIndependentVerificationMitigation):
    """Backward-compatible alias for legacy configuration names."""

    name: str = "cross_verification"


@dataclass(slots=True)
class ReflectionLoopMitigation(ReflectionIndependentVerificationMitigation):
    """Backward-compatible alias for legacy configuration names."""

    name: str = "reflection_loop"


def apply_mitigation_strategies(
    base_prompt: str,
    task_description: str,
    strategies: tuple[MitigationStrategy, ...] = (),
) -> str:
    """Apply mitigation plugins in order to a system prompt."""

    prompt = base_prompt
    for strategy in strategies:
        prompt = strategy.augment_system_prompt(prompt, task_description)
    return prompt


MITIGATION_REGISTRY: dict[str, type[MitigationStrategy]] = {
    "structured_output_validation": StructuredOutputValidationMitigation,
    "supervisor_orchestrator": SupervisorOrchestratorMitigation,
    "reflection_independent_verification": ReflectionIndependentVerificationMitigation,
    # Backward-compatible names.
    "structured_protocol": StructuredProtocolMitigation,
    "supervisor_pattern": SupervisorPatternMitigation,
    "cross_verification": CrossVerificationMitigation,
    "reflection_loop": ReflectionLoopMitigation,
}


def build_mitigation_strategies(names: Sequence[str]) -> tuple[MitigationStrategy, ...]:
    """Resolve mitigation plugin names into instantiated strategies."""

    strategies: list[MitigationStrategy] = []
    for raw_name in names:
        normalized_name = raw_name.strip().lower()
        strategy_cls = MITIGATION_REGISTRY.get(normalized_name)
        if strategy_cls is None:
            supported = ", ".join(sorted(MITIGATION_REGISTRY))
            raise ValueError(f"Unknown mitigation strategy '{raw_name}'. Supported strategies: {supported}")
        strategies.append(strategy_cls())
    return tuple(strategies)
