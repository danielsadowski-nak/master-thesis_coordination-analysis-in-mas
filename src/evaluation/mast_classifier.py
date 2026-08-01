"""LLM-as-Judge implementation for the MAST taxonomy.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MASTCategory(str, Enum):
    """Top-level MAST categories."""

    TASK_COMPLIANCE = "1. Task Compliance"
    COORDINATION = "2. Coordination"
    TERMINATION_AND_VERIFICATION = "3. Termination and Verification"


class MASTFailureMode(str, Enum):
    """The 14 MAST failure modes grouped into three categories."""

    DISOBEY_TASK_SPECIFICATION = "1.1 Disobey Task Specification"
    DISOBEY_ROLE_SPECIFICATION = "1.2 Disobey Role Specification"
    STEP_REPETITION = "1.3 Step Repetition"
    LOSS_OF_CONVERSATION_HISTORY = "1.4 Loss of Conversation History"
    UNAWARE_OF_TERMINATION_CONDITIONS = "1.5 Unaware of Termination Conditions"
    CONVERSATION_RESET = "2.1 Conversation Reset"
    FAIL_TO_ASK_FOR_CLARIFICATION = "2.2 Fail to Ask for Clarification"
    TASK_DERAILMENT = "2.3 Task Derailment"
    INFORMATION_WITHHOLDING = "2.4 Information Withholding"
    IGNORED_OTHER_AGENT_INPUT = "2.5 Ignored Other Agent's Input"
    ACTION_REASONING_MISMATCH = "2.6 Action-Reasoning Mismatch"
    PREMATURE_TERMINATION = "3.1 Premature Termination"
    WEAK_VERIFICATION = "3.2 Weak Verification"
    NO_OR_INCORRECT_VERIFICATION = "3.3 No or Incorrect Verification"


MAST_FAILURE_MODE_DEFINITIONS: dict[MASTFailureMode, str] = {
    MASTFailureMode.DISOBEY_TASK_SPECIFICATION: (
        "The agent or system ignores explicit task constraints, requirements, or instructions."
    ),
    MASTFailureMode.DISOBEY_ROLE_SPECIFICATION: (
        "The agent violates the responsibilities, tone, or boundaries of its assigned role."
    ),
    MASTFailureMode.STEP_REPETITION: (
        "The workflow repeats a phase, step, or action that has already been completed."
    ),
    MASTFailureMode.LOSS_OF_CONVERSATION_HISTORY: (
        "The system loses or disregards relevant prior context and reverts to an earlier state."
    ),
    MASTFailureMode.UNAWARE_OF_TERMINATION_CONDITIONS: (
        "The system fails to respect stopping criteria and continues after termination should occur."
    ),
    MASTFailureMode.CONVERSATION_RESET: (
        "The dialogue restarts unexpectedly and loses the accumulated progress of the interaction."
    ),
    MASTFailureMode.FAIL_TO_ASK_FOR_CLARIFICATION: (
        "The system does not request missing information when the task is underspecified or ambiguous."
    ),
    MASTFailureMode.TASK_DERAILMENT: (
        "The system drifts away from the original objective and starts pursuing irrelevant work."
    ),
    MASTFailureMode.INFORMATION_WITHHOLDING: (
        "A relevant agent or component has important information but does not share it when needed."
    ),
    MASTFailureMode.IGNORED_OTHER_AGENT_INPUT: (
        "An agent receives useful input or suggestions from another agent but fails to consider them."
    ),
    MASTFailureMode.ACTION_REASONING_MISMATCH: (
        "The actual action or output does not match the reasoning or plan that preceded it."
    ),
    MASTFailureMode.PREMATURE_TERMINATION: (
        "The system stops the interaction before the task, verification, or communication requirements are met."
    ),
    MASTFailureMode.WEAK_VERIFICATION: (
        "Verification exists, but it is superficial or incomplete and misses important checks."
    ),
    MASTFailureMode.NO_OR_INCORRECT_VERIFICATION: (
        "The system does not verify the result properly, or verification is performed incorrectly."
    ),
}


class MASTFewShotExample(BaseModel):
    """Single positive example used to calibrate the judge prompt."""

    mode: MASTFailureMode
    category: MASTCategory
    example: str
    evidence_hint: str


MAST_FEW_SHOT_EXAMPLES: list[MASTFewShotExample] = [
    MASTFewShotExample(
        mode=MASTFailureMode.DISOBEY_TASK_SPECIFICATION,
        category=MASTCategory.TASK_COMPLIANCE,
        example="The agent was asked to summarize the bug report, but instead rewrote the task and ignored the requested format.",
        evidence_hint="Look for explicit constraint violations or ignored instructions.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.DISOBEY_ROLE_SPECIFICATION,
        category=MASTCategory.TASK_COMPLIANCE,
        example="The code reviewer starts generating implementation code instead of reviewing the code.",
        evidence_hint="The agent behaves like a different role than assigned.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.STEP_REPETITION,
        category=MASTCategory.TASK_COMPLIANCE,
        example="The planner repeats the same instruction multiple times after the executor already acknowledged it.",
        evidence_hint="Repeated phase/action with no new information.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.LOSS_OF_CONVERSATION_HISTORY,
        category=MASTCategory.TASK_COMPLIANCE,
        example="The agent forgets the previously agreed file path and asks for it again immediately afterward.",
        evidence_hint="Relevant context disappears despite being present earlier.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.UNAWARE_OF_TERMINATION_CONDITIONS,
        category=MASTCategory.TASK_COMPLIANCE,
        example="The agent keeps producing follow-up steps even after the task has already been marked complete.",
        evidence_hint="Continues beyond stopping criteria.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.CONVERSATION_RESET,
        category=MASTCategory.COORDINATION,
        example="The conversation restarts from scratch after partial progress, dropping the earlier plan and state.",
        evidence_hint="System abruptly restarts the interaction context.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.FAIL_TO_ASK_FOR_CLARIFICATION,
        category=MASTCategory.COORDINATION,
        example="The task is underspecified, but the agent guesses instead of asking for the missing detail.",
        evidence_hint="Missing information should trigger a clarification request.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.TASK_DERAILMENT,
        category=MASTCategory.COORDINATION,
        example="The agent starts discussing unrelated deployment issues while the benchmark asks for code correction.",
        evidence_hint="Attention shifts away from the original goal.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.INFORMATION_WITHHOLDING,
        category=MASTCategory.COORDINATION,
        example="A navigator identifies the relevant file path but does not pass it to the repair agent.",
        evidence_hint="Important information is known but not communicated.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.IGNORED_OTHER_AGENT_INPUT,
        category=MASTCategory.COORDINATION,
        example="One agent recommends a correction, but the next agent ignores the suggestion and repeats the same mistake.",
        evidence_hint="Peer input is present but not incorporated.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.ACTION_REASONING_MISMATCH,
        category=MASTCategory.COORDINATION,
        example="The agent explains one plan but the emitted action follows a different path.",
        evidence_hint="Reasoning and action diverge.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.PREMATURE_TERMINATION,
        category=MASTCategory.TERMINATION_AND_VERIFICATION,
        example="The system ends the interaction before it verifies the output or shares the missing result.",
        evidence_hint="Task ends too early.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.WEAK_VERIFICATION,
        category=MASTCategory.TERMINATION_AND_VERIFICATION,
        example="A reviewer only skims the code and does not run tests or inspect task constraints.",
        evidence_hint="Verification exists but is superficial or incomplete.",
    ),
    MASTFewShotExample(
        mode=MASTFailureMode.NO_OR_INCORRECT_VERIFICATION,
        category=MASTCategory.TERMINATION_AND_VERIFICATION,
        example="The verifier declares success even though the output still fails the visible test.",
        evidence_hint="Verification is missing or plainly wrong.",
    ),
]


class MASTModeAssessment(BaseModel):
    """Per-mode judgement returned by the classifier."""

    mode: MASTFailureMode
    present: bool
    evidence: str = ""
    confidence: float = 0.0


class MASTJudgement(BaseModel):
    """Structured LLM-as-Judge output for a single trace."""

    task_successful: bool
    summary: str
    assessments: list[MASTModeAssessment] = Field(default_factory=list)
    primary_failure_modes: list[MASTFailureMode] = Field(default_factory=list)
    overall_confidence: float = 0.0


class MASTClassifier:
    """MAST-aligned judge that prefers structured model output."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def classify(self, trace_text: str) -> MASTJudgement:
        """Classify one trace with a structured MAST judgment."""

        if self.model is None:
            return self._fallback_judgement(trace_text)

        messages = self._build_messages(trace_text)
        structured_model = getattr(self.model, "with_structured_output", None)
        if callable(structured_model):
            try:
                result = structured_model(MASTJudgement).invoke(messages)
                if isinstance(result, MASTJudgement):
                    return result
                return MASTJudgement.model_validate(result)
            except Exception:
                pass

        response = self.model.invoke(messages) if hasattr(self.model, "invoke") else self.model(messages)
        return self._coerce_response(response, trace_text)

    def _build_messages(self, trace_text: str) -> list[dict[str, str]]:
        definitions = "\n".join(
            f"- {mode.value}: {definition}" for mode, definition in MAST_FAILURE_MODE_DEFINITIONS.items()
        )
        examples = "\n".join(
            f"- {example.mode.value} [{example.category.value}]\n  Example: {example.example}\n  Evidence hint: {example.evidence_hint}"
            for example in MAST_FEW_SHOT_EXAMPLES
        )
        prompt = (
            "You are an expert judge for coordination failures in multi-agent systems. "
            "Follow a high-agreement evaluation style: independently inspect the trace, decide yes/no for every failure mode, "
            "then provide a compact final judgment with explicit evidence. "
            "Only mark a failure mode as present if the trace contains direct evidence.\n\n"
            f"Definitions:\n{definitions}\n\n"
            f"Few-shot examples:\n{examples}\n\n"
            "Return a structured judgment with task_successful, summary, assessments, primary_failure_modes, and overall_confidence."
        )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Trace:\n{trace_text}"},
        ]

    def build_prompt(self, trace_text: str) -> str:
        """Build the full judge prompt as a plain string for logging or debugging."""

        return "\n".join(message["content"] for message in self._build_messages(trace_text))

    def _coerce_response(self, response: Any, trace_text: str) -> MASTJudgement:
        if isinstance(response, MASTJudgement):
            return response
        if isinstance(response, dict):
            try:
                return MASTJudgement.model_validate(response)
            except Exception:
                pass
        content = getattr(response, "content", None)
        if isinstance(content, dict):
            try:
                return MASTJudgement.model_validate(content)
            except Exception:
                pass
        text = str(content if content is not None else response)
        return MASTJudgement(
            task_successful="success" in text.lower(),
            summary=text[:500],
            assessments=self._empty_assessments(),
            primary_failure_modes=[],
            overall_confidence=0.0,
        )

    def _fallback_judgement(self, trace_text: str) -> MASTJudgement:
        matches = self._keyword_matches(trace_text)
        return MASTJudgement(
            task_successful=not bool(matches),
            summary=(
                "Heuristic fallback only. Configure a judge model to obtain MAST-aligned judgments."
            ),
            assessments=[
                MASTModeAssessment(
                    mode=mode,
                    present=mode in matches,
                    evidence="Keyword match in trace text." if mode in matches else "",
                    confidence=0.35 if mode in matches else 0.05,
                )
                for mode in MASTFailureMode
            ],
            primary_failure_modes=matches,
            overall_confidence=0.15,
        )

    def _keyword_matches(self, trace_text: str) -> list[MASTFailureMode]:
        lowered = trace_text.lower()
        matches: list[MASTFailureMode] = []
        keyword_map = {
            MASTFailureMode.STEP_REPETITION: ["repeat", "repetition"],
            MASTFailureMode.INFORMATION_WITHHOLDING: ["withholding", "withheld", "did not share"],
            MASTFailureMode.WEAK_VERIFICATION: ["reviewer", "verify", "verification"],
            MASTFailureMode.NO_OR_INCORRECT_VERIFICATION: ["did not run tests", "incorrect verification", "no verification"],
        }
        for mode, keywords in keyword_map.items():
            if any(keyword in lowered for keyword in keywords):
                matches.append(mode)
        return matches

    def _empty_assessments(self) -> list[MASTModeAssessment]:
        return [MASTModeAssessment(mode=mode, present=False) for mode in MASTFailureMode]
