"""Basic smoke tests for the thesis scaffold."""

from evaluation.mast_classifier import MASTClassifier, MASTFailureMode
from frameworks.base_runner import RunMetrics, StepKind, TraceResult, TraceStep


def test_trace_result_model_round_trip() -> None:
    result = TraceResult(
        success=True,
        final_output="done",
        full_trace=[TraceStep(step_index=1, kind=StepKind.FINAL, content="done")],
        metrics=RunMetrics(latency_seconds=1.0, total_tokens=5),
        raw_log_path="results/traces/example.jsonl",
    )

    assert result.success is True
    assert result.full_trace[0].kind == StepKind.FINAL


def test_mast_classifier_fallback_returns_all_modes() -> None:
    classifier = MASTClassifier()
    judgement = classifier.classify("The reviewer did not run tests and the agent repeated the same step.")

    assert len(judgement.assessments) == len(MASTFailureMode)
