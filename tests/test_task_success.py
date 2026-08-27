"""Tests for independent task-success scoring."""

from __future__ import annotations

from evaluation.task_success import extract_task_id, normalize_output_for_criteria_scoring, score_output_against_criteria


def test_extract_task_id_from_benchmark_path() -> None:
    assert extract_task_id("coordination_suite/coord-info-asymmetry-constraint") == "coord-info-asymmetry-constraint"


def test_extract_task_id_from_phase_d_benchmark_none_condition() -> None:
    assert (
        extract_task_id("coordination_suite/coord-role-overlap-duplicate-actions/none")
        == "coord-role-overlap-duplicate-actions"
    )


def test_extract_task_id_from_phase_d_benchmark_strategy_condition() -> None:
    assert (
        extract_task_id("coordination_suite/coord-role-overlap-duplicate-actions/structured_output_validation")
        == "coord-role-overlap-duplicate-actions"
    )


def test_score_output_against_criteria_empty_output_is_not_success() -> None:
    scored = score_output_against_criteria("coord-info-asymmetry-constraint", "")

    assert scored["criteria_success"] is False
    assert scored["criteria_matched"] == 0
    assert scored["criteria_total"] == 3


def test_score_output_against_criteria_matching_output_is_success() -> None:
    output = (
        "We will forbid IP address logging and add redaction as mitigation. "
        "A compliance checklist and verification step will be executed before final delivery."
    )

    scored = score_output_against_criteria("coord-info-asymmetry-constraint", output)

    assert scored["criteria_success"] is True
    assert scored["criteria_matched"] == scored["criteria_total"]


def test_normalize_output_uses_json_final_answer_when_present() -> None:
    output = '{"plan":"x", "final_answer":"Use mitigation and verification checklist."}'

    normalized = normalize_output_for_criteria_scoring(output)

    assert normalized == "Use mitigation and verification checklist."


def test_normalize_output_falls_back_when_json_has_no_final_answer() -> None:
    output = '{"plan":"x", "summary":"y"}'

    normalized = normalize_output_for_criteria_scoring(output)

    assert normalized == output


def test_normalize_output_keeps_free_text_unchanged() -> None:
    output = "Plain text response without JSON envelope."

    normalized = normalize_output_for_criteria_scoring(output)

    assert normalized == output


def test_normalize_output_falls_back_for_empty_json_final_answer() -> None:
    output = '{"final_answer":"   ", "plan":"x"}'

    normalized = normalize_output_for_criteria_scoring(output)

    assert normalized == output
