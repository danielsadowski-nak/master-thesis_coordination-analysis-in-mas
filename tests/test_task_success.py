"""Tests for independent task-success scoring."""

from __future__ import annotations

from evaluation.task_success import extract_task_id, score_output_against_criteria


def test_extract_task_id_from_benchmark_path() -> None:
    assert extract_task_id("coordination_suite/coord-info-asymmetry-constraint") == "coord-info-asymmetry-constraint"


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
