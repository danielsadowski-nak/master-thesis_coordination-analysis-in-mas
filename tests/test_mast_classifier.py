"""Tests for the MAST classifier prompt and fallback behaviour."""

from __future__ import annotations

from evaluation.mast_classifier import MASTClassifier, MAST_FEW_SHOT_EXAMPLES, MASTFailureMode


def test_mast_classifier_contains_all_examples() -> None:
    assert len(MAST_FEW_SHOT_EXAMPLES) == len(MASTFailureMode)


def test_mast_classifier_prompt_mentions_agreement_and_evidence() -> None:
    classifier = MASTClassifier()
    prompt = classifier.build_prompt("Agent repeated the same step and did not verify the result.")

    assert "high-agreement" in prompt.lower()
    assert "only mark a failure mode as present if the trace contains direct evidence" in prompt.lower()
    assert "3.2 Weak Verification" in prompt
