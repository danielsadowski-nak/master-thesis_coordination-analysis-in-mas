"""Independent task-success scoring from Coordination Suite success criteria.

Success for the thesis primary analyses must not depend solely on the MAST
judge or on runner-claimed success. This module implements a transparent,
keyword-anchored approximation of the codebook success criteria that can be
audited and replaced by full human adjudication.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# Deterministic, auditable criteria anchors derived from coordination_suite_v1.
# Each criterion is a list of alternative evidence patterns (any match counts).
TASK_SUCCESS_CRITERIA: dict[str, list[list[str]]] = {
    "coord-info-asymmetry-constraint": [
        [r"ip address", r"ip addresses", r"logging.*ip", r"no[- ]?ip", r"forbid.*ip"],
        [r"mitigation", r"do not store", r"redact", r"anonym", r"compliance"],
        [r"verification", r"checklist", r"compliance check", r"audit"],
    ],
    "coord-conflicting-done-criteria": [
        [r"unified", r"shared definition", r"single definition", r"one definition of done"],
        [r"unit test", r"regression"],
        [r"not complete", r"before completion", r"after reconciliation", r"reconcile"],
    ],
    "coord-unclear-handoff-incomplete-artifact": [
        [r"acceptance criteria", r"rollback", r"file path", r"scope"],
        [r"missing", r"incomplete", r"unclear", r"not specified"],
        [r"cannot execute", r"before execution", r"clarif", r"handoff"],
    ],
    "coord-weak-final-verification": [
        [r"check 1", r"first check", r"verification 1", r"test", r"inspect"],
        [r"check 2", r"second check", r"verification 2", r"independent", r"two"],
        [r"evidence", r"because", r"observed", r"result"],
    ],
    "coord-role-overlap-duplicate-actions": [
        [r"owner", r"responsibility", r"owns the final"],
        [r"final response", r"consolidated", r"single response", r"one coherent"],
        [r"no contradiction", r"consistent", r"merged"],
    ],
    "coord-ambiguous-termination-condition": [
        [r"termination", r"stopping", r"both branches", r"acceptance criteria", r"blocker"],
        [r"not complete", r"unverified", r"cannot claim completion", r"still missing"],
        [r"continue", r"escalate", r"next action"],
    ],
    "coord-combined-asymmetry-verification": [
        [r"rate[- ]?limit", r"5 times", r"5/min", r"five calls"],
        [r"verification", r"check"],
        [r"batch", r"cache", r"fewer calls", r"low-call", r"reduce calls"],
    ],
    "coord-clarification-before-execution": [
        [r"\?", r"clarif", r"what is the", r"please specify", r"need to know"],
        [r"environment", r"rollback", r"downtime"],
        [r"cannot finalize", r"before execution", r"until clarification", r"not ready"],
    ],
}


def extract_task_id(benchmark: str | None) -> str:
    """Extract task id from benchmark names like coordination_suite/coord-...."""

    if not benchmark:
        return ""
    text = str(benchmark)
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def normalize_output_for_criteria_scoring(final_output: Any) -> str:
    """Normalize model output for criteria scoring.

    Rule (applied uniformly across conditions):
    - If final_output is parseable JSON object with a non-empty final_answer field,
      score only final_answer.
    - Otherwise score the raw final_output text.
    """

    raw_text = "" if final_output is None else str(final_output)
    stripped = raw_text.strip()
    if not stripped:
        return raw_text

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return raw_text

    if not isinstance(parsed, dict):
        return raw_text

    final_answer = parsed.get("final_answer")
    if final_answer is None:
        return raw_text

    final_answer_text = str(final_answer).strip()
    if not final_answer_text:
        return raw_text
    return final_answer_text


def score_output_against_criteria(task_id: str, final_output: str) -> dict[str, Any]:
    """Score one output against the independent criteria anchors."""

    criteria = TASK_SUCCESS_CRITERIA.get(task_id)
    if not criteria:
        return {
            "task_id": task_id,
            "criteria_success": None,
            "criteria_matched": 0,
            "criteria_total": 0,
            "criteria_details": [],
            "scorer": "missing_task_mapping",
        }

    text = final_output or ""
    details: list[dict[str, Any]] = []
    matched = 0
    for index, alternatives in enumerate(criteria, start=1):
        hit = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in alternatives)
        details.append({"criterion_index": index, "matched": bool(hit), "patterns": alternatives})
        if hit:
            matched += 1

    return {
        "task_id": task_id,
        "criteria_success": matched == len(criteria),
        "criteria_matched": matched,
        "criteria_total": len(criteria),
        "criteria_details": details,
        "scorer": "codebook_keyword_anchors_v1",
    }


def annotate_criteria_success(df: pd.DataFrame) -> pd.DataFrame:
    """Add independent success columns to a run-level dataframe."""

    if df.empty:
        out = df.copy()
        out["task_id"] = []
        out["criteria_success"] = []
        out["criteria_matched"] = []
        out["criteria_total"] = []
        out["criteria_scorer"] = []
        return out

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        task_id = extract_task_id(row.get("benchmark") or row.get("task_id"))
        scoring_text = normalize_output_for_criteria_scoring(row.get("final_output"))
        scored = score_output_against_criteria(task_id, scoring_text)
        rows.append(
            {
                "task_id": scored["task_id"],
                "criteria_success": scored["criteria_success"],
                "criteria_matched": scored["criteria_matched"],
                "criteria_total": scored["criteria_total"],
                "criteria_scorer": scored["scorer"],
            }
        )
    scored_df = pd.DataFrame(rows)
    out = df.reset_index(drop=True).copy()
    for column in scored_df.columns:
        out[column] = scored_df[column]
    return out


def load_suite_success_criteria(suite_path: Path | str) -> dict[str, list[str]]:
    """Load human-readable success criteria text from the suite JSONL."""

    path = Path(suite_path)
    mapping: dict[str, list[str]] = {}
    if not path.exists():
        return mapping
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        mapping[str(payload["task_id"])] = list(payload.get("success_criteria") or [])
    return mapping
