"""Tests for Coordination Suite thesis runners."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_coordination_baseline import select_coordination_tasks as select_baseline_tasks
from experiments.run_coordination_mitigation import select_coordination_tasks as select_mitigation_tasks


COORDINATION_SOURCE = Path("/workspace/data/coordination_tasks/coordination_suite_v1.jsonl")


def test_select_coordination_tasks_loads_default_suite() -> None:
    tasks = select_baseline_tasks(COORDINATION_SOURCE)

    assert len(tasks) >= 8
    assert tasks[0].metadata["benchmark"] == "coordination_suite"


def test_select_coordination_tasks_supports_fixed_ids() -> None:
    tasks = select_mitigation_tasks(
        COORDINATION_SOURCE,
        task_ids=["coord-weak-final-verification", "coord-clarification-before-execution"],
    )

    assert [task.task_id for task in tasks] == [
        "coord-weak-final-verification",
        "coord-clarification-before-execution",
    ]


def test_select_coordination_tasks_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError):
        select_baseline_tasks(COORDINATION_SOURCE, task_ids=["missing-task"])