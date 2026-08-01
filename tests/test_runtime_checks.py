"""Tests for runtime dependency checks."""

from __future__ import annotations

import pytest

from utils.runtime_checks import assert_framework_runtime_ready, detect_missing_framework_imports


def test_detect_missing_framework_imports_reports_known_missing_modules() -> None:
    missing = detect_missing_framework_imports(["autogen", "crewai", "metagpt"])
    assert isinstance(missing, dict)
    for framework, modules in missing.items():
        assert framework in {"autogen", "crewai", "metagpt"}
        assert modules


def test_assert_framework_runtime_ready_passes_for_langgraph() -> None:
    assert_framework_runtime_ready(["langgraph"])


def test_assert_framework_runtime_ready_raises_for_missing_frameworks() -> None:
    with pytest.raises(RuntimeError):
        assert_framework_runtime_ready(["autogen", "crewai", "metagpt"])
