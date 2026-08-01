"""Configuration loading helpers.

Generated with GitHub Copilot assistance - reviewed and adapted by author.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Typed configuration for experiment runs."""

    framework: str = "langgraph"
    benchmark: str = "swe_bench_verified"
    task_description: str = "Solve the benchmark task with a controlled multi-agent workflow."
    max_steps: int = 25
    num_runs: int = 50
    seed: int = 42
    output_dir: Path = Path("results/experiments")
    trace_dir: Path = Path("results/traces")
    checkpoint_dir: Path = Path("results/checkpoints")
    benchmark_source: Path | None = None
    model_name: str = "gpt-4.1"
    temperature: float = 0.0
    judge_model_name: str = "gpt-4o"
    mast_judge_enabled: bool = False
    mast_judge_temperature: float = 0.0
    mast_judge_max_retries: int = 2
    langsmith_enabled: bool = False
    langsmith_project: str = "mas-coordination-analysis"
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_tags: tuple[str, ...] = ()
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Load a YAML configuration file into a typed model."""

    config_path = Path(config_path)
    _load_environment_file(config_path.parent)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    known_fields = set(ExperimentConfig.model_fields)
    standard_data = {key: value for key, value in data.items() if key in known_fields}
    extra = {key: value for key, value in data.items() if key not in known_fields}
    if extra:
        standard_data["extra"] = extra

    for field_name, env_names, parser in _ENV_FIELD_OVERRIDES:
        env_value = _read_environment_value(env_names)
        if env_value is not None:
            standard_data[field_name] = parser(env_value) if parser is not None else env_value

    if not standard_data.get("llm_api_key"):
        fallback_api_key = os.getenv("OPENAI_API_KEY")
        if fallback_api_key:
            standard_data["llm_api_key"] = fallback_api_key

    return ExperimentConfig.model_validate(standard_data)


def apply_llm_runtime_environment(*, api_key: str | None = None, base_url: str | None = None) -> dict[str, Any]:
    """Populate standard LLM environment variables from configuration values.

    This keeps the project flexible: if no key is configured, the experiment
    falls back to scaffold/fallback behavior rather than failing silently.
    """

    resolved_api_key = api_key or os.getenv("MAS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    resolved_base_url = base_url or os.getenv("MAS_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if resolved_api_key:
        os.environ["OPENAI_API_KEY"] = resolved_api_key
    if resolved_base_url:
        os.environ["OPENAI_BASE_URL"] = resolved_base_url
    return {"api_key_set": bool(resolved_api_key), "base_url": resolved_base_url}


def _load_environment_file(base_dir: Path) -> None:
    env_files = [base_dir / ".env"]
    env_file_override = os.getenv("MAS_ENV_FILE")
    if env_file_override:
        env_files.append(Path(env_file_override))
    repo_root = Path(__file__).resolve().parents[2]
    env_files.append(repo_root / ".env")

    for env_file in env_files:
        if env_file.exists():
            _parse_environment_file(env_file)
            return


def _parse_environment_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _read_environment_value(env_names: tuple[str, ...]) -> str | None:
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'.")


def _parse_tags(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


_ENV_FIELD_OVERRIDES: tuple[tuple[str, tuple[str, ...], Callable[[str], Any] | None], ...] = (
    ("framework", ("MAS_FRAMEWORK",), str),
    ("benchmark", ("MAS_BENCHMARK",), str),
    ("task_description", ("MAS_TASK_DESCRIPTION",), str),
    ("max_steps", ("MAS_MAX_STEPS",), int),
    ("num_runs", ("MAS_NUM_RUNS",), int),
    ("seed", ("MAS_SEED",), int),
    ("output_dir", ("MAS_OUTPUT_DIR",), Path),
    ("trace_dir", ("MAS_TRACE_DIR",), Path),
    ("checkpoint_dir", ("MAS_CHECKPOINT_DIR",), Path),
    ("benchmark_source", ("MAS_BENCHMARK_SOURCE",), Path),
    ("model_name", ("MAS_MODEL_NAME",), str),
    ("temperature", ("MAS_TEMPERATURE",), float),
    ("judge_model_name", ("MAS_JUDGE_MODEL_NAME",), str),
    ("mast_judge_enabled", ("MAS_MAST_JUDGE_ENABLED",), _parse_bool),
    ("mast_judge_temperature", ("MAS_MAST_JUDGE_TEMPERATURE",), float),
    ("mast_judge_max_retries", ("MAS_MAST_JUDGE_MAX_RETRIES",), int),
    ("langsmith_enabled", ("MAS_LANGSMITH_ENABLED",), _parse_bool),
    ("langsmith_project", ("MAS_LANGSMITH_PROJECT",), str),
    ("langsmith_api_key", ("MAS_LANGSMITH_API_KEY",), str),
    ("langsmith_endpoint", ("MAS_LANGSMITH_ENDPOINT",), str),
    ("langsmith_tags", ("MAS_LANGSMITH_TAGS",), _parse_tags),
    ("llm_api_key", ("MAS_LLM_API_KEY", "OPENAI_API_KEY"), str),
    ("llm_base_url", ("MAS_LLM_BASE_URL", "OPENAI_BASE_URL"), str),
)
