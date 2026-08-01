from pathlib import Path

from utils.config import apply_llm_runtime_environment, load_experiment_config


def test_load_experiment_config_reads_environment_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAS_NUM_RUNS", "7")
    monkeypatch.setenv("MAS_LLM_API_KEY", "env-key")
    monkeypatch.setenv("MAS_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("MAS_MAST_JUDGE_ENABLED", "true")
    monkeypatch.setenv("MAS_MAST_JUDGE_TEMPERATURE", "0.2")
    monkeypatch.setenv("MAS_MAST_JUDGE_MAX_RETRIES", "4")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "num_runs: 3\nmodel_name: gpt-4.1\nllm_api_key: yaml-key\n",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.num_runs == 7
    assert config.llm_api_key == "env-key"
    assert config.model_name == "gpt-4o-mini"
    assert config.mast_judge_enabled is True
    assert config.mast_judge_temperature == 0.2
    assert config.mast_judge_max_retries == 4


def test_load_experiment_config_resolves_repo_paths_from_foreign_cwd(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    config = load_experiment_config(Path("experiments/config.yaml"))

    assert config.output_dir == repo_root / "results/experiments"
    assert config.trace_dir == repo_root / "results/traces"
    assert config.checkpoint_dir == repo_root / "results/checkpoints"


def test_apply_llm_runtime_environment_loads_repo_env(monkeypatch, tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    original = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    monkeypatch.delenv("MAS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        env_path.write_text("MAS_LLM_API_KEY=test-key-from-env\n", encoding="utf-8")
        result = apply_llm_runtime_environment()
    finally:
        if original is None:
            env_path.unlink(missing_ok=True)
        else:
            env_path.write_text(original, encoding="utf-8")

    assert result["api_key_set"] is True
    assert Path(env_path).exists() or original is None
