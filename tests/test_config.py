from pathlib import Path

from utils.config import load_experiment_config


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
