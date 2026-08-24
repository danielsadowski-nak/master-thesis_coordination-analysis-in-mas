# Reproducibility Bundle

This document is the canonical runbook for reproducible setup and first experiments.

## 1) Environment baseline

- Python: **3.12**
- Dependency manager: **uv** (preferred)
- Config: `.env` based

Install:

```bash
uv python install 3.12
uv sync --extra dev
```

Optional framework extras (AutoGen + CrewAI):

```bash
uv sync --extra dev --extra frameworks
```

## 2) Runtime config

```bash
cp .env.example .env
```

Minimum for real runs:

- `MAS_LLM_API_KEY`
- `MAS_MODEL_NAME=gpt-4o-mini`
- `MAS_NUM_RUNS=30`
- `MAS_MAST_JUDGE_ENABLED=true`

Optional:

- `MAS_LANGSMITH_ENABLED=true`
- `MAS_LANGSMITH_PROJECT=...`
- `MAS_LANGSMITH_API_KEY=...`
- `MAS_LANGSMITH_ENDPOINT=...`
- `MAS_LANGSMITH_TAGS=validation,thesis`
- `MAS_MAST_JUDGE_TEMPERATURE=0.0`
- `MAS_MAST_JUDGE_MAX_RETRIES=2`

## 3) Validation-first execution path

Smoke test:

```bash
uv run python scripts/smoke_run.py
```

Runtime checks:

```bash
uv run python scripts/check_runtime.py
```

Low-cost validation:

```bash
uv run python experiments/run_validation.py --tasks 3 --runs 3 --model-name gpt-4o-mini --temperature 0.0
```

## 4) Full baseline and mitigation

Primary internal-validity baseline on the Coordination Suite:

```bash
uv run python experiments/run_coordination_baseline.py \
  --num-runs 30 \
  --model-name gpt-4o-mini \
  --mast-judge-enabled
```

Primary internal-validity mitigation study on the Coordination Suite:

```bash
uv run python experiments/run_coordination_mitigation.py \
  --num-runs 30 \
  --model-name gpt-4o-mini \
  --mast-judge-enabled \
  --thesis-strict

For smoke/debug-only mitigation runs (not thesis evidence), add `--allow-smoke`.
```

Secondary external-validity baseline:

Baseline:

```bash
uv run python experiments/run_full_baseline_comparison.py \
  --swe-source /path/to/swe_bench_verified.jsonl \
  --gaia-source /path/to/gaia.jsonl
```

Mitigation:

```bash
uv run python experiments/run_mitigation_comparison.py \
  --swe-source /path/to/swe_bench_verified.jsonl \
  --gaia-source /path/to/gaia.jsonl
```

## 4.1) Judge validation workflow

Generate an annotation template from completed results:

```bash
uv run python experiments/run_judge_validation.py \
  results/validation_YYYYMMDD_HHMMSS/experiments \
  --output-dir results/judge_validation \
  --sample-size 40 \
  --real-model-only \
  --template-only
```

After manual annotation, compute agreement:

```bash
uv run python experiments/run_judge_validation.py \
  results/validation_YYYYMMDD_HHMMSS/experiments \
  --output-dir results/judge_validation \
  --real-model-only \
  --annotations-csv results/judge_validation/annotation_template.csv
```

## 5) MetaGPT on constrained platforms

If native MetaGPT install is not feasible (e.g., dependency/platform conflicts), use Docker Linux/x86 runtime:

```bash
docker compose -f docker-compose.metagpt.yml build
docker compose -f docker-compose.metagpt.yml run --rm metagpt-runtime python scripts/check_runtime.py --frameworks metagpt --strict
```

Run MetaGPT-only baseline:

```bash
docker compose -f docker-compose.metagpt.yml run --rm metagpt-runtime \
  python experiments/run_full_baseline_comparison.py \
  --swe-source /workspace/data/swe_bench_verified.jsonl \
  --gaia-source /workspace/data/gaia.jsonl \
  --frameworks metagpt \
  --num-runs 30 \
  --model-name gpt-4o-mini \
  --temperature 0.0 \
  --require-native-frameworks
```

## 6) Artifacts

Validation writes to `results/validation_*`:

- `run_manifest.json`
- `reports/validation_summary.json`
- `reports/all_runs.csv` (if records exist)
- run payloads and traces under `experiments/` and `traces/`

Judge-validation writes to `results/judge_validation/`:

- `annotation_template.csv`
- `judge_validation_summary.json`
- `judge_validation_per_mode.csv`
- `judge_validation_disagreements.csv`
- `judge_validation_report.md`
