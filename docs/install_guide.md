# Installationsanleitung

Diese Anleitung ist auf den aktuellen Projektstand (Python 3.12, `uv`, Validation-first Workflow) aktualisiert.

## Voraussetzungen

- Python 3.12
- Git
- optional Docker Desktop (für MetaGPT-Containerpfad)
- optional `uv` (empfohlen)

## 1) Installation mit `uv` (empfohlen)

```bash
uv python install 3.12
uv sync --extra dev
```

Optional (native AutoGen + CrewAI):

```bash
uv sync --extra dev --extra frameworks
```

## 2) Windows-VM (PowerShell)

```powershell
uv python install 3.12
uv sync --extra dev --extra frameworks
Copy-Item .env.example .env
notepad .env
```

Wenn `winget` nicht vorhanden ist, installiere Git/Python manuell über die offiziellen Installer und danach `uv`:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

## 3) `.env` konfigurieren

Pflichtwerte für echte Modellläufe:

- `MAS_LLM_API_KEY`
- `MAS_MODEL_NAME=gpt-4o-mini` (für günstige Validation)
- `MAS_NUM_RUNS=3`
- `MAS_MAST_JUDGE_ENABLED=true`

## 4) Setup prüfen

```bash
uv run python --version
uv run python scripts/check_runtime.py
uv run python scripts/smoke_run.py
uv run python -m pytest -q
```

## 5) Erste Validation starten

```bash
uv run python experiments/run_validation.py --tasks 3 --runs 3 --model-name gpt-4o-mini --temperature 0.0
```

## 6) MetaGPT über Docker (Linux/x86 Runtime)

Für Umgebungen, in denen MetaGPT nicht nativ läuft:

```bash
docker compose -f docker-compose.metagpt.yml build
docker compose -f docker-compose.metagpt.yml run --rm metagpt-runtime python scripts/check_runtime.py --frameworks metagpt --strict
```

MetaGPT-only Baseline:

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
