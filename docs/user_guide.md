# Kurzanleitung für die Nutzung

Diese Anleitung zeigt die aktuellen Kernworkflows für Validation, Baseline und Mitigation.

## 1) Konfiguration vorbereiten

```bash
cp .env.example .env
```

Wichtige Variablen:

- `MAS_LLM_API_KEY`
- `MAS_MODEL_NAME` (z. B. `gpt-4o-mini`)
- `MAS_NUM_RUNS` (z. B. `3` für Validation)
- `MAS_MAST_JUDGE_ENABLED=true`

## 2) Runtime prüfen und Smoke ausführen

```bash
uv run python scripts/check_runtime.py
uv run python scripts/smoke_run.py
```

## 3) Validation (empfohlener Start)

```bash
uv run python experiments/run_validation.py --tasks 3 --runs 3 --model-name gpt-4o-mini --temperature 0.0
```

## 4) Baseline und Mitigation

Baseline:

```bash
uv run python experiments/run_full_baseline_comparison.py \
  --swe-source /pfad/zu/swe_bench_verified.jsonl \
  --gaia-source /pfad/zu/gaia.jsonl
```

Mitigation:

```bash
uv run python experiments/run_mitigation_comparison.py \
  --swe-source /pfad/zu/swe_bench_verified.jsonl \
  --gaia-source /pfad/zu/gaia.jsonl
```

## 5) MetaGPT-Workflow (falls nativ nicht möglich)

```bash
docker compose -f docker-compose.metagpt.yml build
docker compose -f docker-compose.metagpt.yml run --rm metagpt-runtime python scripts/check_runtime.py --frameworks metagpt --strict
```

## 6) Statistik und Power

```bash
uv run python experiments/generate_statistical_report.py results/experiments
uv run python experiments/power_analysis.py
```

## 7) Vorbereitung für einen neuen Copilot-Chat

Wenn du einen neuen Chat öffnest, nutze am besten diesen Startprompt:

```text
Du hilfst mir beim Projekt "Koordinationsfehler in LLM-basierten Multi-Agent-Systemen".
Bitte arbeite auf Basis des aktuellen Repos und folge docs/reproducibility.md als primärem Runbook.
Ziel: Validation -> Baseline -> Mitigation mit Python 3.12 und .env-Konfiguration.
Falls MetaGPT nativ nicht läuft, nutze den Docker-Pfad aus docker-compose.metagpt.yml.
Bitte nenne mir zuerst den konkreten nächsten auszuführenden Befehl.
```

Damit hat der neue Chat sofort den richtigen Kontext und den aktuellen Ablauf.
