# Koordinationsfehler in LLM-basierten Multi-Agent-Systemen

Dieses Repository ist das experimentelle Gerüst für die Masterarbeit **"Koordinationsfehler in LLM-basierten Multi-Agent-Systemen"**.

## Forschungsziel

Vergleich der Koordinationsleistung von:

- LangGraph
- AutoGen (AG2)
- CrewAI
- MetaGPT

mit reproduzierbaren Läufen, Tracing und MAST-basierter Fehleranalyse.

## Transparenzhinweis

Teile dieses Projekts wurden mit Unterstützung von GitHub Copilot erzeugt und anschließend fachlich überarbeitet.

## Installation (empfohlen)

### Python-Version

- Projekt ist auf **Python 3.12** ausgerichtet (`requires-python: >=3.12,<3.13`).

### Mit `uv`

```bash
uv python install 3.12
uv sync --extra dev
```

Optional für native AutoGen + CrewAI Zusatzabhängigkeiten:

```bash
uv sync --extra dev --extra frameworks
```

## Docker-Quickstart (empfohlen für reproduzierbare lokale Tests)

Wenn du die gesamte Testumgebung ohne lokale Python-Installation starten möchtest, kannst du den Container-Workflow verwenden.

```powershell
# 1) Konfiguration vorbereiten
Copy-Item .env.example .env

# 2) Container bauen und Smoke-Test starten
docker compose up --build app

# 3) Tests ausführen
docker compose run --rm tests

# 4) Erste Validation im Container starten
docker compose run --rm validation
```

Die Docker-Umgebung verwendet die Datei [docker-compose.yml](docker-compose.yml) und den Containeraufbau aus [Dockerfile](Dockerfile).

## Windows-VM Quickstart (PowerShell)

Wenn `winget` fehlt: Git/Python manuell installieren und dann `uv` per Script.

```powershell
# im Repo-Ordner
uv python install 3.12
uv sync --extra dev --extra frameworks
Copy-Item .env.example .env
notepad .env
uv run python scripts/check_runtime.py
uv run python scripts/smoke_run.py
uv run python experiments/run_validation.py --tasks 3 --runs 3 --model-name gpt-4o-mini --temperature 0.0
```

## Konfiguration

```bash
cp .env.example .env
```

Mindestens setzen:

- `MAS_LLM_API_KEY`
- `MAS_MODEL_NAME` (z. B. `gpt-4o-mini`)
- `MAS_NUM_RUNS` (z. B. `3`)
- `MAS_MAST_JUDGE_ENABLED=true`

## MetaGPT-Hinweis (wichtig)

Auf macOS/Windows kann MetaGPT lokal problematisch sein (Plattform-/Dependency-Konflikte).
Dafür ist ein Linux/x86 Docker-Pfad vorbereitet:

```bash
docker compose -f docker-compose.metagpt.yml build
docker compose -f docker-compose.metagpt.yml run --rm metagpt-runtime python scripts/check_runtime.py --frameworks metagpt --strict
```

**Thesis rule:** Scaffold-/Fallback-Läufe gehören nicht in die primären Ergebnis-Tabellen. MetaGPT wird standardmäßig aus der sauberen Reanalyse ausgeschlossen, bis native Läufe vorliegen.

## Experimental Protocol (Validation -> Baseline -> Mitigation)

1. **Smoke**: `scripts/smoke_run.py`
2. **Validation**: `experiments/run_validation.py` (LangGraph, N=3..5)
3. **Baseline**: `experiments/run_coordination_baseline.py` (N=30..50)
4. **Judge validation (Phase C)**: `experiments/build_phase_c_sample_v4.py` + Double-Coding
5. **Mitigation**: `experiments/run_coordination_mitigation.py`

## Thesis-ready Auswertung vorhandener Ergebnisse

```bash
# 1) Baseline ohne Scaffold/MetaGPT-Kontamination neu auswerten
uv run python experiments/reanalyze_baseline.py \
  results/coordination_baseline_2026-08-09/experiments \
  --output-dir results/coordination_baseline_2026-08-09/reports/thesis_clean_v1 \
  --exclude-frameworks metagpt

# 2) Phase-C-Sample neu ziehen (Success + Failure, n=60)
uv run python experiments/build_phase_c_sample_v4.py \
  results/coordination_baseline_2026-08-09/experiments \
  --output-dir results/judge_validation_phase_c_v4 \
  --sample-size 60 \
  --exclude-frameworks metagpt

# 3) Readiness der laufenden Annotation prüfen
uv run python experiments/phase_c_readiness_status.py
```

## Zentrale Dokumentation

- Thesis Readiness Plan: [docs/thesis_readiness_plan.md](docs/thesis_readiness_plan.md)
- Protocol Deviations: [docs/protocol_deviations.md](docs/protocol_deviations.md)
- Installation: [docs/install_guide.md](docs/install_guide.md)
- Reproduzierbarkeit: [docs/reproducibility.md](docs/reproducibility.md)
- Nutzung: [docs/user_guide.md](docs/user_guide.md)
- Coordination-Suite Design: [docs/task_design.md](docs/task_design.md)
- Task-Qualitätsbewertung: [docs/task_quality_assessment.md](docs/task_quality_assessment.md)
- Studienprotokoll: [docs/study_protocol.md](docs/study_protocol.md)
- Coordination-Suite Codebook: [docs/coordination_suite_codebook.md](docs/coordination_suite_codebook.md)
- Annotation und Judge-Validierung: [docs/annotation_protocol.md](docs/annotation_protocol.md)
- Phase-C Data Card: [docs/phase_c_data_card.md](docs/phase_c_data_card.md)
- Phase-C Lineage Snapshot: [docs/phase_c_lineage_2026_08_02.md](docs/phase_c_lineage_2026_08_02.md)

## Referenzen

- MAST Paper: [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)
- MAST Taxonomie/Referenz: [GitHub](https://github.com/multi-agent-systems-failure-taxonomy/MAST)
