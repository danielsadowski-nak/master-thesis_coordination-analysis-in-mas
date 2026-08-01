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

## Experimental Protocol (Validation -> Baseline -> Mitigation)

1. **Smoke**: `scripts/smoke_run.py`
2. **Validation**: `experiments/run_validation.py` (LangGraph, N=3..5)
3. **Baseline**: `experiments/run_full_baseline_comparison.py` (N=30..50)
4. **Mitigation**: `experiments/run_mitigation_comparison.py`

## Zentrale Dokumentation

- Installation: [docs/install_guide.md](docs/install_guide.md)
- Reproduzierbarkeit: [docs/reproducibility.md](docs/reproducibility.md)
- Nutzung: [docs/user_guide.md](docs/user_guide.md)
- Coordination-Suite Design: [docs/task_design.md](docs/task_design.md)

## Referenzen

- MAST Paper: [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)
- MAST Taxonomie/Referenz: [GitHub](https://github.com/multi-agent-systems-failure-taxonomy/MAST)
