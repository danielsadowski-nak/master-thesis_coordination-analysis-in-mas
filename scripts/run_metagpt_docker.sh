#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not installed." >&2
  exit 1
fi

if [[ ! -f ".env" ]]; then
  echo ".env is required. Create it first: cp .env.example .env" >&2
  exit 1
fi

docker compose -f docker-compose.metagpt.yml build
docker compose -f docker-compose.metagpt.yml run --rm metagpt-runtime "$@"
