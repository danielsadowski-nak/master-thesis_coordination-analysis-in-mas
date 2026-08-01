FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev,frameworks]"

CMD ["python", "scripts/smoke_run.py"]
