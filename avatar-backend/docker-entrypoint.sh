#!/usr/bin/env bash
set -euo pipefail

cd /app
mkdir -p /app/audio /app/.cache/huggingface /app/.cache/modelscope

if [[ "${UVICORN_RELOAD:-false}" == "true" ]]; then
    exec python -m uvicorn fastapi_app:app \
        --host 0.0.0.0 \
        --port "${PORT:-8109}" \
        --reload \
        --reload-dir /app \
        --reload-include '*.py'
fi

exec python -m uvicorn fastapi_app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8109}"
