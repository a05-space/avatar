#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/liu/dwc/avatar/avatar-backend"
CONDA_ENV_PATH="$PROJECT_DIR/.conda"

cd "$PROJECT_DIR"
source "$CONDA_ENV_PATH/bin/activate"
exec uv run fastapi_app.py
