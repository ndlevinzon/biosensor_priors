#!/usr/bin/env bash
# Thin HPC wrapper around scripts/clean_pipeline_artifacts.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python scripts/clean_pipeline_artifacts.py "$@"
