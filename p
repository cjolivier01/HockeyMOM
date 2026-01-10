#!/usr/bin/env bash
set -euo pipefail

PY_BIN="${PY_BIN:-python3}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  PY_BIN="python"
fi

PYTHONPATH="$(pwd):${PYTHONPATH:-}" exec "${PY_BIN}" "$@"
