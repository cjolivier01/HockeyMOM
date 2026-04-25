#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "${SCRIPT_DIR}/../.bazel_setup.sh"

if [[ -z "${PYTHON_BIN_PATH:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN_PATH="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN_PATH="$(command -v python3)"
  else
    echo "error: could not find python or python3 on PATH" >&2
    exit 1
  fi
fi

# pybind11_bazel's python_configure looks for this env var.
export PYTHON_BIN_PATH

bazelisk "$@" ${BAZEL_FLAGS}
