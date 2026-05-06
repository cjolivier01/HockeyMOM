#!/usr/bin/env bash

resolve_repo_python() {
  local required_modules=("$@")
  local candidate
  local -a candidates=()

  if [[ -n "${PYTHON_BIN_PATH:-}" ]]; then
    candidates+=("${PYTHON_BIN_PATH}")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python" "${CONDA_PREFIX}/bin/python3")
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
      continue
    fi
    if [[ "${#required_modules[@]}" -eq 0 ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    if "${candidate}" - "${required_modules[@]}" <<'PY' >/dev/null 2>&1
import importlib
import sys

for module_name in sys.argv[1:]:
    importlib.import_module(module_name)
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  if [[ "${#required_modules[@]}" -eq 0 ]]; then
    echo "Unable to resolve a repo Python runtime. Set PYTHON_BIN_PATH or CONDA_PREFIX." >&2
  else
    echo "Unable to resolve a repo Python runtime with modules: ${required_modules[*]}." >&2
    echo "Set PYTHON_BIN_PATH or CONDA_PREFIX to the runtime Bazel was configured against." >&2
  fi
  return 1
}
