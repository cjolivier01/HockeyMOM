#!/usr/bin/env bash

set -euo pipefail

repo_root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

orig_home="${HOME:-}"
resolve_python() {
  local candidate
  for candidate in \
    "${PYTHON_BIN_PATH:-}" \
    "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}" \
    "$(command -v python 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)" \
    "${orig_home:+${orig_home}/miniforge3/bin/python}" \
    "/home/${USER:-}/miniforge3/bin/python" \
    "/home/colivier/miniforge3/bin/python"
  do
    if [ -z "${candidate}" ] || [ ! -x "${candidate}" ]; then
      continue
    fi
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

python_bin="$(resolve_python)" || {
  echo "Unable to find a python interpreter with torch available." >&2
  exit 1
}

"${python_bin}" -m pytest \
  "${repo_root}/tests/test_mot_video_prefetch.py" \
  -q \
  -c "${repo_root}/pyproject.toml"
