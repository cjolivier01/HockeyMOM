#!/usr/bin/env bash

set -euo pipefail

repo_root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

source "${repo_root}/tests/python_runtime.sh"
python_bin="$(resolve_repo_python cv2 torch)"

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/cli-smoke.XXXXXX")"
trap 'rm -rf "${tmp_root}"' EXIT

export HOME="${tmp_root}/home"
mkdir -p "${HOME}"

run_smoke() {
  local script_name="$1"
  local output_file="${tmp_root}/${script_name%.py}.log"

  "${python_bin}" "${repo_root}/hmlib/cli/${script_name}" --smoke-test --game-id=smoke-ci >"${output_file}" 2>&1
  grep -q "Smoke test OK." "${output_file}"
}

run_smoke "hmtrack.py"
run_smoke "stitch.py"
