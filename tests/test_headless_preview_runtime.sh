#!/usr/bin/env bash

set -euo pipefail

repo_root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

source "${repo_root}/tests/python_runtime.sh"
python_bin="$(resolve_repo_python cv2 torch)"

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/headless-preview.XXXXXX")"
trap 'rm -rf "${tmp_root}"' EXIT

export HOME="${tmp_root}/home"
mkdir -p "${HOME}"

"${python_bin}" "${repo_root}/tests/test_headless_preview_runtime.py"
