#!/usr/bin/env bash

set -euo pipefail

repo_root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/rocm-runtime.XXXXXX")"
trap 'rm -rf "${tmp_root}"' EXIT

export HOME="${tmp_root}/home"
mkdir -p "${HOME}"

python "${repo_root}/tests/test_rocm_runtime_fallbacks.py"
