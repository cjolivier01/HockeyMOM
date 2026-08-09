#!/usr/bin/env bash

set -euo pipefail

test_file="$1"
shift
required_modules=("$@")

repo_root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

source "${repo_root}/tests/python_runtime.sh"
python_bin="$(resolve_repo_python "${required_modules[@]}")"

"${python_bin}" -m pytest \
  "${repo_root}/${test_file}" \
  -q \
  -c "${repo_root}/pyproject.toml"
