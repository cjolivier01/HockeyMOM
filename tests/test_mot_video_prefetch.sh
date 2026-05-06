#!/usr/bin/env bash

set -euo pipefail

repo_root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

source "${repo_root}/tests/python_runtime.sh"
python_bin="$(resolve_repo_python torch)"

"${python_bin}" -m pytest \
  "${repo_root}/tests/test_mot_video_prefetch.py" \
  -q \
  -c "${repo_root}/pyproject.toml"
