#!/usr/bin/env bash

# Runs the built executable for a given target.
# Example:
#   ./run.sh //tools/python/pipenv:pipenv install funcy~=1.17

set -euo pipefail

target="$1"
shift

exec bazel run "${target}" -- "$@"
