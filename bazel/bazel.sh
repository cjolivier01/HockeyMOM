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

case "${1:-}" in
  build|coverage|run|test)
    CUDA_BAZEL_ARCHS="$(detect_cuda_bazel_archs)"
    export CUDA_BAZEL_ARCHS
    BAZEL_FLAGS="${BAZEL_FLAGS} --@rules_cuda//cuda:archs=${CUDA_BAZEL_ARCHS}"
    ;;
esac

args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--" ]]; then
    before=("${args[@]:0:i}")
    after=("${args[@]:i+1}")
    exec bazelisk "${before[@]}" ${BAZEL_FLAGS} -- "${after[@]}"
  fi
done

exec bazelisk "${args[@]}" ${BAZEL_FLAGS}
