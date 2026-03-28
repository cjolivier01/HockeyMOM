#
# Bazel-related utility code to be sourced by build scripts
#
if [ ! -e "$(which bazelisk)" ]; then
  echo "Need to install bazelisk."
  ./scripts/install_bazelisk.sh
fi

resolve_conda_prefix() {
  local prefix derived_prefix python_bin

  prefix="${CONDA_PREFIX:-}"
  if [ -n "${prefix}" ] && [ -x "${prefix}/bin/python" ]; then
    printf '%s' "${prefix}"
    return 0
  fi

  python_bin="$(command -v python 2>/dev/null || true)"
  if [ -n "${python_bin}" ]; then
    derived_prefix="$("${python_bin}" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
    if [ -n "${derived_prefix}" ] && [ -x "${derived_prefix}/bin/python" ]; then
      if [ -n "${prefix}" ]; then
        printf '%s\n' \
          "CONDA_PREFIX=${prefix} is invalid; using ${derived_prefix} derived from ${python_bin}." >&2
      else
        printf '%s\n' \
          "CONDA_PREFIX is not set; using ${derived_prefix} derived from ${python_bin}." >&2
      fi
      printf '%s' "${derived_prefix}"
      return 0
    fi
  fi

  if [ -n "${prefix}" ]; then
    printf '%s\n' \
      "Warning: CONDA_PREFIX=${prefix} does not contain bin/python, and no replacement prefix was derived from python on PATH." >&2
  fi

  return 1
}

resolve_torch_backend() {
  local python_bin

  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    python_bin="${CONDA_PREFIX}/bin/python"
  else
    python_bin="$(command -v python 2>/dev/null || true)"
  fi

  if [ -z "${python_bin}" ]; then
    return 1
  fi

  "${python_bin}" -c '
import os
import sys

cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", cwd)]

import torch

if getattr(torch.version, "hip", None):
    print("rocm")
elif getattr(torch.version, "cuda", None):
    print("cuda")
else:
    print("cpu")
' 2>/dev/null || true
}

RESOLVED_CONDA_PREFIX="$(resolve_conda_prefix || true)"
if [ -n "${RESOLVED_CONDA_PREFIX}" ]; then
  export CONDA_PREFIX="${RESOLVED_CONDA_PREFIX}"
fi

CPU="$(uname -m)"
if [ "$CPU" == "x86_64" ]; then
  CPU="k8"
fi

# Get a clean "login" PATH from your login shell
LOGIN_SHELL="${SHELL:-/bin/bash}"
LOGIN_PATH=$(
  env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" \
    "$LOGIN_SHELL" -lc 'printf %s "$PATH"'
)

# Maybe add the current conda env
if [ ! -z "${CONDA_PREFIX}" ]; then
  LOGIN_PATH="${CONDA_PREFIX}/bin:${LOGIN_PATH}"
fi

if [ -e "${HOME}/.profile" ]; then
  PATH="${LOGIN_PATH}"
  source "${HOME}/.profile"
  LOGIN_PATH="${PATH}"
fi

# echo "LOGIN_PATH=${LOGIN_PATH}"

BAZEL_FLAGS="--action_env=PATH=${LOGIN_PATH}"

TORCH_BACKEND="$(resolve_torch_backend | tr -d '\n' || true)"
if [ -n "${TORCH_BACKEND}" ]; then
  BAZEL_FLAGS="${BAZEL_FLAGS} --define=torch_backend=${TORCH_BACKEND}"
fi
