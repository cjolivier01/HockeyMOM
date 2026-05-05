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
normalize_cuda_arch_for_rules_cuda() {
  local arch="$1"

  arch="${arch#sm_}"
  arch="${arch#compute_}"

  case "$arch" in
    30|32|35|37|50|52|53|60|61|62|70|72|75|80|86|87|89|90|100|101|120)
      printf 'sm_%s\n' "$arch"
      return 0
      ;;
  esac

  if [ "$arch" -gt 120 ] 2>/dev/null; then
    printf '%s\n' 'sm_120'
    return 0
  fi

  return 1
}

detect_cuda_archs_from_nvidia_smi() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1

  nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
    | awk '
      NF {
        gsub(/[^0-9]/, "", $1)
        if (length($1) == 1) {
          $1 = $1 "0"
        }
        print $1
      }
    ' \
    | while IFS= read -r arch; do
        normalize_cuda_arch_for_rules_cuda "$arch" || true
      done \
    | awk 'NF && !seen[$0]++' \
    | paste -sd ';' -
}

detect_cuda_archs_from_nvcc() {
  command -v nvcc >/dev/null 2>&1 || return 1

  nvcc --list-gpu-code 2>/dev/null \
    | while IFS= read -r arch; do
        normalize_cuda_arch_for_rules_cuda "$arch" || true
      done \
    | awk 'NF && !seen[$0]++' \
    | paste -sd ';' -
}

detect_cuda_bazel_archs() {
  local archs=""

  if [ -n "${CUDA_BAZEL_ARCHS:-}" ]; then
    printf '%s\n' "$CUDA_BAZEL_ARCHS"
    return 0
  fi

  archs="$(detect_cuda_archs_from_nvidia_smi)" || true
  if [ -n "$archs" ]; then
    printf '%s\n' "$archs"
    return 0
  fi

  archs="$(detect_cuda_archs_from_nvcc)" || true
  if [ -n "$archs" ]; then
    printf '%s\n' "$archs"
    return 0
  fi

  printf '%s\n' 'sm_75;sm_80;sm_86;sm_87;sm_89;sm_90;sm_100;sm_101;sm_120'
}
