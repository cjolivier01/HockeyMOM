#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKSPACE_DIR=$(realpath "${SCRIPT_DIR}/..")

if [ ! -e "$(which bazelisk)" ]; then
  echo "Need to install bazelisk."
  "${WORKSPACE_DIR}/scripts/install_bazelisk.sh"
fi

if [ -z "${CPU}" ]; then
  # Prefer machine hardware name; uname -p can be "unknown" on Linux.
  ARCH="$(uname -m 2>/dev/null)"
  if [ -z "${ARCH}" ] || [ "${ARCH}" = "unknown" ]; then
    ARCH="$(uname -p 2>/dev/null)"
  fi

  case "${ARCH}" in
    x86_64|amd64)
      CPU="k8"
      ;;
    aarch64|arm64)
      CPU="aarch64"
      ;;
    *)
      CPU="${ARCH}"
      ;;
  esac
fi

# Get a clean "login" PATH from your login shell.
LOGIN_SHELL="${SHELL:-/bin/bash}"
LOGIN_PATH=$(
  env -i HOME="${HOME}" USER="${USER}" LOGNAME="${LOGNAME}" \
    "${LOGIN_SHELL}" -lc 'printf %s "$PATH"'
)

# Maybe add the current conda env.
if [ -n "${CONDA_PREFIX}" ]; then
  LOGIN_PATH="${CONDA_PREFIX}/bin:${LOGIN_PATH}"
fi

if [ -e "${HOME}/.profile" ]; then
  PATH="${LOGIN_PATH}"
  source "${HOME}/.profile"
  LOGIN_PATH="${PATH}"
fi

BAZEL_FLAGS="--action_env=PATH=${LOGIN_PATH}"

# Work around CUDA rsqrt/rsqrtf declaration conflicts on newer glibc headers.
GLIBC_VERSION="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [ -n "${GLIBC_VERSION}" ] && [ "$(printf '%s\n' "2.38" "${GLIBC_VERSION}" | sort -V | head -n1)" = "2.38" ]; then
  CUDA_GLIBC_WORKAROUND_HEADER="${WORKSPACE_DIR}/tools/cuda/glibc_math_workaround.h"
  BAZEL_FLAGS="${BAZEL_FLAGS} --define=glibc_math_rsqrt_conflict=1"
  BAZEL_FLAGS="${BAZEL_FLAGS} --@rules_cuda//cuda:copts=-U_GNU_SOURCE"
  BAZEL_FLAGS="${BAZEL_FLAGS} --@rules_cuda//cuda:copts=-D_DEFAULT_SOURCE"
  if [ -f "${CUDA_GLIBC_WORKAROUND_HEADER}" ]; then
    BAZEL_FLAGS="${BAZEL_FLAGS} --@rules_cuda//cuda:copts=-include"
    BAZEL_FLAGS="${BAZEL_FLAGS} --@rules_cuda//cuda:copts=${CUDA_GLIBC_WORKAROUND_HEADER}"
  else
    echo "Warning: missing CUDA glibc workaround header: ${CUDA_GLIBC_WORKAROUND_HEADER}" 1>&2
  fi
fi

BAZEL_FLAGS="${BAZEL_FLAGS} --cpu=${CPU}"

bazelisk "$@" ${BAZEL_FLAGS}
