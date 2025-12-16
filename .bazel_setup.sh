#
# Bazel-related utility code to be sourced by build scripts
#
if [ ! -e "$(which bazelisk)" ]; then
  echo "Need to install bazelisk."
  ./scripts/install_bazelisk.sh
fi

CPU="$(uname -p)"
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
