#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source "${SCRIPT_DIR}"/../.bazel_setup.sh

BAZEL_FLAGS="${BAZEL_FLAGS} --cpu=${CPU}"

bazelisk $@ ${BAZEL_FLAGS}
