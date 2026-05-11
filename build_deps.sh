#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/external/hugin"
bazelisk run //:install_tree -- --prefix="${CONDA_PREFIX:?CONDA_PREFIX must be set}"
