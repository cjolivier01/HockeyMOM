#!/usr/bin/env bash
set -euo pipefail

# Back-compat shim.
#
# The webapp + GCP import tooling moved to the HockeyMOMWeb repo:
#   ../HockeyMOMWeb/gcp_import_webapp.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HMWEB_DIR="${SCRIPT_DIR}/../HockeyMOMWeb"

if [[ -x "${HMWEB_DIR}/gcp_import_webapp.sh" ]]; then
  exec "${HMWEB_DIR}/gcp_import_webapp.sh" "$@"
fi

echo "ERROR: HockeyMOMWeb not found." >&2
echo "Clone it next to this repo at: ${HMWEB_DIR}" >&2
echo "Then run: ../HockeyMOMWeb/gcp_import_webapp.sh" >&2
exit 2

