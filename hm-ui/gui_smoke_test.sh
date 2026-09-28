#!/usr/bin/env bash

set -euo pipefail

hm_ui="$1"
spec="$2"

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "Skipping hm-ui GUI smoke test: xvfb-run is not installed" >&2
  exit 0
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "Skipping hm-ui GUI smoke test: timeout is not installed" >&2
  exit 0
fi

state_dir="$(mktemp -d "${TEST_TMPDIR:-/tmp}/hm-ui-gui-smoke.XXXXXX")"
trap 'rm -rf "${state_dir}"' EXIT

set +e
LIBGL_ALWAYS_SOFTWARE=1 WINIT_UNIX_BACKEND=x11 timeout 5s xvfb-run -a \
  "${hm_ui}" \
  --spec "${spec}" \
  --state "${state_dir}/state.json" \
  --title "HM UI GUI Smoke Test"
status=$?
set -e

if [[ "${status}" -ne 124 ]]; then
  echo "hm-ui exited during the GUI smoke interval with status ${status}" >&2
  exit 1
fi

# Surviving the interval is not enough: a spec that fails to parse leaves the UI
# running with no windows and exits 124 too. State is only written after a
# successful parse, so require the spec's controls to appear in it.
for control in Seam_Blend_Mode Seam_Feather_Percent Shadow_Lift_Black_Point; do
  if ! grep -q "${control}" "${state_dir}/state.json" 2>/dev/null; then
    echo "hm-ui did not render ${control} from the smoke spec" >&2
    cat "${state_dir}/state.json" >&2 2>/dev/null || echo "(no state file)" >&2
    exit 1
  fi
done
