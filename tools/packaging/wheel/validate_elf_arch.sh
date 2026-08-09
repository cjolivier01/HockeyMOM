#!/usr/bin/env bash

set -euo pipefail

binary_path="$1"
expected_arch="$2"

machine="$(readelf -h -- "${binary_path}" | sed -n 's/^[[:space:]]*Machine:[[:space:]]*//p')"
case "${expected_arch}" in
  x86_64)
    expected_machine="Advanced Micro Devices X86-64"
    ;;
  aarch64)
    expected_machine="AArch64"
    ;;
  *)
    echo "Unsupported expected ELF architecture: ${expected_arch}" >&2
    exit 2
    ;;
esac

if [[ "${machine}" != "${expected_machine}" ]]; then
  echo "hm-ui architecture mismatch: expected ${expected_arch}, readelf reported ${machine:-unknown}" >&2
  exit 1
fi
