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

if readelf -l -- "${binary_path}" | grep -q 'Requesting program interpreter'; then
  echo "hm-ui must be statically linked before it can be bundled in the portable Linux wheel" >&2
  exit 1
fi

if readelf -d -- "${binary_path}" 2>/dev/null | grep -q '(NEEDED)'; then
  echo "hm-ui has dynamic library dependencies and cannot be bundled in the portable Linux wheel" >&2
  exit 1
fi
