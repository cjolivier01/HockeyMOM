#!/usr/bin/env bash

set -euo pipefail

binary_path="$1"
expected_arch="$2"
maximum_glibc="$3"

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

required_glibc="$(
  readelf --version-info -- "${binary_path}" \
    | sed -n 's/.*GLIBC_\([0-9][0-9.]*\).*/\1/p' \
    | sort -Vu \
    | tail -1
)"
if [[ -n "${required_glibc}" ]] && [[ "$(printf '%s\n' "${maximum_glibc}" "${required_glibc}" | sort -V | tail -1)" != "${maximum_glibc}" ]]; then
  echo "hm-ui requires GLIBC ${required_glibc}, newer than declared baseline ${maximum_glibc}" >&2
  exit 1
fi
