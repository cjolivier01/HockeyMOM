#!/usr/bin/env bash

set -euo pipefail

repo_root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/bazel-setup-cuda-archs.XXXXXX")"
trap 'rm -rf "${tmp_root}"' EXIT

fake_bin="${tmp_root}/bin"
fake_conda="${tmp_root}/conda"
fake_home="${tmp_root}/home"
mkdir -p "${fake_bin}" "${fake_conda}/bin" "${fake_home}"

cat >"${fake_bin}/bazelisk" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"${fake_bin}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"--query-gpu=compute_cap"* ]]; then
  printf '%b' "${NVIDIA_SMI_COMPUTE_CAPS:-7.0\\n}"
  exit 0
fi
echo "unexpected nvidia-smi args: $*" >&2
exit 1
EOF

cat >"${fake_bin}/nvcc" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"--list-gpu-code"* ]]; then
  printf '%s\n' ${NVCC_GPU_CODES:-sm_75 sm_80 sm_120}
  exit 0
fi
echo "unexpected nvcc args: $*" >&2
exit 1
EOF

cat >"${fake_conda}/bin/python" <<EOF
#!/usr/bin/env bash
if [[ "\$*" == *"import torch"* ]]; then
  printf '%s\n' "cpu"
  exit 0
fi
if [[ "\$*" == *"import sys; print(sys.prefix)"* ]]; then
  printf '%s\n' "${fake_conda}"
  exit 0
fi
exit 0
EOF

chmod +x \
  "${fake_bin}/bazelisk" \
  "${fake_bin}/nvidia-smi" \
  "${fake_bin}/nvcc" \
  "${fake_conda}/bin/python"

export CONDA_PREFIX="${fake_conda}"
export HOME="${fake_home}"
export LOGNAME="${LOGNAME:-hm-test}"
export PATH="${fake_bin}:${fake_conda}/bin:/usr/bin:/bin"
export SHELL="/bin/bash"
export USER="${USER:-hm-test}"

source "${repo_root}/.bazel_setup.sh" >/dev/null 2>&1

export PATH="${fake_bin}:${fake_conda}/bin:/usr/bin:/bin"

assert_eq() {
  local expected="$1"
  local actual="$2"
  local label="$3"

  if [[ "${actual}" != "${expected}" ]]; then
    echo "${label}: expected '${expected}', got '${actual}'" >&2
    return 1
  fi
}

actual="$(detect_cuda_archs_from_nvidia_smi)"
assert_eq "sm_70" "${actual}" "nvidia-smi arch detection"

actual="$(detect_cuda_archs_from_nvcc)"
assert_eq "sm_75;sm_80;sm_120" "${actual}" "nvcc arch detection"

warn_file="${tmp_root}/unsupported-all.warn"
actual="$(
  NVIDIA_SMI_COMPUTE_CAPS=$'7.0\n' \
  NVCC_GPU_CODES="sm_75 sm_80 sm_120" \
  detect_cuda_bazel_archs 2>"${warn_file}"
)"
assert_eq "sm_75;sm_80;sm_120" "${actual}" "fallback to compiler-supported archs"
grep -Fq "supports none of detected GPU arch(s): sm_70" "${warn_file}"

warn_file="${tmp_root}/unsupported-partial.warn"
actual="$(
  NVIDIA_SMI_COMPUTE_CAPS=$'7.0\n8.0\n' \
  NVCC_GPU_CODES="sm_75 sm_80" \
  detect_cuda_bazel_archs 2>"${warn_file}"
)"
assert_eq "sm_80" "${actual}" "filter unsupported detected archs"
grep -Fq "does not support detected GPU arch(s): sm_70" "${warn_file}"

actual="$(CUDA_BAZEL_ARCHS="sm_70" detect_cuda_bazel_archs)"
assert_eq "sm_70" "${actual}" "explicit CUDA_BAZEL_ARCHS override"
