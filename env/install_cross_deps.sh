#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

if ! dpkg --print-foreign-architectures | grep -qx "arm64"; then
  echo "Adding arm64 as a foreign architecture..."
  sudo dpkg --add-architecture arm64
fi

if [[ -f /etc/apt/sources.list.d/ubuntu.sources ]]; then
  UBUNTU_CODENAME="$(. /etc/os-release; echo "${UBUNTU_CODENAME:-noble}")"

  # us.archive.ubuntu.com does not host arm64 indices. Keep host arches here.
  if grep -q '^Architectures:' /etc/apt/sources.list.d/ubuntu.sources; then
    sudo sed -E -i 's/^Architectures:.*/Architectures: amd64 i386/' /etc/apt/sources.list.d/ubuntu.sources
  else
    tmp_sources="$(mktemp)"
    awk '
      { print }
      /^Components:/ { print "Architectures: amd64 i386" }
    ' /etc/apt/sources.list.d/ubuntu.sources > "${tmp_sources}"
    sudo cp "${tmp_sources}" /etc/apt/sources.list.d/ubuntu.sources
    rm -f "${tmp_sources}"
  fi

  # Add arm64 sources from ports.ubuntu.com.
  sudo tee /etc/apt/sources.list.d/ubuntu-arm64-ports.sources >/dev/null <<EOF
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports/
Suites: ${UBUNTU_CODENAME} ${UBUNTU_CODENAME}-updates ${UBUNTU_CODENAME}-backports
Components: main restricted universe multiverse
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports/
Suites: ${UBUNTU_CODENAME}-security
Components: main restricted universe multiverse
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
fi

sudo apt-get update

# Toolchain for building aarch64 binaries from amd64 hosts.
sudo apt-get install -y --no-install-recommends \
  crossbuild-essential-arm64 \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu \
  libc6-dev-arm64-cross \
  libstdc++-13-dev-arm64-cross

# arm64 development libraries used by Bazel targets/deps in jetson builds.
sudo apt-get install -y --no-install-recommends \
  libgl1-mesa-dev:arm64 \
  libglu1-mesa-dev:arm64 \
  libglew-dev:arm64 \
  libglfw3-dev:arm64 \
  libglib2.0-dev:arm64 \
  libjson-glib-dev:arm64 \
  libsoup2.4-dev:arm64 \
  libgstreamer1.0-dev:arm64 \
  libgstreamer-plugins-base1.0-dev:arm64 \
  libgstrtspserver-1.0-dev:arm64 \
  libopencv-dev:arm64

# Optional NVIDIA CUDA cross-libs for true Jetson (aarch64) linking.
# If these packages are unavailable in configured apt repositories, we emit a
# warning with explicit next steps instead of failing silently.
if apt-cache show cuda-cudart-cross-aarch64-12-4 >/dev/null 2>&1; then
  sudo apt-get install -y --no-install-recommends \
    cuda-cudart-cross-aarch64-12-4 \
    cuda-libraries-dev-cross-aarch64-12-4 \
    cuda-nvcc-cross-aarch64-12-4
else
  echo "warning: CUDA aarch64 cross packages not found in apt sources." >&2
  echo "warning: install NVIDIA's CUDA apt repo (for example via cuda-keyring) if Jetson CUDA linking is required." >&2
fi
