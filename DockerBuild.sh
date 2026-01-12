#!/bin/bash
set -euo pipefail

print_help() {
  cat <<'EOF'
DockerBuild.sh: build the HockeyMOM CUDA Docker image

Usage:
  ./DockerBuild.sh [options...]
  ./DockerBuild.sh --help

Options are passed to: scripts/hm_cuda_container.py build

Common options:
  --tag TAG                     Docker image tag (default: env/tag if present)
  --network default|host|none    Build network mode
  --cuda-base IMAGE             Base image (default: nvidia/cuda:12.4.1-devel-ubuntu22.04)
  --torch-index-url URL
  --torch-version VERSION
  --torchvision-version VERSION
  --torchaudio-version VERSION

Examples:
  ./DockerBuild.sh
  ./DockerBuild.sh --tag hm
  ./DockerBuild.sh --network host
EOF
}

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

PYTHONPATH="$(pwd)" exec python scripts/hm_cuda_container.py build "$@"
