#!/bin/bash

REPO_PYTHONPATH="$(pwd)"
if [ -d "$(pwd)/src" ]; then
  REPO_PYTHONPATH="${REPO_PYTHONPATH}:$(pwd)/src"
fi

OPENMM_PYTHONPATH="$(pwd)/openmm/mmcv:$(pwd)/openmm/mmengine:$(pwd)/openmm/mmeval:$(pwd)/openmm/mmdetection:$(pwd)/openmm/mmpose"

OMP_NUM_THREADS=24 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${OPENMM_PYTHONPATH}:${REPO_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m hmlib.cli.stitch --game-id=${GAME_ID} ${OFFSETS} $@
