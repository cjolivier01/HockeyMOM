#!/bin/bash

#EXPERIMENT_FILE="$(pwd)/config/models/hm/hm_bytetrack.py"
# EXPERIMENT_FILE="$(pwd)/config/models/hm/hm_end_to_end.py"

#START_FRAME=0

# BATCH_SIZE=1

#WRAPPER_CMD="nsys profile --show-outputs=true --wait=primary --trace=cuda,nvtx,cublas,cudnn,openacc --python-sampling=true --python-backtrace=cuda"
#WRAPPER_CMD="nsys profile"
#WRAPPER_CMD="echo"

#STITCHING_ARGS="--save-stitched"
# Legacy hmtrack flags for saving detections/tracks were removed, so only use the
# still-supported camera CSV option here to avoid CLI errors.
SAVE_DATA_ARGS="--save-camera-data"
# EXPOSURE="--stitch-auto-adjust-exposure=1"

echo "Experiment name: ${EXP_NAME}"

if [ ! -z "${VIDEO}" ]; then
  VIDEO="--input_video=${VIDEO}"
fi
set -x
OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONPATH="$(pwd):$(pwd)/src" \
  ${WRAPPER_CMD} python -m hmlib.cli.hmtrack \
  ${SAVE_DATA_ARGS} \
  ${EXPOSURE} \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} ${TEST_SIZE_ARG} \
  ${VIDEO} $@
