#!/bin/bash

EXPERIMENT_FILE="$(pwd)/../openmm/mmtracking/configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py"

TEST_SIZE_ARG="--test-size=300x780"


#START_FRAME=0

BATCH_SIZE=1


#WRAPPER_CMD="nsys profile --show-outputs=true --wait=primary --trace=cuda,nvtx,cublas,cudnn,openacc --python-sampling=true --python-backtrace=cuda"
#WRAPPER_CMD="nsys profile"
#WRAPPER_CMD="echo"

#STITCHING_ARGS="--save-stitched"

echo "Experiment name: ${EXP_NAME}"

if [ ! -z "${VIDEO}" ]; then
  VIDEO="--input_video=${VIDEO}"
fi
set -x
OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONPATH="$(pwd)/build:$(pwd)/src/lib:$(pwd)/xmodels/mixsort:$(pwd)/xmodels/mixsort/MixViT:$(pwd)/src" \
  ${WRAPPER_CMD} python src/hmtrack.py \
  -expn="hm_mmtrack" \
  -f="${EXPERIMENT_FILE}" \
  -b=${BATCH_SIZE} \
  --gpus=0,1,2,3 \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} ${TEST_SIZE_ARG} \
  --min-box-area=35 \
  --config=track \
  --tracker=mmtrack \
  --test-size=300x780 \
  ${VIDEO} $@ tracking
