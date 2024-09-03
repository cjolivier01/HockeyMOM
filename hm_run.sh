#!/bin/bash

#EXPERIMENT_FILE="$(pwd)/config/models/hm/hm_bytetrack.py"
EXPERIMENT_FILE="$(pwd)/config/models/hm/hm_end_to_end.py"
POSE_MODELS="--pose-config=./openmm/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py --pose-checkpoint=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

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
  PYTHONPATH="$(pwd)/build:$(pwd)/src" \
  ${WRAPPER_CMD} python src/hmtrack.py \
  --save-tracking-data \
  -expn="hm_run" \
  -f="${EXPERIMENT_FILE}" \
  -b=${BATCH_SIZE} ${POSE_MODELS} \
  --gpus=0,1,2,3 \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} ${TEST_SIZE_ARG} \
  ${VIDEO} $@
