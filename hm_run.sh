#!/bin/bash

EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_ch.py"

#
# Videos
#
#VIDEO="${HOME}/Videos/lbd2/stitched_output-with-audio.avi"
#VIDEO="${HOME}/Videos/tvbb2"
#VIDEO="${HOME}/Videos/sharksbb1-1"
VIDEO="${HOME}/Videos/sharks-bb1-2"

#GAME_ID="tvbb"
#GAME_ID="tvbb2"
#GAME_ID="sharksbb1-1"
GAME_ID="sharks-bb1-2"
#GAME_ID="sharks-bb3-2"

TEST_SIZE_ARG="--test-size=300x780"

BLEND_MODE="--blend-mode=laplacian"
#BLEND_MODE="--blend-mode=gpu-hard-seam"

START_FRAME=0
#START_FRAME=1900
#START_FRAME=2900
#START_FRAME=8730
#START_FRAME=30000
#START_FRAME=47000
#START_FRAME=16890
#START_FRAME=50000
#START_FRAME=46200

#TRACKER="--tracker=centertrack"
#TRACKER="--tracker=fair"
#BATCH_SIZE=32
#BATCH_SIZE=16
#BATCH_SIZE=16
BATCH_SIZE=1
#BATCH_SIZE=8

#STITCHING_PARAMS="--lfo=42.63559569682018 --rfo=0"

#CONFIDENCE="--conf_thres=0.25"

STITCHING_ARGS="--no-save-stitched"

echo "Experiment name: ${EXP_NAME}"

if [ ! -z "${VIDEO}" ]; then
  VIDEO="--input_video=${VIDEO}"
fi
set -x
OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONPATH="$(pwd)/build:$(pwd)/src/lib:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hmtrack.py \
  -expn="${GAME_ID}" \
  -f="${EXPERIMENT_FILE}" \
  ${PRETRAINED_MODEL} \
  -b=${BATCH_SIZE} \
  --gpus=0,1,2 \
  --infer \
  ${STITCHING_ARGS} \
  ${TRACKER} \
  ${CONFIDENCE} \
  ${BLEND_MODE} \
  --game-id="${GAME_ID}" \
  --start-frame=${START_FRAME} \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} ${TEST_SIZE_ARG} \
  --min-box-area=35 \
  --config=track \
  ${VIDEO} $@ tracking
