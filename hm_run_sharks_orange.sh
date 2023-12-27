#!/bin/bash

EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_ch.py"

#
# Models
#
PRETRAINED_MODEL="./pretrained/dla34/crowdhuman_dla34.pth"

MIXFORMER_SCRIPT="mixformer_deit_hockey"

#
# Videos
#
#VIDEO="${HOME}/Videos/sharksbb1-2/stiched_output-with-audio.avi"
VIDEO="${HOME}/Videos/sharksbb1-2/"

EXP_NAME="$(basename $0 .sh)"

GAME_ID="--game-id sharksbb1-2.1"

#TEST_SIZE_ARG="--test-size=200x520"
TEST_SIZE_ARG="--test-size=300x780"
#TEST_SIZE_ARG="--test-size=400x1040"

START_FRAME=0
#START_FRAME=1900
#START_FRAME=2900
#START_FRAME=8730
#START_FRAME=47000
#START_FRAME=10590

#TRACKER="hm"
TRACKER="fair"
#BATCH_SIZE=64
BATCH_SIZE=16

STITCHING_PARAMS="--lfo=0 --rfo=18.55423488076549"

echo "Experiment name: ${EXP_NAME}"

OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONPATH="$(pwd)/build:$(pwd)/src/lib:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hmtrack.py \
  -expn="${EXP_NAME}" \
  -f="${EXPERIMENT_FILE}" \
  -c="${PRETRAINED_MODEL}" \
  -b=${BATCH_SIZE} \
  --gpus=0,1,2 \
  --infer \
  --tracker=${TRACKER} \
  --start-frame=${START_FRAME} \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} ${GAME_ID} ${TEST_SIZE_ARG} \
  --min-box-area=35 \
  --rink="sharks_orange" \
  --config=track \
  --cam-ignore-largest \
  --script="${MIXFORMER_SCRIPT}" \
  --input_video="${VIDEO}" \
  $@ \
  tracking \


