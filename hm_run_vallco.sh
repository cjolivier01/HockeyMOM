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
#VIDEO="${HOME}/Videos/lbd2/stitched_output-with-audio.avi"
#VIDEO="${HOME}/Videos/stockton2/stitched_output-with-audio.avi"
VIDEO="${HOME}/Videos/stockton2/"

EXP_NAME="$(basename $0 .sh)"

GAME_ID="--game-id stockton2"

TEST_SIZE_ARG="--test-size=200x520"

#START_FRAME=0
#START_FRAME=1900
#START_FRAME=2900
#START_FRAME=8730
START_FRAME=47000
#START_FRAME=10590

#TRACKER="hm"
TRACKER="fair"

#STITCHING_PARAMS="--lfo=0 --rfo=49.79510285632735"

echo "Experiment name: ${EXP_NAME}"

OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONPATH="$(pwd)/build:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hm_track_mixsort.py \
  -expn="${EXP_NAME}" \
  -f="${EXPERIMENT_FILE}" \
  -c="${PRETRAINED_MODEL}" \
  -b=64 -d=1 \
  --infer \
  --tracker=${TRACKER} \
  --start-frame=${START_FRAME} \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} ${GAME_ID} ${TEST_SIZE_ARG} \
  --min-box-area=35 \
  --rink="vallco" \
  --config=track \
  --cam-ignore-largest \
  --script="${MIXFORMER_SCRIPT}" \
  --input_video="${VIDEO}" $@

