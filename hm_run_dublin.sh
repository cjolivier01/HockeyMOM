#!/bin/bash

#EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_sportsmot.py"
#EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_hockey.py"
#EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_ch_ht.py"
EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_ch.py"

#
# Models
#
#PRETRAINED_MODEL="pretrained/yolox/yolox_x_sports_train.pth"
#PRETRAINED_MODEL="pretrained/yolox/yolox_x_ch.pth"
#PRETRAINED_MODEL="pretrained/yolox/my_ch.pth.tar"
#PRETRAINED_MODEL="pretrained/yolox/yolox_x_my_ch_to_hockey_tracking_dataset.pth.tar"
#PRETRAINED_MODEL="./latest_ckpt-e080.pth.tar"
#PRETRAINED_MODEL="./latest_ckpt-e076.pth.tar"
#PRETRAINED_MODEL="./pretrained/dla34/crowdhuman_dla34.pth"
PRETRAINED_MODEL="./pretrained/centertrack/crowdhuman.pth"


MIXFORMER_SCRIPT="mixformer_deit_hockey"
#MIXFORMER_SCRIPT="mixformer_deit_ch_ht"

#
# Videos
#
#VIDEO="${HOME}/Videos/roseville/Sacramento.10.15.2023.mp4"
#VIDEO="/mnt/data/Videos/roseville/stitched_output-with-audio.avi"
#VIDEO="${HOME}/src/datasets/hockeyTrackingDataset/clips/CHI_VS_TOR/003.mp4"i
#VIDEO="/mnt/data/Videos/roseville/clips/at10mins_small.mp4"
#VIDEO="${HOME}/Videos/roseville/clips/at10mins_small.avi"
#VIDEO="${HOME}/Videos/roseville/clips/at10mins_small.mp4"
#VIDEO="/mnt/data/Videos/SportsMOT/v_00HRwkvvjtQ_c001.mp4"
#VIDEO="${HOME}/src/datasets/hockeyTrackingDataset/clips/PIT_vs_WAS_2016/001.mp4"
VIDEO="${HOME}/Videos/tvbb2/stitched_output-with-audio.avi"
#VIDEO="${HOME}/Videos/tvbb2"

GAME_ID="--game-id tvbb2"

#TEST_SIZE_ARG="--test-size=400x1040"i
TEST_SIZE_ARG="--test-size=300x780"
#TEST_SIZE_ARG="--test-size=200x520"

EXP_NAME="$(basename $0 .sh)"

START_FRAME=0
#START_FRAME=14400
#START_FRAME=125850

#TRACKER="hm"
#TRACKER="fair"
TRACKER="centertrack --num_classes=1"

#STITCHING_PARAMS="--lfo=78.94010751153608 --rfo=0"

echo "Experiment name: ${EXP_NAME}"

OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONPATH="$(pwd)/build:$(pwd)/src/lib:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hmtrack.py \
  -expn="${EXP_NAME}" \
  -f="${EXPERIMENT_FILE}" \
  -c="${PRETRAINED_MODEL}" \
  -b=16 -d=1 \
  --infer \
  --tracker=${TRACKER} \
  --start-frame=${START_FRAME} \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} ${GAME_ID} ${TEST_SIZE_ARG} \
  --min-box-area=35 \
  --rink="dublin" \
  --config=track \
  --cam-ignore-largest \
  --script="${MIXFORMER_SCRIPT}" \
  --input_video="${VIDEO}" $@ \
  tracking


