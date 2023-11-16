#!/bin/bash

#EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_sportsmot.py"
#EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_hockey.py"
EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_ch_ht.py"

#
# Models
#
#PRETRAINED_MODEL="models/mixsort/pretrained/yolox_x_sports_train.pth"
#PRETRAINED_MODEL="pretrained/mixsort/latest_ckpt.pth.tar"
#PRETRAINED_MODEL="pretrained/mixsort/last_epoch_ckpt.pth.tar"
#PRETRAINED_MODEL="models/mixsort/pretrained/yolox_x_ch.pth"
#PRETRAINED_MODEL="pretrained/mixsort/my_ch.pth.tar"
#PRETRAINED_MODEL="./latest_ckpt-e70.pth.tar"
PRETRAINED_MODEL="pretrained/mixsort/yolox_x_my_ch_to_hockey_tracking_dataset.pth.tar"

MIXFORMER_SCRIPT="mixformer_deit_hockey"

#
# Videos
#
#VIDEO="${HOME}/Videos/roseville/Sacramento.10.15.2023.mp4"
#VIDEO="/mnt/data/Videos/roseville/stitched_output-with-audio.avi"
#VIDEO="${HOME}/src/datasets/hockeyTrackingDataset/clips/CHI_VS_TOR/003.mp4"
#VIDEO="/mnt/data/Videos/roseville/clips/at10mins_small.mp4"
#VIDEO="${HOME}/Videos/roseville/clips/at10mins_small.avi"
#VIDEO="${HOME}/Videos/roseville/clips/at10mins_small.mp4"
#VIDEO="/mnt/data/Videos/SportsMOT/v_00HRwkvvjtQ_c001.mp4"
#VIDEO="${HOME}/src/datasets/hockeyTrackingDataset/clips/PIT_vs_WAS_2016/001.mp4"
VIDEO="${HOME}/Videos/sharksbb2/stitched_output-with-audio.avi"
#VIDEO="${HOME}/Videos/blackhawks/stitched_output-with-audio.avi"

#EXP_NAME="mixsort-run-$(uuidgen)"
EXP_NAME="mixsort-run-sharksbb2"

START_FRAME=1900

#HYPER_PARAMS="--conf=0.008 --track_thresh=0.08 --track_thresh_low=0.1"
HYPER_PARAMS="--conf=0.1 --track_thresh=0.3 --track_thresh_low=0.1"

echo "Experiment name: ${EXP_NAME}"

  # --track_thresh=0.3 \
  # --track_thresh_low=0.05 \

PYTHONPATH="$(pwd)/build:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hm_track_mixsort.py \
  -expn="${EXP_NAME}" \
  -f="${EXPERIMENT_FILE}" \
  -c="${PRETRAINED_MODEL}" \
  -b=1 \
  -d=1 \
  --start-frame=${START_FRAME} \
  --infer \
  ${HYPER_PARAMS} \
  --min-box-area=35 \
  --config=track \
  --script="${MIXFORMER_SCRIPT}" \
  --input_video="${VIDEO}"
