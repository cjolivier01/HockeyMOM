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
#PRETRAINED_MODEL="./latest_ckpt-e076.pth.tar"
PRETRAINED_MODEL="./trained_models/fairmot/crowdhuman_dla34.pth"

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
#VIDEO="${HOME}/Videos/lbd2/stitched_output-with-audio.avi"
#VIDEO="${HOME}/Videos/blackhawks/stitched_output-with-audio.avi"
#VIDEO="${HOME}/Videos/tvbb/stitched_output-with-audio.avi"
#VIDEO="${HOME}/Videos/tvbb/right.mp4"
VIDEO="${HOME}/Videos/tvbb"

EXP_NAME="mixsort-run-fairmot"

#START_FRAME=0
#START_FRAME=1900
#START_FRAME=2900
#START_FRAME=6200
START_FRAME=8000
#START_FRAME=10590

#TRACKER="hm"
TRACKER="fair"

#YPER_PARAMS="--conf=0.1 --track_thresh=0.3 --track_thresh_low=0.1"
#HYPER_PARAMS="--conf=0.01 --track_thresh=0.01 --track_thresh_low=0.005"
#HYPER_PARAMS="--conf=0.001 --track_thresh=0.005 --track_thresh_low=0.0001"

#STITCHING_PARAMS="--lfo=15.392 --rfo=0"

echo "Experiment name: ${EXP_NAME}"

OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONPATH="$(pwd)/build:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hm_track_mixsort.py \
  -expn="${EXP_NAME}" \
  -f="${EXPERIMENT_FILE}" \
  -c="${PRETRAINED_MODEL}" \
  -b=4 -d=1 \
  --infer \
  --tracker=${TRACKER} \
  --start-frame=${START_FRAME} \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} \
  --min-box-area=35 \
  --rink="vallco" \
  --config=track \
  --cam-ignore-largest \
  --script="${MIXFORMER_SCRIPT}" \
  --input_video="${VIDEO}" $@
