#!/bin/bash

EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_ch_cvat.py"
#
# Models
#
PRETRAINED_MODEL="pretrained/yolox/my_ch.pth.tar"

MIXFORMER_SCRIPT="mixformer_deit_hockey"

#
# Videos
#
VIDEO="${HOME}/Videos/tvbb/right.mp4"

EXP_NAME="$(basename $0 .sh)"

START_FRAME=6200
MAX_FRAMES=200
HYPER_PARAMS="--conf=0.001 --track_thresh=0.01 --track_thresh_low=0.005"

echo "Experiment name: ${EXP_NAME}"

OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
  PYTHONPATH="$(pwd)/build:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hm_track_mixsort.py \
  -expn="${EXP_NAME}" \
  -f="${EXPERIMENT_FILE}" \
  -c="${PRETRAINED_MODEL}" \
  -b=1 -d=1 \
  --infer \
  --cvat-output \
  --start-frame=${START_FRAME} \
  --max-frames=${MAX_FRAMES} \
  ${HYPER_PARAMS} \
  --save-frame-dir="$(pwd)/YOLOX_outputs/${EXP_NAME}/frames" \
  --min-box-area=35 \
  --rink="vallco" \
  --config=track \
  --cam-ignore-largest \
  --script="${MIXFORMER_SCRIPT}" \
  --input_video="${VIDEO}" $@
