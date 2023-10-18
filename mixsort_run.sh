#!/bin/bash

EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_sportsmot.py"
PRETRAINED_MODEL="models/mixsort/pretrained/yolox_x_sports_train.pth"
#PRETRAINED_MODEL="pretrained/mixsort/latest_ckpt.pth.tar"
VIDEO="${HOME}/Videos/roseville/Sacramento.10.15.2023.mp4"

#EXP_NAME="mixsort-run-$(uuidgen)"
EXP_NAME="mixsort-run"

echo "Experiment name: ${EXP_NAME}"

PYTHONPATH="$(pwd)/build:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hm_track_mixsort.py \
  -expn="${EXP_NAME}" \
  -f="${EXPERIMENT_FILE}" \
  -c="${PRETRAINED_MODEL}" \
  -b=1 -d=1 \
  --infer \
  --min-box-area=55 \
  --config=track \
  --input_video="${VIDEO}"

