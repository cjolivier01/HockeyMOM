#!/bin/bash

EXPERIMENT_FILE="models/mixsort/exps/example/mot/yolox_x_sportsmot.py"
PRETRAINED_MODEL="models/mixsort/pretrained/yolox_x_sports_train.pth"
VIDEO="${HOME}/Videos/roseville/Sacramento.10.15.2023.mp4"

PYTHONPATH="$(pwd)/build:$(pwd)/models/mixsort:$(pwd)/models/mixsort/MixViT:$(pwd)/src" \
  python src/hm_track_mixsort.py \
  -expn=my_experiment \
  -f="${EXPERIMENT_FILE}" \
  -c="${PRETRAINED_MODEL}" \
  -b=1 -d=1 \
  --infer \
  --min-box-area=55 \
  --config=track \
  --input_video="${VIDEO}"

