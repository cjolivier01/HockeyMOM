#!/bin/bash

#VIDEO="${HOME}"/Downloads/hockey.mov
VIDEO="${HOME}"/Downloads/stitched-${1}-with-sound.mp4
#VIDEO="${HOME}/Downloads/pdp_whole_game.mp4"

mkdir -p h-demo-${1}
rm -rf h-demo-${1}/*

PYTHONPATH="$(pwd)/build:$(pwd)/hockeymom" \
  python \
  ./src/demo.py \
  mot \
  --input-video "${VIDEO}" \
  --output-root=./h-demo-${1} \
  --load_model=./trained_models/fairmot/crowdhuman_dla34.pth \
  --conf_thres=0.5 \
  --min-box-area=80
