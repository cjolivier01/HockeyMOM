#!/bin/bash

#VIDEO="${HOME}"/Downloads/hockey.mov
#VIDEO="${HOME}"/Videos/stitched-output-${1}-with-audio.mov
#VIDEO="${HOME}/Downloads/pdp_whole_game.mp4"

VIDEO="${HOME}/Videos/left-${1}.mp4,${HOME}/Videos/right-${1}.mp4"

mkdir -p h-demo-${1}
rm -rf h-demo-${1}/*

PYTHONPATH="$(pwd)/build:$(pwd)/hockeymom" \
  python \
  ./src/infer_full.py \
  mot \
  --input-video "${VIDEO}" \
  --output-root=./h-demo-${1} \
  --load_model=./trained_models/fairmot/crowdhuman_dla34.pth \
  --conf_thres=0.5 \
  --min-box-area=35
