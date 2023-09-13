#!/bin/bash

#VIDEO="${HOME}"/Downloads/hockey.mov
#VIDEO="${HOME}"/Downloads/pdp_hockey_game.mp4
#VIDEO="${HOME}/Downloads/pdp_whole_game.mp4"
VIDEO="${HOME}/Videos/left-${1}.mp4,${HOME}/Videos/right-${1}.mp4"

rm -rf h-demo/*

PYTHONPATH="$(pwd)/build:$(pwd)/hockeymom" \
  python \
  ./src/demo.py \
  mot \
  --input-video "${VIDEO}" \
  --output-root=./h-demo \
  --load_model=./trained_models/fairmot/crowdhuman_dla34.pth \
  --conf_thres=0.5 \
  --min-box-area=80
