#!/bin/bash

#VIDEO="${HOME}"/Downloads/hockey.mov
#VIDEO="${HOME}"/Downloads/pdp_hockey_game.mp4
#VIDEO="${HOME}/Downloads/pdp_whole_game.mp4"
VIDEO="${HOME}/Videos/left-${1}.mp4,${HOME}/Videos/right-${1}.mp4"
VIDEO="${HOME}/Videos/vlc-record-2023-09-15-17h14m11s-stitched-output-2-with-audio.mov-.mp4"

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
