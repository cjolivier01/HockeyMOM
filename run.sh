#!/bin/bash

#VIDEO="${HOME}"/Downloads/hockey.mov
#VIDEO="${HOME}"/Downloads/pdp_hockey_game.mp4
#VIDEO="${HOME}/Downloads/pdp_whole_game.mp4"
#VIDEO="${HOME}/Videos/left.mp4,${HOME}/Videos/right.mp4"
#VIDEO="${HOME}/Videos/clips/ethan_goal.mp4"
#VIDEO="${HOME}/Videos/olivier2_stitched_hd.mp4"
#VIDEO="${HOME}/Videos/ethan_jets_first_goal.mp4"
#VIDEO="${HOME}/Videos/ethan_jets_first_goal.mp4"
VIDEO="${HOME}/Videos/TriValley10u1_9.23.2023.mp4"

mkdir -p h-demo-x
rm -rf h-demo-x/*

PYTHONPATH="$(pwd)/build:$(pwd)/hockeymom" \
  python \
  ./src/demo.py \
  mot \
  --input-video "${VIDEO}" \
  --output-root=./h-demo-x \
  --load_model=./trained_models/fairmot/crowdhuman_dla34.pth \
  --conf_thres=0.5 \
  --min-box-area=80
