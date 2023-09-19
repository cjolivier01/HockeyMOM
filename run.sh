#!/bin/bash

#VIDEO="${HOME}"/Downloads/hockey.mov
#VIDEO="${HOME}"/Downloads/pdp_hockey_game.mp4
#VIDEO="${HOME}/Downloads/pdp_whole_game.mp4"
#VIDEO="${HOME}/Videos/left-${1}.mp4,${HOME}/Videos/right-${1}.mp4"
#VIDEO="${HOME}/Videos/clips/vlc-record-2023-09-15-18h18m32s-stitched-output-2.mov-.mp4"
#VIDEO="${HOME}/Videos/clips/ethan_goal.mp4"
VIDEO="${HOME}/Videos/security_cam_video.mp4"

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
