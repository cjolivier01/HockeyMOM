#!/bin/bash

#VIDEO_DIR="/home/colivier-local/Videos/sharksbb1-2"
#VIDEO_DIR="/home/colivier-local/Videos/onehockey-sharksbb2"
VIDEO_DIR="/home/colivier-local/Videos/onehockey-sharks10a"
GAME_ID="--game-id=onehockey-sharks10a"
OFFSETS="--lfo=0 --rfo=0.627263892570015"

OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/models/mixsort \
	python src/stitch.py --video_dir="${VIDEO_DIR}" ${GAME_ID} --project_file=autooptimiser_out.pto ${OFFSETS}  $@
