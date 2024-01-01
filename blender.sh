#!/bin/bash

#VIDEO_DIR="/home/colivier-local/Videos/sharksbb1-2"
#VIDEO_DIR="/home/colivier-local/Videos/onehockey-sharksbb2"
VIDEO_DIR="/home/colivier-local/Videos/onehockey-sharks10a"
#OFFSETS="--lfo=54.70108123169348 --rfo=0"
OFFSETS="--lfo=0 --rfo=0.627263892570015"
ROTATION="--rotation_angle=-25"

OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/models/mixsort \
	python src/hmlib/stitching/blender.py --video_dir="${VIDEO_DIR}" ${GAME_ID} --project_file=autooptimiser_out.pto ${ROTATION} ${OFFSETS}  $@
