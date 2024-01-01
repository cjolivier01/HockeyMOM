#!/bin/bash

#VIDEO_DIR="/home/colivier-local/Videos/sharksbb1-2"
VIDEO_DIR="/home/colivier-local/Videos/onehockey-sharksbb2"
OFFSETS="--lfo=54.70108123169348 --rfo=0"

OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/models/mixsort \
	python src/hmlib/stitching/blender.py --video_dir="${VIDEO_DIR}" ${GAME_ID} --project_file=autooptimiser_out.pto ${OFFSETS}  $@
