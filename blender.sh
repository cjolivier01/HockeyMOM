#!/bin/bash

#VIDEO_DIR="/home/colivier-local/Videos/sharks-bb3-2"
VIDEO_DIR="/home/colivier-local/Videos/tvbb2"
#ROTATION="--rotation_angle=-25"

OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/models/mixsort \
	python src/hmlib/stitching/blender.py --video_dir="${VIDEO_DIR}" ${GAME_ID} --project_file=autooptimiser_out.pto ${ROTATION} ${OFFSETS}  $@
