#!/bin/bash

VIDEO_DIR="${HOME}/Videos/sharks-bb1-2"
#VIDEO_DIR="${HOME}/Videos/jrmocks"
#VIDEO_DIR="${HOME}/Videos/tvbb"

OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/models/mixsort \
	python src/hmlib/stitching/blender.py --video_dir="${VIDEO_DIR}" ${GAME_ID} --project_file=autooptimiser_out.pto ${ROTATION} ${OFFSETS}  $@
