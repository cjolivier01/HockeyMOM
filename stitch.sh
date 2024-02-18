#!/bin/bash

#GAME_ID="sharks-bb2-2"
#GAME_ID="sharksbb1-1"
#GAME_ID="tvbb"
GAME_ID="sundevils"

VIDEO_DIR="${HOME}/Videos/${GAME_ID}"

OMP_NUM_THREADS=24 \
	PYTHONPATH=$(pwd):$(pwd)/models/mixsort \
	python src/stitch.py --video_dir="${VIDEO_DIR}" --game-id=${GAME_ID} --project_file=autooptimiser_out.pto ${OFFSETS}  $@
