#!/bin/bash

GAME_ID="sharks-bb3-2"

VIDEO_DIR="${HOME}/Videos/${GAME_ID}"

OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/models/mixsort \
	python src/stitch.py --video_dir="${VIDEO_DIR}" --game-id=${GAME_ID} --project_file=autooptimiser_out.pto ${OFFSETS}  $@
