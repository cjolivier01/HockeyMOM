#!/bin/bash

VIDEO_DIR="/home/colivier-local/Videos/sharksbb1-2"

OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/models/mixsort \
	python src/stitch.py --video_dir="${VIDEO_DIR}" --project_file=autooptimizer_out.pto $@
