#!/bin/bash
OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/models/mixsort \
	python src/stitch.py --video_dir=/home/colivier-local/Videos/tvbb --lfo=15.39277049481131 --rfo=0 $@
