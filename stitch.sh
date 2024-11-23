#!/bin/bash

OMP_NUM_THREADS=24 \
	PYTHONPATH=$(pwd):$(pwd)/xmodels/mixsort \
	python src/stitch.py --game-id=${GAME_ID} --stitch-auto-adjust-exposure ${OFFSETS}  $@
