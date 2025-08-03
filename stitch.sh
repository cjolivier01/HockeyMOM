#!/bin/bash

OMP_NUM_THREADS=24 \
	PYTHONPATH=$(pwd) \
	python -m hmlib.cli.stitch --game-id=${GAME_ID} --stitch-auto-adjust-exposure=1 ${OFFSETS}  $@
