#!/bin/bash
# STITCH_AUTO_ADJUST="--stitch-auto-adjust-exposure=1"
OMP_NUM_THREADS=24 \
	PYTHONPATH=$(pwd) \
	python -m hmlib.cli.stitch $STITCH_AUTO_ADJUST --game-id=${GAME_ID} ${OFFSETS}  $@
