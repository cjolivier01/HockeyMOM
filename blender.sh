#!/bin/bash
OMP_NUM_THREADS=16 \
	PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/xmodels/mixsort \
	python src/hmlib/stitching/blender.py $@
