#!/bin/bash

PYTHONPATH="$(pwd)/build:$(pwd)/hockeymom:$(pwd)/models/mixsort:$(pwd)/external/MOTChallengeEvalKit" \
  python \
  $(pwd)/external/MOTChallengeEvalKit/MOT/MOTVisualization.py
