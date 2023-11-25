#!/bin/bash
set -e
EXT=".avi"
BASE=$(basename $1 .avi)
F1=${BASE}-0.${EXT}
F2=${BASE}-1.${EXT}
F3=${BASE}-2.${EXT}
F4=${BASE}-3.${EXT}
ffmpeg -i $1 -ss 0 -t 1800 -c copy $F1 -ss 1800 -t 1800 -c copy $F2 -ss 3600 -t 1800 -c copy $F3 -ss 5400 -c copy $F4