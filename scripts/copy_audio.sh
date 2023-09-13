#!/bin/bash
set -e
INPUT_VIDEO_WITH_AUDIO="$1"
INPUT_VIDEO_NO_AUDIO="$2"
OUTPUT_VIDEO_WITH_AUDIO="$3"
#ffmpeg -i "$1" -vn -acodec copy output-audio.aac
#ffmpeg -i "$2" -i output-audio.aac -map 0:v -map 1:a -c:v copy -shortest "$3"
ffmpeg \
  -i "${INPUT_VIDEO_WITH_AUDIO}" \
  -i "${INPUT_VIDEO_NO_AUDIO}" \
  -c:v copy -c:a aac -strict experimental \
  -map 1:v:0 \
  -map 0:a:0 \
  -shortest \
  "${OUTPUT_VIDEO_WITH_AUDIO}"

