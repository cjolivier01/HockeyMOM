#!/bin/bash
set -e
ffmpeg -i "$1" -vn -acodec copy output-audio.aac
ffmpeg -i "$2" -i output-audio.aac -map 0:v -map 1:a -c:v copy -shortest "$2"
