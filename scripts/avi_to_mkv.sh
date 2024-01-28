#!/bin/bash
MKV_NAME="$(basename $1 .avi).mkv"
ffmpeg -i "$1" -c:v copy -c:a copy "${MKV_NAME}" 
