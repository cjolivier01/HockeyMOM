#!/bin/bash
#SCALE_VIDEO_CMD="-vf scale=4096:-1"
BITRATE=5000k
ffmpeg -re -i ~/Videos/sharks-bb2-2/tracking_output-with-audio.mp4 ${SCALE_VIDEO_CMD} -b:v ${BITRATE} -c:a aac -ar 44100 -ac 1 -f flv rtmp://localhost/live/stream
#ffmpeg -re -i ~/Videos/sharks-bb2-2/tracking_output-with-audio.mp4 -vf scale=4096:-1 -b:v 5000k -c:a aac -ar 44100 -ac 1 -f flv rtmp://localhost/live/stream
