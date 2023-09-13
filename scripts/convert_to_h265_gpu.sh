#!/bin/bash

#  -crf 40
# -b:v 2700k

/usr/local/bin/ffmpeg \
  -y \
  -hwaccel cuda  \
  -i "$1" \
  -c:a copy \
  -c:v libx265 \
  -crf 40 \
  -b:a 192k \
  -preset medium \
  -tune fastdecode \
  "$2"
