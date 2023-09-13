#!/bin/bash
/usr/local/bin/ffmpeg -y -hwaccel cuda -i "$1" -c:a copy -c:v libx265 "$2"
