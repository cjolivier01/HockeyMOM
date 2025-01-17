#!/bin/bash
 ffmpeg function: ffmpeg -hwaccel cuda -i "$1" -vf "unsharp=3:3:1.5:3:3:0.5" -c:v hevc_nvenc -preset lossless -acodec copy $2
 
