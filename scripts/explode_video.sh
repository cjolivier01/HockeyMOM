#!/bin/bash
ffmpeg -q:v 2 %06d.jpg -i $@
