#!/bin/bash
ffmpeg -q:v 2 -start_number 0 %06d.png -i $@
