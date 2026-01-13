#!/bin/bash
gst-launch-1.0 -e \
	nvurisrcbin uri="file:///$(realpath $1)" name=src \
  src. ! queue ! mux.sink_0 \
  nvstreammux name=mux batch-size=1 width=1920 height=1080 batched-push-timeout=40000 \
  ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" \
  ! nveglglessink sync=true

