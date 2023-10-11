#!/bin/bash

rm -f panorama.tif nona*.tif

# generate project file from images, FOV=108 degrees
pto_gen -o my_project.pto -f 108 left.png right.png

# find control points
cpfind --linearmatch --celeste my_project.pto -o cpfind_out.pto

# Optimize the project
autooptimiser -a -m -l -s -o autooptimiser_out.pto cpfind_out.pto

# Remap the images
#nona -m TIFF_m -z NONE --bigtiff --clip-exposure -o nona my_project.pto
#nona -m TIFF_m -z NONE --bigtiff -o nona my_project.pto
nona -m TIFF_m -o nona cpfind_out.pto

# Blend the images to create the panorama
#enblend -o panorama.tif my_project*.tif
/mnt/data/src/multiblend/src/multiblend -o panorama.tif nona*.tif
