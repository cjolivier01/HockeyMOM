#!/bin/bash

rm -f panorama.tif

# generate project file from images, FOV=108 degrees
pto_gen -o my_project.pto -f 108 left.png right.png

# find control points
cpfind --linearmatch --celeste my_project.pto -o my_project.pto

# Optimize the project
autooptimiser -a -m -l -s -o my_project.pto my_project.pto

# Remap the images
nona -m TIFF_m -o my_project my_project.pto

# Blend the images to create the panorama
#enblend -o panorama.tif my_project*.tif
/mnt/data/src/multiblend/src/multiblend -o panorama.tif my_project*.tif
