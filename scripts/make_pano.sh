#!/bin/bash

rm -f panorama.tif nona*.tif

PROJECT_FILE="hm_project.pto"
#PROJECT_FILE="my_project.pto"

# generate project file from images, FOV=108 degrees
#pto_gen -p 0 -o my_project.pto -f 108 left.png right.png
#pto_gen -p 1 -o my_project.pto -f 108 left.png right.png

# find control points
#cpfind --linearmatch my_project.pto -o my_project.pto
#cpfind --linearmatch --celeste my_project.pto -o my_project.pto

# Optimize the project

# auto-level, auto-size
# autooptimiser -a -m -l -s -o autooptimiser_out.pto "${PROJECT_FILE}"

autooptimiser -s -m -n -o autooptimiser_out.pto "${PROJECT_FILE}"


# no auto size
# autooptimiser -a -m -l -o autooptimiser_out.pto "${PROJECT_FILE}"

# no optimize
# cp "${PROJECT_FILE}" autooptimiser_out.pto

#autooptimiser -a -m -o autooptimiser_out.pto "${PROJECT_FILE}"

# Remap the images
#nona -m TIFF_m -z NONE --bigtiff --clip-exposure -o nona my_project.pto
nona --bigtiff -m TIFF_m -z NONE --bigtiff -o nona autooptimiser_out.pto

#nona -m TIFF_m -o nona my_project.pto

#echo "Making mapping files..."
nona --bigtiff -m TIFF_m -z NONE -c -o mapping_ autooptimiser_out.pto

# Blend the images to create the panorama
#enblend -o panorama.tif nona*.tif
$HOME/src/multiblend/src/multiblend -o panorama.tif nona*.tif

#
# mapping files
#
#ffmpeg -i left-10sec.mp4 -i out0000_x.pgm -i out0000_y.pgm -lavfi remap -qscale 1 left-10sec-remapped.mp4
