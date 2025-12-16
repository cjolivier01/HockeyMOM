#!/bin/bash
set -e
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
if [ "$(uname -p)" == "aarch64" ]; then
  autooptimiser -x 0.75 -a -m -l -s -o autooptimiser_out.pto "${PROJECT_FILE}"
else
  autooptimiser -a -m -l -s -o autooptimiser_out.pto "${PROJECT_FILE}"
fi

echo "Making mapping files..."
nona --bigtiff -m TIFF_m -z NONE -c -o mapping_ autooptimiser_out.pto

# Blend the images to create the panorama
#enblend -o panorama.tif mapping_????.tif
#enblend --verbose=1 --save-masks=seam_file.png -o panorama.tif  mapping_0000.tif mapping_0001.tif
# enblend --verbose=1 --save-masks=seam_file.png -o panorama.tif mapping_????.tif
# enblend -v --save-masks=seam_file.png -o panorama.tif  mapping_????.tif
multiblend --save-seams=seam_file.png -o panorama.tif mapping_????.tif

#
# mapping files
#
#ffmpeg -i left-10sec.mp4 -i out0000_x.pgm -i out0000_y.pgm -lavfi remap -qscale 1 left-10sec-remapped.mp4
