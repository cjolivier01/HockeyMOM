h

# generate project file from images, FOV=108 degrees
pto_gen -o my_project.pto -f 108 images/left.png images/right.png

# find control points
cpfind --linearmatch my_project.pto -o my_project.pto

# Optimize the project
autooptimiser -a -m -l -s -o my_project.pto my_project.pto

# Remap the images
nona -m TIFF_m -o my_project my_project.pto

# Blend the images to create the panorama
enblend -o panorama.tif my_project*.tif
