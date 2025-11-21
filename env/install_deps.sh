#!/bin/bash
set -e

sudo apt install -y \
  ccache \
  libpano13-dev \
  libglew-dev \
  libexiv2-dev \
  libfftw3-dev \
  libopenexr-dev \
  libx264-dev \
  libx265-dev \
  libvpx-dev \
  libfdk-aac-dev \
  libmp3lame-dev \
  libopus-dev \
  libilmbase-dev \
  liblcms2-dev \
  libsqlite3-dev \
  graphviz \
  graphviz-dev \
  gettext \
  libyaml-cpp-dev \
  liblapack-dev \
  libglfw3-dev \
  libtiff-dev \
  libpng-dev \
  fftw-dev \
  libprotobuf-dev \
  protobuf-compiler

# Enblend
sudo apt install -y \
  libgsl-dev \
  help2man \
  texlive-latex-base

WXW="$(apt search libwxgtk3 | grep dev | grep ^libwx | sed 's/\// /g' | awk '{print$1}')"
if [ ! -z "${WXW}" ]; then
  sudo apt install -y $WXW
fi
