# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install essential build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    yasm \
    pkg-config \
    wget \
    unzip \
    && apt-get clean

# Install NVIDIA driver dependencies and the NVIDIA codec SDK
RUN apt-get install -y \
    nvidia-driver-533 \
    nvidia-utils-533 \
    nvidia-cuda-dev \
    nvidia-cuda-toolkit \
    && apt-get clean

# Install FFmpeg build dependencies
RUN apt-get install -y \
    autoconf \
    automake \
    cmake \
    libtool \
    libass-dev \
    libfreetype6-dev \
    libsdl2-dev \
    libtheora-dev \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    pkg-config \
    texinfo \
    zlib1g-dev \
    && apt-get clean

# Download FFmpeg source code
WORKDIR /root
RUN https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && make install
RUN git clone https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg && git checkout release/5.0

# Configure FFmpeg with the desired codecs and options
WORKDIR /root/FFmpeg
RUN ./configure \
    --enable-cuda-nvcc \
    --enable-cuvid \
    --enable-nvenc \
    --enable-nonfree \
    --enable-libnpp \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    && make -j$(nproc) \
    && make install

# Cleanup
WORKDIR /root
RUN rm -rf FFmpeg

# Entry point
CMD ["/bin/bash"]

