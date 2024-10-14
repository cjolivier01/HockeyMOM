FROM nvidia/cuda:12.3.2-base-ubuntu22.04

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

# Install Vigra/Hugin dependencies
RUN apt-get install -y \
  ccache \
  libpano13-dev \
  libglew-dev \
  libexiv2-dev \
  libopenexr-dev \
  libilmbase-dev \
  liblcms2-dev \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libsqlite3-dev \
  libavformat-dev \
  liblapack-dev \
  gettext \
  libyaml-cpp-dev \
  liblapack-dev \
  libglfw3-dev \
  libfftw3-dev \
  libglew-dev \
  libgl-dev \
  libjpeg-turbo8-dev \
  libtiff-dev \
  libopus-dev \
  libboost-all-dev \
  libprotobuf-dev \
  libpng-dev \
  libopenblas-dev \
  libomp-dev


# Install NVIDIA driver dependencies and the NVIDIA codec SDK
RUN apt-get install -y \
  cuda-12-3 \
  cudnn9-cuda-12 \
  cuda-toolkit-12-3 \
  libffmpeg-nvenc-dev \
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
  libx264-dev \
  libx265-dev \
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

# Some more dependencies
RUN apt-get install -y \
  vim \
  libgnutls28-dev \
  && apt-get clean

# Download FFmpeg source code
WORKDIR /root
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && git checkout n11.1.5.3 && make install
RUN git clone https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg && git checkout n5.1.4

# Configure FFmpeg with the desired codecs and options
WORKDIR /root/FFmpeg
RUN NVCCFLAGS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_89,code=sm_89" \
  ./configure \
  --prefix="/usr/local" \
  --extra-cflags='-I/usr/local/cuda/include' \
  --extra-ldflags='-L/usr/local/cuda/lib64' \
  --disable-doc \
  --enable-decoder=aac \
  --enable-decoder=h264 \
  --enable-decoder=h264_cuvid \
  --enable-decoder=hevc_cuvid \
  --enable-decoder=av1_cuvid \
  --enable-decoder=rawvideo \
  --enable-indev=lavfi \
  --enable-encoder=libx264 \
  --enable-encoder=h264_nvenc \
  --enable-encoder=hevc_nvenc \
  --enable-encoder=av1_nvenc \
  --enable-demuxer=mov \
  --enable-muxer=mp4 \
  --enable-filter=scale \
  --enable-filter=testsrc2 \
  --enable-protocol=file \
  --enable-protocol=https \
  --enable-gnutls \
  --enable-shared \
  --enable-gpl \
  --enable-nonfree \
  --enable-cuda-nvcc \
  --enable-libx264 \
  --enable-libnpp \
  --enable-nvenc \
  --enable-cuvid \
  --enable-nvdec \
  --enable-pic \
  --enable-rpath \
  --extra-cflags=-I/usr/local/cuda/include \
  --extra-ldflags=-L/usr/local/cuda/lib64 \
  && make -j$(nproc) \
  && make install

# Cleanup
# WORKDIR /root
# RUN rm -rf FFmpeg

RUN apt-get install -y python3-pip && apt-get clean

RUN pip install \
  cython \
  numpy==1.23.5 \
  screeninfo \
  yacs \
  moviepy \
  opencv-python \
  PyYAML \
  cython-bbox \
  scipy \
  progress \
  motmetrics \
  matplotlib \
  openpyxl \
  Pillow-simd \
  psutil \
  filterpy \
  screeninfo \
  gast \
  astunparse \
  fvcore \
  opencv-python \
  opencv-contrib-python \
  cameratransform \
  tifffile \
  scikit-build \
  scikit-learn \
  lmdb \
  loguru \
  pycocotools \
  sympy \
  numba \
  pyquaternion

# Again, force the proper numpy version
RUN pip install numpy==1.23.5

#
# Build PyTorch
#
WORKDIR /root
RUN mkdir src && cd src && git clone https://github.com/pytorch/pytorch --branch=v2.2.1 --recursive

# TODO: move to cuda install above
RUN apt-get install -y cuda-12-3 cudnn9-cuda-12

WORKDIR /root/src/pytorch
RUN pip install -r requirements.txt
RUN TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.9" \
  CUDA_HOME=/usr/local/cuda \
  CUDA_ROOT=/usr/local/cuda \
  CMAKE_CUDA_COMPILER="/usr/bin/nvcc" \
  CUDNN_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu \
  CUDNN_INCLUDE_PATH=/usr/include \
  CUDNN_INCLUDE_DIR=/usr/include \
  BUILD_TEST=0 \
  USE_CUDA=1 \
  USE_NUMPY=1 \
  USE_ROCM=OFF \
  BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 \
  python3 setup.py bdist_wheel

# WORKDIR /root
# RUN git clone https://github.com/cjolivier01/vigra && \
#   cd vigra && \
#   git checkout colivier/hm

  # && mkdir build \
  # && cd build \
  # && ../cfig \
  # && make $(nproc) \
  # && make install

# Cleanup
# WORKDIR /root
# RUN rm -rf vigra

# Entry point
CMD ["/bin/bash"]

