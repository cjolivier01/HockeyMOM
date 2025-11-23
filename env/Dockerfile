FROM nvidia/cuda:13.0.0-base-ubuntu24.04 

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Build up the base system
RUN apt-get update

# Install necessary packages
RUN apt-get install -y wget ssh openssh-server curl

# Update package list and install essential build tools
RUN apt-get update && apt-get install -y \
  build-essential \
  git git-lfs \
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
  libomp-dev \
  && apt-get clean

# Install NVIDIA driver dependencies and the NVIDIA codec SDK
RUN apt-get install -y \
  cuda-toolkit-13-0 \
  cudnn9-cuda-13 \
  libffmpeg-nvenc-dev \
  && apt-get clean

# # Install FFmpeg build dependencies
RUN apt-get install -y \
  vim \
  autoconf \
  automake \
  cmake \
  ninja-build \
  libtool \
  libass-dev \
  libfreetype6-dev \
  libsdl2-dev \
  libx264-dev \
  libx265-dev \
  libtheora-dev \
  libva-dev \
  libgnutls28-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  pkg-config \
  texinfo \
  zlib1g-dev \
  && apt-get clean

RUN apt-get install -y libmkl-full-dev && apt-get clean
RUN apt-get install -y libwxgtk3.2-dev && apt-get clean

####################################################################
#
# User setup
#
####################################################################
ARG USERNAME
ARG UID
ARG GID

ENV SHELL=/bin/bash

# Create a new user with the provided USERNAME, UID, and GID
RUN groupadd -g $GID $USERNAME \
    && useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME

# Set a password for the user
# RUN echo "$USERNAME:password" | chpasswd

# Set the user and prevent the container from running as root
USER $USERNAME

# Set the working directory
WORKDIR /home/$USERNAME

####################################################################
#
# Download and install Mambaforge for the non-root user
#
####################################################################
USER root
RUN mkdir -p /opt/$USERNAME
RUN chown $USERNAME /opt/$USERNAME

# USER $USERNAME
# WORKDIR /home/$USERNAME
# RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
# RUN bash ./Mambaforge-Linux-x86_64.sh -b -p /opt/$USERNAME/dconda
# RUN rm Mambaforge-Linux-x86_64.sh

# Set the environment path
# ENV PATH /opt/$USERNAME/dconda/bin:$PATH

# # Create a Python environment called 'ubuntu'
# RUN mamba create -n ubuntu python=3.12 -y

# # Activate the environment automatically on login
# RUN echo "source /opt/$USERNAME/dconda/bin/activate ubuntu" >> /home/$USERNAME/.bashrc
RUN echo "alias cls=clear" >> /home/$USERNAME/.bashrc

####################################################################
#
#
# Build packages
#
#
####################################################################
USER $USERNAME
RUN mkdir src
WORKDIR /home/$USERNAME/src

####################################################################
#
# FFMPeg
#
####################################################################
USER root
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && git checkout n11.1.5.3 && make install
WORKDIR /home/$USERNAME/src
RUN rm -rf nv-codec-headers

USER $USERNAME
RUN git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg && cd ffmpeg && git checkout n5.1.4

# Configure FFmpeg with the desired codecs and options
WORKDIR /home/$USERNAME/src/ffmpeg
RUN NVCCFLAGS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_89,code=sm_89" \
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
  && make -j$(nproc)

USER root
RUN make -j $(nproc) install

# USER $USERNAME
# WORKDIR /home/$USERNAME/src

####################################################################
#
# PIP Dependencies
#
####################################################################
# RUN conda init

USER root
RUN apt-get install -y python3-pip
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
    pyquaternion \
    typing_extensions

# # Again, force the proper numpy version
RUN pip install numpy==1.23.5


# ####################################################################
# #
# # Build Vigra (dependency of Hugin tools)
# # We need to build it because it has a patch in one of the headers
# # that fixes a compile problem with Hugin.
# #
# ####################################################################
WORKDIR /home/$USERNAME/src
RUN git clone https://github.com/cjolivier01/vigra && \
   cd vigra && \
   git checkout colivier/hm \
   && mkdir build \
   && cd build \
   && ../cfig \
   && make -j $(nproc)

USER $USERNAME
WORKDIR /home/$USERNAME/src/vigra/build
USER root
RUN make install
WORKDIR /home/$USERNAME/src
RUN rm -rf vigra

# ####################################################################
# #
# # Make Wheels Dirs
# #
# ####################################################################
USER root
RUN mkdir /wheels


# ####################################################################
# #
# # Build PyTorch
# #
# ####################################################################
USER $USERNAME
WORKDIR /home/$USERNAME/src
RUN git clone https://github.com/cjolivier01/pytorch --branch=v2.9.0 --recursive

WORKDIR /home/$USERNAME/src/pytorch
USER root
RUN pip install -r requirements.txt

USER $USERNAME
RUN \
  MAX_JOBS=12 \
  TORCH_CUDA_ARCH_LIST="7.5;8.9;9.0;10.0;10.3;11.0;12.0;12.1" \
  BUILD_TEST=0 \
  USE_CUDA=1 \
  USE_NUMPY=1 \
  USE_ROCM=OFF \
  BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 \
  python3 setup.py bdist_wheel

USER root
RUN pip install dist/*.whl
RUN cp dist/*.whl /wheels/

# ####################################################################
# #
# # Build TorchVision
# #
# ####################################################################
USER $USERNAME
WORKDIR /home/$USERNAME/src
RUN git clone https://github.com/pytorch/vision --branch=v0.18.0 --recursive
WORKDIR /home/$USERNAME/src/vision
RUN \
  MAX_JOBS=12 \
  TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.9;9.0" \
  BUILD_TEST=0 \
  USE_CUDA=1 \
  USE_CUDNN=1 \
  USE_NUMPY=1 \
  USE_ROCM=OFF \
  BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 \
  python3 setup.py bdist_wheel

USER root
RUN pip install dist/*.whl
RUN cp dist/*.whl /wheels/

####################################################################
#
# Build TorchAudio
#
####################################################################
USER $USERNAME
WORKDIR /home/$USERNAME/src
RUN git clone https://github.com/pytorch/audio --branch=v2.1.2 --recursive
WORKDIR /home/$USERNAME/src/audio
RUN \
  MAX_JOBS=16 \
  TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.9;9.0" \
  BUILD_TEST=0 \
  USE_CUDA=1 \
  USE_NUMPY=1 \
  USE_ROCM=OFF \
  BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 \
  python3 setup.py bdist_wheel

USER root
RUN pip install dist/*.whl
RUN cp dist/*.whl /wheels/

####################################################################
#
# Build and install Hugin
#
####################################################################

USER $USERNAME
WORKDIR /home/$USERNAME/src
RUN git clone https://github.com/cjolivier01/hugin --branch=colivier/hockeymom --recursive
WORKDIR /home/$USERNAME/src/hugin
RUN mkdir build && cd build && \
  cmake .. && \
  make -j $(nproc)

USER root
WORKDIR /home/$USERNAME/src/hugin/build
RUN make install

####################################################################
#
# Build and install OpenCV
#
####################################################################


#
# Clean up builds
#
USER root
WORKDIR /home/$USERNAME
RUN rm -rf src/*

#
# Some environment stuff
#
USER $USERNAME
# RUN echo "export HMBASE_CONDA_PATH=/opt/$USERNAME/dconda/bin" >> /etc/profile
# RUN echo "export HMBASE_CONDA_ENV=ubuntu"  >> /etc/profile


####################################################################
#
# Switch back to root to configure SSH
#
####################################################################
USER root
RUN mkdir /var/run/sshd

# Update SSH configuration to expose it on a non-standard port and permit password authentication
RUN sed -i 's/#Port 22/Port 22298/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#HostKey/HostKey/' /etc/ssh/sshd_config

RUN ssh-keygen -y -t rsa -f /etc/ssh/ssh_host_rsa_key -N ''
RUN ssh-keygen -y -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -N ''
RUN ssh-keygen -y -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -N ''
RUN chmod 600 /etc/ssh/ssh_host_*_key
RUN chmod 644 /etc/ssh/ssh_host_*_key.pub

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# Expose the SSH port
EXPOSE 22298

# Start the SSH daemon
CMD ["/usr/sbin/sshd", "-d", "-p", "22298"]
# CMD ["/bin/bash"]
