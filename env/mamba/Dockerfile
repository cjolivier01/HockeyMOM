FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Build up the base system
RUN apt-get update
# RUN apt-get install -y libnvidia-compute-550

# Install necessary packages
RUN apt-get install -y wget ssh openssh-server

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
  libomp-dev \
  && apt-get clean

# Install NVIDIA driver dependencies and the NVIDIA codec SDK
# CUDA 12.3, CUDNN 8 (this package "libcudnn8-dev" seems to be built for CUDA 12.2)
RUN apt-get install -y \
  cuda-12-3 \
  libcudnn8-dev \
  cuda-toolkit-12-3 \
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
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
RUN bash ./Mambaforge-Linux-x86_64.sh -b -p /home/$USERNAME/dconda
RUN rm Mambaforge-Linux-x86_64.sh

# Set the environment path
ENV PATH /home/$USERNAME/dconda/bin:$PATH

# # Create a Python environment called 'ubuntu'
RUN mamba create -n ubuntu python=3.11 -y

# # Activate the environment automatically on login
RUN echo "source /home/${USER}/dconda/bin/activate ubuntu" >> /home/$USERNAME/.bashrc
RUN echo "alias cls=clear" >> /home/$USERNAME/.bashrc

####################################################################
#
# Build packages
#
####################################################################



####################################################################
#
# Switch back to root to configure SSH
#
####################################################################
USER root
RUN mkdir /var/run/sshd
# RUN echo 'myuser:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# # Expose the default SSH port
EXPOSE 22

# # Start the SSH daemon
# CMD ["/usr/sbin/sshd", "-D"]
CMD ["/bin/bash"]
