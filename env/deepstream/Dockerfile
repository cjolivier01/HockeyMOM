FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel
# FROM nvcr.io/nvidia/deepstream:6.4-triton-multiarch


ENV DEBIAN_FRONTEND=noninteractive


ARG USERNAME
ARG UID
ARG GID

ENV SHELL=/bin/bash

USER root
RUN apt update

RUN apt-get install -y \
  sudo \
  nasm \
  gnutls-dev \
  libx264-dev \
  x264 \
  aptitude \
  libvpx-dev \
  libffmpeg-nvenc-dev
# RUN apt-get clean

# why is this necessary?
RUN apt-get install -y --reinstall libx264-dev libx264-163 x264

####################################################################
#
#
# Build packages
#
#
####################################################################

####################################################################
#
# FFMPeg
#
####################################################################
USER root
# RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && git checkout n11.1.5.3 && make install
# WORKDIR /home/$USERNAME/src
# RUN rm -rf nv-codec-headers

WORKDIR /root
RUN git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg && cd ffmpeg && git checkout n5.1.4

# Configure FFmpeg with the desired codecs and options
WORKDIR /root/ffmpeg
RUN ./configure \
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

RUN make -j $(nproc) install
RUN rm -rf ffmpeg

RUN apt-get install -y openssh-server
RUN apt-get install -y gedit


#
# VSCode Dev Containers packages
#
RUN apt-get install -y \
	apt-utils \
        bash-completion \
        openssh-client \
        gnupg2 \
        dirmngr \
        iproute2 \
        procps \
        lsof \
        htop \
        net-tools \
        psmisc \
        curl \
        tree \
        wget \
        rsync \
        ca-certificates \
        unzip \
        bzip2 \
        xz-utils \
        zip \
        nano \
        vim-tiny \
        less \
        jq \
        lsb-release \
        apt-transport-https \
        dialog \
        libc6 \
        libgcc1 \
        libkrb5-3 \
        libgssapi-krb5-2 \
        libicu[0-9][0-9] \
        liblttng-ust[0-9] \
        libstdc++6 \
        zlib1g \
        locales \
        sudo \
        ncdu \
        man-db \
        strace \
        manpages \
        manpages-dev \
        init-system-helpers        

RUN apt-get upgrade -y

# Create a new user with the provided USERNAME, UID, and GID
USER root
RUN find /opt/nvidia/deepstream/ -type d  | xargs sudo chmod 777
RUN echo '%sudo   ALL=(ALL:ALL) ALL' >> /etc/sudoers
RUN echo "$USERNAME ALL=(ALL:ALL) ALL" >> /etc/sudoers

# Create the SSH directory and set up a basic configuration
RUN mkdir /var/run/sshd

# Set the root password (for demo purposes, use a secure password in production)
RUN echo 'root:password' | chpasswd

RUN groupadd -r docker
RUN useradd -r -g docker colivier
RUN useradd -r -g docker colivier-local

# Allow root login via SSH (for demo purposes)
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# RUN cat /etc/sudoers
# RUN usermod -aG sudo $USERNAME

# Set a password for the user
# RUN echo "$USERNAME:password" | chpasswd

# Set the user and prevent the container from running as root
USER $USERNAME

# Set the working directory
WORKDIR /home/$USERNAME


EXPOSE 22299

# Start the SSH daemon
USER root
#CMD ["/usr/sbin/sshd", "-d", "-p", "22299"]
CMD ["/bin/bash"]
