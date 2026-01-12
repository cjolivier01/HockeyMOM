ARG CUDA_BASE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_BASE}

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=hm
ARG UID=1000
ARG GID=1000

# Keep bash in RUN steps so we can source conda, use braces, etc.
SHELL ["/bin/bash", "-lc"]

####################################################################
#
# Base system deps (build + runtime)
#
####################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  ccache \
  curl \
  git \
  git-lfs \
  openssh-client \
  sudo \
  wget \
  rsync \
  unzip \
  zip \
  build-essential \
  pkg-config \
  yasm \
  nasm \
  autoconf \
  automake \
  cmake \
  ninja-build \
  libtool \
  vim \
  graphviz \
  graphviz-dev \
  gettext \
  libyaml-cpp-dev \
  libprotobuf-dev \
  protobuf-compiler \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libgl-dev \
  libglew-dev \
  libjpeg-turbo8-dev \
  libtiff-dev \
  libpng-dev \
  libopenexr-dev \
  libexiv2-dev \
  libfftw3-dev \
  libilmbase-dev \
  liblcms2-dev \
  libsqlite3-dev \
  liblapack-dev \
  libopenblas-dev \
  libomp-dev \
  libboost-all-dev \
  libpano13-dev \
  libglfw3-dev \
  libgsl-dev \
  help2man \
  texlive-latex-base \
  libwxgtk3.0-gtk3-dev \
  libx264-dev \
  libx265-dev \
  libvpx-dev \
  libfdk-aac-dev \
  libmp3lame-dev \
  libopus-dev \
  libass-dev \
  libfreetype6-dev \
  libsdl2-dev \
  libtheora-dev \
  libva-dev \
  libgnutls28-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  texinfo \
  zlib1g-dev \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libswscale-dev \
  libswresample-dev \
  && rm -rf /var/lib/apt/lists/*

####################################################################
#
# User setup
#
####################################################################
RUN set -eux; \
  if getent group "${GID}" >/dev/null 2>&1; then \
    group_name="$(getent group "${GID}" | cut -d: -f1)"; \
  else \
    group_name="${USERNAME}"; \
    groupadd --gid "${GID}" "${group_name}"; \
  fi; \
  if id -u "${USERNAME}" >/dev/null 2>&1; then \
    true; \
  else \
    useradd -m -s /bin/bash --uid "${UID}" --gid "${group_name}" -G sudo "${USERNAME}"; \
  fi; \
  echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/"${USERNAME}"; \
  chmod 0440 /etc/sudoers.d/"${USERNAME}"

####################################################################
#
# Conda (needed for Bazel conda_repository + Python runtime)
#
####################################################################
ENV CONDA_DIR=/opt/conda
RUN wget -O /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
  && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
  && rm -f /tmp/miniforge.sh \
  && "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false \
  && "${CONDA_DIR}/bin/conda" config --system --set show_channel_urls true \
  && "${CONDA_DIR}/bin/conda" create -y -n hm python=3.10 pip \
  && "${CONDA_DIR}/bin/conda" clean -a -y

ENV CONDA_PREFIX=${CONDA_DIR}/envs/hm
ENV PATH=${CONDA_PREFIX}/bin:${CONDA_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

RUN pip install --no-cache-dir -U pip setuptools wheel

####################################################################
#
# Bazelisk (used by ./run.sh and Bazel builds)
#
####################################################################
RUN curl -fsSL -o /usr/local/bin/bazelisk \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64 \
  && chmod +x /usr/local/bin/bazelisk \
  && ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel

####################################################################
#
# Build FFmpeg with CUDA/NVENC/NVDEC support
#
####################################################################
WORKDIR /tmp
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
  && cd nv-codec-headers \
  && git checkout n11.1.5.3 \
  && make -j"$(nproc)" install \
  && cd /tmp \
  && rm -rf nv-codec-headers

RUN git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg \
  && cd ffmpeg \
  && git checkout n5.1.4 \
  && NVCCFLAGS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_89,code=sm_89" \
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
  && make -j"$(nproc)" \
  && make -j"$(nproc)" install \
  && cd /tmp \
  && rm -rf ffmpeg

####################################################################
#
# Build VIGRA (patched) for Hugin
#
####################################################################
WORKDIR /tmp
RUN git clone https://github.com/cjolivier01/vigra \
  && cd vigra \
  && git checkout colivier/hm \
  && mkdir -p build \
  && cd build \
  && ../cfig \
  && make -j"$(nproc)" \
  && make -j"$(nproc)" install \
  && cd /tmp \
  && rm -rf vigra

####################################################################
#
# Copy source tree (kept in the image for development)
#
####################################################################
WORKDIR /opt/hm/src/hm
COPY . /opt/hm/src/hm
RUN chown -R "${UID}:${GID}" /opt/hm

####################################################################
#
# Python deps (runtime + build)
#
####################################################################
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
ARG TORCHAUDIO_VERSION=2.4.1

RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

# PyNvVideoCodec is hosted on NVIDIA's index for many setups.
# Avoid `nvidia-pyindex` here: its build backend expects `pip` to be importable
# inside an isolated PEP517 build env, which isn't always true.
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com PyNvVideoCodec

RUN pip install --no-cache-dir -r requirements.txt

####################################################################
#
# Build + install native tools and wheels
#
####################################################################
RUN cd /opt/hm/src/hm/external/hugin \
  && bazelisk run //:install_tree -- --prefix=/usr/local

RUN cd /opt/hm/src/hm \
  && ./run.sh //hockeymom:bdist_wheel \
  && ./run.sh //hmlib:bdist_wheel \
  && pip install --no-cache-dir dist/hockeymom-*.whl dist/hmlib-*.whl

####################################################################
#
# Runtime defaults
#
####################################################################
USER "${USERNAME}"
WORKDIR /home/"${USERNAME}"
RUN mkdir -p "/home/${USERNAME}/Videos"

ENV PYTHONPATH=/opt/hm/src/hm:/opt/hm/src/hm/src

CMD ["/bin/bash"]
