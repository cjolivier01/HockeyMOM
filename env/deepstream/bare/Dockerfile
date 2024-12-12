FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel
# FROM nvcr.io/nvidia/deepstream:6.4-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive


ARG USERNAME
ARG UID
ARG GID

ENV SHELL=/bin/bash

USER root
RUN apt-get update
RUN apt-get install sudo
# CMD ["/bin/bash"]
