#FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel
FROM nvcr.io/nvidia/deepstream:6.4-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive


ARG USERNAME
ARG UID
ARG GID

ENV SHELL=/bin/bash

USER root
RUN apt-get update
RUN apt-get install sudo
# CMD ["/bin/bash"]

RUN echo 'root:password' | chpasswd

# User is needed by git
RUN groupadd -r $USERNAME
#RUN groupadd -r colivier
#RUN groupadd -r colivier-local
RUN groupadd -r docker
RUN useradd -r -g $USERNAME -G $USERNAME,docker,sudo $USERNAME
#RUN useradd -r -g colivier -G colivier,docker,sudo colivier
#RUN useradd -r -g colivier-local -G colivier-local,docker,sudo colivier-local

RUN echo '%sudo   ALL=(ALL:ALL) ALL' >> /etc/sudoers
RUN echo "$USERNAME ALL=(ALL:ALL) ALL" >> /etc/sudoers
