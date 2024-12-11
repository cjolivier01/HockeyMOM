#!/bin/bash

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

source ${SCRIPT_DIR}/../tools.sh

DOCKER_OPTS=""

if [ -z "${DOCKER_TAG}" ]; then
  DOCKER_TAG=$(get_tag)
fi

if [ ! -z "${DOCKER_TAG}" ]; then
  DOCKER_OPTS="${DOCKER_TAG}"
  echo "DOCKER_TAG=${DOCKER_TAG}"
fi

#   -v /etc/sudoers:/etc/sudoers:ro 

if [ "$(is_arm)" == "1" ]; then
  GPU_FLAGS="--runtime nvidia"
else
  GPU_FLAGS="--gpus all"
fi

docker run ${GPU_FLAGS} --privileged --user=$(id -u):$(id -g) -it \
  -e DEEPSTREAM_CONTAINER=1 \
  -e DISPLAY=${DISPLAY} \
  --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e CUDA_CACHE_DISABLE=0 \
  --memory 32g \
  -p 22298:22298 \
  -v /mnt/home:/mnt/home \
  -v /mnt/data:/mnt/data \
  -v ${HOME}:${HOME} \
  -v ${HOME}/.ssh:${HOME}/.ssh \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /etc/shadow:/etc/shadow:ro \
  -v /etc/gshadow:/etc/gshadow:ro \
  --workdir=${HOME} $@ ${DOCKER_OPTS}
