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

IMAGE="$(docker ps | grep deepstream | awk '{print$1}')"
if [ -z "${IMAGE}" ]; then
  echo "Could not find running deepastream image"
  exit 1
fi

#   -v /etc/sudoers:/etc/sudoers:ro 

docker exec --privileged --user=$(id -u):$(id -g) -it \
  -e DEEPSTREAM_CONTAINER=1 \
  -e DISPLAY=${DISPLAY} \
  -e DISPLAY=$DISPLAY -e CUDA_CACHE_DISABLE=0 \
  --workdir=${HOME} ${IMAGE} $@ /bin/bash
