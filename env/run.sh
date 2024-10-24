#!/bin/bash

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

source ${SCRIPT_DIR}/tools.sh

DOCKER_OPTS=""

DOCKER_TAG=$(get_tag)
if [ ! -z "${DOCKER_TAG}" ]; then
  DOCKER_OPTS="${DOCKER_TAG}"
  echo "DOCKER_TAG=${DOCKER_TAG}"
fi

docker run --gpus all --privileged --user=$(id -u):$(id -g) -it \
  --memory 32g \
  -v ${HOME}:${HOME} \
  -v ${HOME}/.ssh:${HOME}/.ssh \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /etc/shadow:/etc/shadow:ro \
  -v /etc/gshadow:/etc/gshadow:ro \
  -v /etc/sudoers:/etc/sudoers:ro \
  --workdir=${HOME} $@ ${DOCKER_OPTS}
