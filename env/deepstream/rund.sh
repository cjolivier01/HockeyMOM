#!/bin/bash

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

source ${SCRIPT_DIR}/../tools.sh

DOCKER_OPTS=""

DOCKER_TAG=$(get_tag)
if [ ! -z "${DOCKER_TAG}" ]; then
  DOCKER_OPTS="${DOCKER_TAG}"
  echo "DOCKER_TAG=${DOCKER_TAG}"
fi

# -v /etc/sudoers:/etc/sudoers:ro

docker run --gpus all --privileged --user=$(id -u):$(id -g) -d \
  --memory 32g \
  -p 22298:22298 \
  -v ${HOME}:${HOME} \
  -v ${HOME}/.ssh:${HOME}/.ssh \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /etc/shadow:/etc/shadow:ro \
  -v /etc/gshadow:/etc/gshadow:ro \
  --workdir=${HOME} $@ ${DOCKER_OPTS}
