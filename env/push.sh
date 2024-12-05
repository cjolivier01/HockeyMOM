#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath "${BASH_SOURCE[0]}"))

source ${SCRIPT_DIR}/tools.sh

DOCKER_OPTS=""

DOCKER_TAG=$(get_tag)
if [ ! -z "${DOCKER_TAG}" ]; then
  DOCKER_OPTS="${DOCKER_TAG}"
  echo "DOCKER_TAG=${DOCKER_TAG}"
fi

SHORT_USER="${USER%%-*}"

if [ "${SHORT_USER}" == "colivier" ]; then
  SHORT_USER="cjolivier01"
fi

echo "Pushign ${SHORT_USER}/${DOCKER_TAG}"
docker tag ${DOCKER_TAG} ${SHORT_USER}/${DOCKER_TAG}
docker push ${SHORT_USER}/${DOCKER_TAG}

