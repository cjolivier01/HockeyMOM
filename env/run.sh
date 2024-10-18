#!/bin/bash

# -v /mnt/home/$USER:/mnt/home/$USER

docker run --gpus all --privileged --user=$(id -u):$(id -g) -it \
  --memory 32g \
  -v ${HOME}:${HOME} \
  -v ${HOME}/.ssh:${HOME}/.ssh \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /etc/shadow:/etc/shadow:ro \
  -v /etc/gshadow:/etc/gshadow:ro \
  -v /etc/sudoers:/etc/sudoers:ro \
  --workdir=${HOME} $@
