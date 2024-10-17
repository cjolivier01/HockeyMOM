#!/bin/bash
docker run --gpus all --privileged --user=$(id -u):$(id -g) -d \
  -v /home:/home \
  -v /mnt/home:/mnt/home \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro -v /etc/gshadow:/etc/gshadow:ro -v /etc/sudoers:/etc/sudoers:ro \
  --name hockeymom \
  --workdir=/home/$USER $@ hm
