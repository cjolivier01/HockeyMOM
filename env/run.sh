#!/bin/bash
docker run --gpus all --privileged --user=$(id -u):$(id -g) -it \
  --memory 40g \
  -v /home/$USER:/home/$USER \
  -v /mnt/home/$USER:/mnt/home/$USER \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro -v /etc/gshadow:/etc/gshadow:ro -v /etc/sudoers:/etc/sudoers:ro \
  --workdir=/home/$USER $@ hm
