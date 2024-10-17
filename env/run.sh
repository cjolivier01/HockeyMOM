#!/bin/bash
docker run --gpus all --privileged -it --user=$(id -u):$(id -g) \
  -v /home/$USER:/home/$USER \
  -v /mnt/home/$USER:/mnt/home/$USER \
  --workdir=/home/$USER $@ hm
