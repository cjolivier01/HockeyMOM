#!/bin/bash
docker build --memory 45G  --build-arg USERNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) . -t hm