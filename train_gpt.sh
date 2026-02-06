#!/bin/bash
./camgpt_train.sh --file list ~/RVideos/all.lst --no-pose --include-rink --steps 300000 --data-workers 32 --preload-csv shard $@
