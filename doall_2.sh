#!/bin/bash
for i in $(cat ~/RVideos/all2.lst); do echo $i; ./hm_run.sh --game-id=$i $@; done
