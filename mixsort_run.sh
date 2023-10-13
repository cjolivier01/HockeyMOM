#!/bin/bash
python src/hm_track_mixsort.py -expn=my_experiment -f=models/mixsort/exps/example/mot/yolox_x_sportsmot.py -c=models/mixsort/pretrained/yolox_x_sports_train.pth -b=1 -d=1 --config=track
