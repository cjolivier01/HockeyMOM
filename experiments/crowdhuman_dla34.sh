cd src
PYTHONPATH="$(pwd):$(pwd)/hmlib:$(pwd)/../models/mixsort:${PYTHONPATH}" \
  LD_LIBRARY_PATH=/home/colivier/.conda/envs/ubuntu-gpu/lib:/apcv/shared/cuda/cudnn/8.2.0.37/lib64:/apcv/shared/conda-envs/ai-1555/lib:/apcv/shared/conda-envs/ai-1555/lib/x86_64-linux-gnu:/cm/local/apps/cuda/libs/470.199.02-0ubuntu0.20.04.1/lib64:/cm/shared/apps/cuda11.4/toolkit/11.4.0/targets/x86_64-linux/lib \
  python train.py mot --exp_id crowdhuman_dla34 --gpus 0,1 --batch_size 18 --load_model '../pretrained/dla34/ctdet_coco_dla_2x.pth' --num_epochs 60 --print_iter 25 --lr_step '50' --data_cfg '../src/hmlib/cfg/crowdhuman.json' $@
cd ..
