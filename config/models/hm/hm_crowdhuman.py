_base_ = [
    "../datasets/crowdhuman_mot17.py",
    "../default_hm_runtime.py",
    "./hm_yolox_s.py",
]

import os

yolox_img_scale = (480, 1312)

detector_standalone_model = dict(
    input_size=yolox_img_scale,
    init_cfg=dict(
        type="Pretrained",
        checkpoint="/mnt/data/pretrained/mmdetection/yolox_s_8x8_300e_coco_80e_ch.pth",
    ),
)

# DGX A100 40GB
# samples_per_gpu_a100_40gb_yolox_s = 38
samples_per_gpu_a100_40gb_yolox_s = 16
samples_per_gpu_a100_40gb_yolox_s_tiny = 3
samples_per_gpu_a100_40gb_yolox_l = 28

# samples_per_gpu = samples_per_gpu_a100_40gb_yolox_s
samples_per_gpu = samples_per_gpu_a100_40gb_yolox_s_tiny

image_pad_value = 114.0

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     to_rgb=False,
# )

inference_pipeline = [
    dict(type="LoadImageFromFile"),
    # Crop first in order to not affect any subsequent coordinates,
    # as well as not waste compute on parts of the image that will be thrown away
    dict(type="HmCrop", keys=["img"], save_clipped_images=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=yolox_img_scale,
        allow_flip=False,
        transforms=[
            dict(type="HmImageToTensor", keys=["img"]),
            dict(type="HmResize", keep_ratio=True),
            dict(type="RandomFlip"),
            # dict(type="Normalize", **img_norm_cfg),
            dict(
                type="HmPad",
                pad_val=image_pad_value,
                size_divisor=32,
            ),
            dict(type="HmVideoCollect", keys=["img", "clipped_image"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    inference=dict(
        type="CocoDataset",
        ann_file="data/crowdhuman/annotations/val.json",
        img_prefix="data/crowdhuman/val/Images/",
        pipeline=inference_pipeline,
    ),
)

import os

import torch

node_count = int(os.environ.get("SLURM_NNODES", "1"))
gpus_per_node = torch.cuda.device_count()

optimizer = dict(
    type="SGD",
    lr=0.02 / (gpus_per_node * node_count) * samples_per_gpu,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=None)

# some hyper parameters
total_epochs = 80
num_last_epochs = 10
resume_from = None

# learning policy
lr_config = dict(
    policy="YOLOX",
    warmup="exp",
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05,
)

runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

checkpoint_config = dict(interval=10)
# checkpoint_config = dict(interval=1)
evaluation = dict(metric=["bbox"], interval=100)
search_metrics = ["MOTA", "IDF1", "FN", "FP", "IDs", "MT", "ML"]

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
# fp16 = dict(loss_scale=dict(init_scale=512.0))
