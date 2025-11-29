"""Hm2 detector-only config: YOLOX-S trained on CrowdHuman."""

auto_scale_lr = dict(base_batch_size=64)
backend_args = None
base_lr = 0.0006000000000000001
custom_hooks = [
    dict(num_last_epochs=4, priority=48, type="YOLOXModeSwitchHook"),
    dict(priority=48, type="SyncNormHook"),
    dict(ema_type="ExpMomentumEMA", momentum=0.0001, priority=49, type="EMAHook", update_buffers=True),
]
data_root = "data/crowdhuman/"
dataset_type = "CocoDataset"
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type="CheckpointHook"),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False, dist_cfg=dict(backend="nccl"), mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0)
)
gpus_per_node = 1
interval = 1
launcher = "none"
load_from = None
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(type="Swish"),
        deepen_factor=0.33,
        norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
        out_indices=(
            2,
            3,
            4,
        ),
        spp_kernal_sizes=(
            5,
            9,
            13,
        ),
        type="CSPDarknet",
        use_depthwise=False,
        widen_factor=0.5,
    ),
    bbox_head=dict(
        act_cfg=dict(type="Swish"),
        feat_channels=128,
        in_channels=128,
        loss_bbox=dict(eps=1e-16, loss_weight=5.0, mode="square", reduction="sum", type="IoULoss"),
        loss_cls=dict(loss_weight=1.0, reduction="sum", type="CrossEntropyLoss", use_sigmoid=True),
        loss_l1=dict(loss_weight=1.0, reduction="sum", type="L1Loss"),
        loss_obj=dict(loss_weight=1.0, reduction="sum", type="CrossEntropyLoss", use_sigmoid=True),
        norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
        num_classes=80,
        stacked_convs=2,
        strides=(
            8,
            16,
            32,
        ),
        type="YOLOXHead",
        use_depthwise=False,
    ),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=10,
                random_size_range=(
                    480,
                    800,
                ),
                size_divisor=32,
                type="BatchSyncRandomResize",
            ),
        ],
        pad_size_divisor=32,
        type="DetDataPreprocessor",
    ),
    init_cfg=dict(checkpoint="/mnt/data/pretrained/mmdetection/yolox_s_8x8_300e_coco_80e_ch.pth", type="Pretrained"),
    neck=dict(
        act_cfg=dict(type="Swish"),
        in_channels=[
            128,
            256,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
        num_csp_blocks=1,
        out_channels=128,
        type="YOLOXPAFPN",
        upsample_cfg=dict(mode="nearest", scale_factor=2),
        use_depthwise=False,
    ),
    test_cfg=dict(nms=dict(iou_threshold=0.65, type="nms"), score_thr=0.01),
    train_cfg=dict(assigner=dict(center_radius=2.5, type="SimOTAAssigner")),
    type="YOLOX",
)
node_count = 1
num_last_epochs = 4
optim_wrapper = dict(
    optimizer=dict(lr=0.0006000000000000001, momentum=0.9, nesterov=True, type="SGD", weight_decay=0.0005),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type="OptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=True, convert_to_iter_based=True, end=1, type="mmdet.QuadraticWarmupLR"),
    dict(
        T_max=2,
        begin=1,
        by_epoch=True,
        convert_to_iter_based=True,
        end=2,
        eta_min=3.0000000000000004e-05,
        type="CosineAnnealingLR",
    ),
    dict(begin=2, by_epoch=True, end=6, factor=1, type="ConstantLR"),
]
resume = True
samples_per_gpu = 6
samples_per_gpu_v100_16gb_yolox_s = 6
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=6,
    dataset=dict(
        ann_file="crowdhuman_val.json",
        backend_args=None,
        data_prefix=dict(img="val/"),
        data_root="data/crowdhuman/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    2240,
                    800,
                ),
                type="Resize",
            ),
            dict(size_divisor=32, type="Pad"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(
    ann_file="data/crowdhuman/crowdhuman_val.json", backend_args=None, metric="bbox", type="CocoMetric"
)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=(
            2240,
            800,
        ),
        type="Resize",
    ),
    dict(size_divisor=32, type="Pad"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
total_epochs = 6
train_cfg = dict(max_epochs=6, type="EpochBasedTrainLoop", val_interval=1)
train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        dataset=dict(
            datasets=[
                dict(
                    ann_file="crowdhuman_train.json",
                    backend_args=None,
                    data_prefix=dict(img="train"),
                    data_root="data/crowdhuman/",
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[
                        dict(backend_args=None, type="LoadImageFromFile"),
                        dict(type="LoadAnnotations", with_bbox=True),
                    ],
                    type="CocoDataset",
                ),
                dict(
                    ann_file="crowdhuman_val.json",
                    backend_args=None,
                    data_prefix=dict(img="val/"),
                    data_root="data/crowdhuman/",
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[
                        dict(backend_args=None, type="LoadImageFromFile"),
                        dict(type="LoadAnnotations", with_bbox=True),
                    ],
                    type="CocoDataset",
                ),
            ],
            type="ConcatDataset",
        ),
        pipeline=[
            dict(
                img_scale=(
                    2240,
                    800,
                ),
                pad_val=114.0,
                type="Mosaic",
            ),
            dict(
                border=(
                    -1120,
                    -400,
                ),
                max_rotate_degree=0.0,
                scaling_ratio_range=(
                    0.8,
                    1.6,
                ),
                type="RandomAffine",
            ),
            dict(
                img_scale=(
                    2240,
                    800,
                ),
                pad_val=114.0,
                ratio_range=(
                    0.8,
                    1.2,
                ),
                type="MixUp",
            ),
            dict(type="YOLOXHSVRandomAug"),
            dict(prob=0.5, type="RandomFlip"),
            dict(
                keep_ratio=True,
                scale=(
                    2240,
                    800,
                ),
                type="Resize",
            ),
            dict(size_divisor=32, type="Pad"),
            dict(type="PackDetInputs"),
        ],
        type="MultiImageMixDataset",
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_dataset = dict(
    dataset=dict(
        datasets=[
            dict(
                ann_file="crowdhuman_train.json",
                backend_args=None,
                data_prefix=dict(img="train"),
                data_root="data/crowdhuman/",
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=[
                    dict(backend_args=None, type="LoadImageFromFile"),
                    dict(type="LoadAnnotations", with_bbox=True),
                ],
                type="CocoDataset",
            ),
            dict(
                ann_file="crowdhuman_val.json",
                backend_args=None,
                data_prefix=dict(img="val/"),
                data_root="data/crowdhuman/",
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=[
                    dict(backend_args=None, type="LoadImageFromFile"),
                    dict(type="LoadAnnotations", with_bbox=True),
                ],
                type="CocoDataset",
            ),
        ],
        type="ConcatDataset",
    ),
    pipeline=[
        dict(
            img_scale=(
                2240,
                800,
            ),
            pad_val=114.0,
            type="Mosaic",
        ),
        dict(
            border=(
                -1120,
                -400,
            ),
            max_rotate_degree=0.0,
            scaling_ratio_range=(
                0.8,
                1.6,
            ),
            type="RandomAffine",
        ),
        dict(
            img_scale=(
                2240,
                800,
            ),
            pad_val=114.0,
            ratio_range=(
                0.8,
                1.2,
            ),
            type="MixUp",
        ),
        dict(type="YOLOXHSVRandomAug"),
        dict(prob=0.5, type="RandomFlip"),
        dict(
            keep_ratio=True,
            scale=(
                2240,
                800,
            ),
            type="Resize",
        ),
        dict(size_divisor=32, type="Pad"),
        dict(type="PackDetInputs"),
    ],
    type="MultiImageMixDataset",
)
train_pipeline = [
    dict(
        img_scale=(
            2240,
            800,
        ),
        pad_val=114.0,
        type="Mosaic",
    ),
    dict(
        border=(
            -1120,
            -400,
        ),
        max_rotate_degree=0.0,
        scaling_ratio_range=(
            0.8,
            1.6,
        ),
        type="RandomAffine",
    ),
    dict(
        img_scale=(
            2240,
            800,
        ),
        pad_val=114.0,
        ratio_range=(
            0.8,
            1.2,
        ),
        type="MixUp",
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        keep_ratio=True,
        scale=(
            2240,
            800,
        ),
        type="Resize",
    ),
    dict(size_divisor=32, type="Pad"),
    dict(type="PackDetInputs"),
]
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=6,
    dataset=dict(
        ann_file="crowdhuman_val.json",
        backend_args=None,
        data_prefix=dict(img="val/"),
        data_root="data/crowdhuman/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    2240,
                    800,
                ),
                type="Resize",
            ),
            dict(size_divisor=32, type="Pad"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_dataset = dict(
    ann_file="crowdhuman_val.json",
    backend_args=None,
    data_prefix=dict(img="val/"),
    data_root="data/crowdhuman/",
    pipeline=[
        dict(type="LoadImageFromFile"),
        dict(
            keep_ratio=True,
            scale=(
                2240,
                800,
            ),
            type="Resize",
        ),
        dict(size_divisor=32, type="Pad"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
            ),
            type="PackDetInputs",
        ),
    ],
    test_mode=True,
    type="CocoDataset",
)
val_evaluator = dict(
    ann_file="data/crowdhuman/crowdhuman_val.json", backend_args=None, metric="bbox", type="CocoMetric"
)
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
work_dir = "./work_dirs/hm_crowdhuman_yolox_s_2240x800"
yolox_img_scale = (
    2240,
    800,
)
