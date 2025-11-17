"""End-to-end hm2 config that composes detector, tracker, pose, and scoreboard."""

#
# Joined version of hm_end_to_end.py
#
auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
base_lr = 0.0005
batch_size = 4
custom_hooks = [
    dict(num_last_epochs=10, priority=48, type="YOLOXModeSwitchHook"),
    dict(priority=48, type="SyncNormHook"),
    dict(ema_type="ExpMomentumEMA", momentum=0.0001, priority=49, type="EMAHook", update_buffers=True),
]
data_root = "data/MOT17/"
dataset_type = "MOTChallengeDataset"
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=10, type="CheckpointHook"),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(draw=False, type="TrackVisualizationHook"),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False, dist_cfg=dict(backend="nccl"), mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0)
)
img_scale = (
    1312,
    480,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
inference_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        keys=[
            "img",
        ],
        save_clipped_images=True,
        type="HmCrop",
    ),
    dict(
        keys=[
            "img",
        ],
        type="HmImageToTensor",
    ),
    dict(
        img_scale=(
            1312,
            480,
        ),
        keep_ratio=True,
        type="HmResize",
    ),
    dict(pad_val=114.0, size_divisor=32, type="HmPad"),
    dict(
        meta_keys=(
            "img_metas",
        ),
        type="mmdet.PackTrackInputs",
    ),
]
interval = 5
load_from = None
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
max_epochs = 80
model = dict(
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=10,
                random_size_range=(
                    576,
                    1024,
                ),
                size_divisor=32,
                type="BatchSyncRandomResize",
            ),
        ],
        pad_size_divisor=32,
        type="TrackDataPreprocessor",
        use_det_processor=True,
    ),
    dataset_meta=dict(classes=None),
    detector=dict(
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
        init_cfg=dict(
            checkpoint="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
            prefix="detector.",
            type="Pretrained",
        ),
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
        test_cfg=dict(nms=dict(iou_threshold=0.7, type="nms"), score_thr=0.01),
        train_cfg=dict(assigner=dict(center_radius=2.5, type="SimOTAAssigner")),
        type="YOLOX",
    ),
    num_classes_override=1,
    post_tracking_pipeline=[
        dict(enabled=False, image_label="original_images", type="HmNumberClassifier"),
    ],
    tracker=dict(
        init_track_thr=0.7,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        motion=dict(type="KalmanFilter"),
        num_frames_retain=30,
        obj_score_thrs=dict(high=0.6, low=0.1),
        type="ByteTracker",
        weight_iou_with_det_scores=True,
    ),
    type="HmEndToEnd",
)
num_last_epochs = 10
number_classifier = dict(
    enabled=True,
    init_cfg=dict(checkpoint="pretrained/svhnc/model-65000.pth", type="Pretrained"),
    type="HmNumberClassifier",
)
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, momentum=0.9, nesterov=True, type="SGD", weight_decay=0.0005),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type="OptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=True, convert_to_iter_based=True, end=1, type="QuadraticWarmupLR"),
    dict(
        T_max=70, begin=1, by_epoch=True, convert_to_iter_based=True, end=70, eta_min=2.5e-05, type="CosineAnnealingLR"
    ),
    dict(begin=70, by_epoch=True, end=80, factor=1, type="ConstantLR"),
]
post_tracking_pipeline = [
    dict(enabled=False, image_label="original_images", type="HmNumberClassifier"),
]
resume = False
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="annotations/half-val_cocoformat.json",
        data_prefix=dict(img_path="train"),
        data_root="data/MOT17/",
        pipeline=[
            dict(
                transforms=[
                    dict(backend_args=None, type="LoadImageFromFile"),
                    dict(
                        keep_ratio=True,
                        scale=(
                            1312,
                            480,
                        ),
                        type="Resize",
                    ),
                    dict(
                        pad_val=dict(
                            img=(
                                114.0,
                                114.0,
                                114.0,
                            )
                        ),
                        size_divisor=32,
                        type="Pad",
                    ),
                    dict(type="LoadTrackAnnotations"),
                ],
                type="TransformBroadcaster",
            ),
            dict(type="PackTrackInputs"),
        ],
        test_mode=True,
        type="MOTChallengeDataset",
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="TrackImgSampler"),
)
test_evaluator = dict(
    metric=[
        "HOTA",
        "CLEAR",
        "Identity",
    ],
    postprocess_tracklet_cfg=[
        dict(max_num_frames=20, min_num_frames=5, type="InterpolateTracklets"),
    ],
    type="MOTChallengeMetric",
)
test_pipeline = [
    dict(
        transforms=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1312,
                    480,
                ),
                type="Resize",
            ),
            dict(
                pad_val=dict(
                    img=(
                        114.0,
                        114.0,
                        114.0,
                    )
                ),
                size_divisor=32,
                type="Pad",
            ),
            dict(type="LoadTrackAnnotations"),
        ],
        type="TransformBroadcaster",
    ),
    dict(type="PackTrackInputs"),
]
train_cfg = dict(max_epochs=80, type="EpochBasedTrainLoop", val_begin=70, val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        dataset=dict(
            datasets=[
                dict(
                    ann_file="annotations/half-train_cocoformat.json",
                    data_prefix=dict(img="train"),
                    data_root="data/MOT17",
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=("pedestrian",)),
                    pipeline=[
                        dict(backend_args=None, type="LoadImageFromFile"),
                        dict(type="LoadAnnotations", with_bbox=True),
                    ],
                    type="CocoDataset",
                ),
                dict(
                    ann_file="annotations/crowdhuman_train.json",
                    data_prefix=dict(img="train"),
                    data_root="data/crowdhuman",
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=("pedestrian",)),
                    pipeline=[
                        dict(backend_args=None, type="LoadImageFromFile"),
                        dict(type="LoadAnnotations", with_bbox=True),
                    ],
                    type="CocoDataset",
                ),
                dict(
                    ann_file="annotations/crowdhuman_val.json",
                    data_prefix=dict(img="val"),
                    data_root="data/crowdhuman",
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=("pedestrian",)),
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
                bbox_clip_border=False,
                img_scale=(
                    1312,
                    480,
                ),
                pad_val=114.0,
                type="Mosaic",
            ),
            dict(
                bbox_clip_border=False,
                border=(
                    -656,
                    -240,
                ),
                scaling_ratio_range=(
                    0.1,
                    2,
                ),
                type="RandomAffine",
            ),
            dict(
                bbox_clip_border=False,
                img_scale=(
                    1312,
                    480,
                ),
                pad_val=114.0,
                ratio_range=(
                    0.8,
                    1.6,
                ),
                type="MixUp",
            ),
            dict(type="YOLOXHSVRandomAug"),
            dict(prob=0.5, type="RandomFlip"),
            dict(
                clip_object_border=False,
                keep_ratio=True,
                scale=(
                    1312,
                    480,
                ),
                type="Resize",
            ),
            dict(
                pad_val=dict(
                    img=(
                        114.0,
                        114.0,
                        114.0,
                    )
                ),
                size_divisor=32,
                type="Pad",
            ),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type="FilterAnnotations",
            ),
            dict(type="PackDetInputs"),
        ],
        type="MultiImageMixDataset",
    ),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline = [
    dict(
        bbox_clip_border=False,
        img_scale=(
            1312,
            480,
        ),
        pad_val=114.0,
        type="Mosaic",
    ),
    dict(
        bbox_clip_border=False,
        border=(
            -656,
            -240,
        ),
        scaling_ratio_range=(
            0.1,
            2,
        ),
        type="RandomAffine",
    ),
    dict(
        bbox_clip_border=False,
        img_scale=(
            1312,
            480,
        ),
        pad_val=114.0,
        ratio_range=(
            0.8,
            1.6,
        ),
        type="MixUp",
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        clip_object_border=False,
        keep_ratio=True,
        scale=(
            1312,
            480,
        ),
        type="Resize",
    ),
    dict(
        pad_val=dict(
            img=(
                114.0,
                114.0,
                114.0,
            )
        ),
        size_divisor=32,
        type="Pad",
    ),
    dict(
        keep_empty=False,
        min_gt_bbox_wh=(
            1,
            1,
        ),
        type="FilterAnnotations",
    ),
    dict(type="PackDetInputs"),
]
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="annotations/half-val_cocoformat.json",
        data_prefix=dict(img_path="train"),
        data_root="data/MOT17/",
        pipeline=[
            dict(
                transforms=[
                    dict(backend_args=None, type="LoadImageFromFile"),
                    dict(
                        keep_ratio=True,
                        scale=(
                            1312,
                            480,
                        ),
                        type="Resize",
                    ),
                    dict(
                        pad_val=dict(
                            img=(
                                114.0,
                                114.0,
                                114.0,
                            )
                        ),
                        size_divisor=32,
                        type="Pad",
                    ),
                    dict(type="LoadTrackAnnotations"),
                ],
                type="TransformBroadcaster",
            ),
            dict(type="PackTrackInputs"),
        ],
        test_mode=True,
        type="MOTChallengeDataset",
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="TrackImgSampler"),
)
val_evaluator = dict(
    metric=[
        "HOTA",
        "CLEAR",
        "Identity",
    ],
    postprocess_tracklet_cfg=[
        dict(max_num_frames=20, min_num_frames=5, type="InterpolateTracklets"),
    ],
    type="MOTChallengeMetric",
)
video_out_pipeline = [
    dict(type="HmConfigureScoreboard"),
    dict(type="HmCaptureScoreboard"),
    dict(pre_clip=True, type="HmPerspectiveRotation"),
    dict(type="HmCropToVideoFrame"),
    dict(
        image_labels=[
            "img",
            "end_zone_img",
        ],
        type="HmRenderScoreboard",
    ),
    dict(enabled=False, image_label="img", type="HmUnsharpMask"),
    dict(type="HmImageOverlays", watermark_config=dict(image="images/sports_ai_watermark.png")),
]
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    name="visualizer",
    type="TrackLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
