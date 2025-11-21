"""Dataset definition for CrowdHuman training used by hm2 detection models."""

_base_ = [
    "./coco_classes.py",
]

ch_img_scale = (640, 640)
#ch_img_scale = (400, 1300)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Mosaic",
        img_scale=ch_img_scale,
        pad_val=114.0,
        bbox_clip_border=False,
    ),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-ch_img_scale[0] // 2, -ch_img_scale[1] // 2),
        bbox_clip_border=False,
    ),
    dict(
        type="MixUp",
        img_scale=ch_img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False,
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="Resize", img_scale=ch_img_scale, keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=ch_img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    workers_per_gpu=2,
    train=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="CocoDataset",
            ann_file=[
                "data/crowdhuman/annotations/train.json",
                "data/crowdhuman/annotations/val.json",
            ],
            img_prefix=[
                "data/crowdhuman/train/Images/",
                "data/crowdhuman/val/Images/",
            ],
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="CocoDataset",
            ann_file=[
                "data/crowdhuman/annotations/val.json",
            ],
            img_prefix=[
                "data/crowdhuman/val/Images/",
            ],
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
        ),
        pipeline=test_pipeline,
    ),
    test=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="CocoDataset",
            ann_file=[
                "data/crowdhuman/annotations/val.json",
            ],
            img_prefix=[
                "data/crowdhuman/val/Images/",
            ],
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
        ),
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=10, metric="bbox")
