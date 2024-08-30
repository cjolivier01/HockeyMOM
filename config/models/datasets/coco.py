# _base_ = [
#     "./coco_classes.py",
# ]

# dataset settings
data_root = "data/coco/"

#dataset_img_scale = (640, 640)
dataset_img_scale = (480, 1312)

train_pipeline = [
    dict(type="Mosaic", img_scale=dataset_img_scale, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-dataset_img_scale[0] // 2, -dataset_img_scale[1] // 2),
    ),
    dict(
        type="MixUp", img_scale=dataset_img_scale, ratio_range=(0.8, 1.6), pad_val=114.0
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type="Resize", img_scale=dataset_img_scale, keep_ratio=True),
    dict(
        type="Pad",
        size=dataset_img_scale,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=dataset_img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(
                type="Pad",
                size=dataset_img_scale,
                pad_val=dict(img=(114.0, 114.0, 114.0)),
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="CocoDataset",
            ann_file=data_root + "annotations/instances_train2017.json",
            img_prefix=data_root + "train2017/",
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
            filter_empty_gt=False,
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=test_pipeline,
    ),
)
