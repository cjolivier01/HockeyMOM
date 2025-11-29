"""Composite dataset configuration mixing CrowdHuman and MOT17 annotations."""

_base_ = [
    "./crowdhuman.py",
]

data = dict(
    workers_per_gpu=3,
    train=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            ann_file=[
                "data/MOT17/annotations/half-train_cocoformat.json",
                "data/crowdhuman/annotations/train.json",
                "data/crowdhuman/annotations/val.json",
            ],
            img_prefix=[
                "data/MOT17/train",
                "data/crowdhuman/train/Images/",
                "data/crowdhuman/val/Images/",
            ],
        ),
    ),
    val=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            ann_file=[
                "data/MOT17/annotations/half-val_cocoformat.json",
                "data/crowdhuman/annotations/val.json",
            ],
            img_prefix=[
                "data/MOT17/val",
                "data/crowdhuman/val/Images/",
            ],
        ),
    ),
    test=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            ann_file=[
                "data/crowdhuman/annotations/val.json",
            ],
            img_prefix=[
                "data/crowdhuman/val/Images/",
            ],
        ),
    ),
)
