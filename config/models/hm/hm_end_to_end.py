_base_ = [
    "./hm_bytetrack.py",
]

post_detection_pipeline = [
    # Prune detections which are outside of boundaries (if any)
    dict(type="BoundaryLines"),
    dict(type="IceRinkSegmBoundaries"),
    # dict(type="HmExtractBoundingBoxes"),
    # dict(type="HmTopDownGetBboxCenterScale", padding=1.25),
    # dict(type="HmTopDownAffine"),
]

number_classifier = dict(
    type="HmNumberClassifier",
    enabled=True,
    init_cfg=dict(
        type="Pretrained",
        checkpoint="pretrained/svhnc/model-65000.pth",
    ),
)

model = dict(
    type="HmEndToEnd",
    post_detection_pipeline=post_detection_pipeline,
    num_classes_override=1,
    neck=number_classifier,
)
