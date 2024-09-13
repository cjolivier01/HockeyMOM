_base_ = [
    "./hm_bytetrack.py",
]

post_detection_pipeline = [
    # Prune detections which are outside of boundaries (if any)
    dict(type="BoundaryLines"),
    dict(type="SegmBoundaries"),
    # dict(type="HmExtractBoundingBoxes"),
    # dict(type="HmTopDownGetBboxCenterScale", padding=1.25),
    # dict(type="HmTopDownAffine"),
]

model = dict(
    type="HmEndToEnd",
    post_detection_pipeline=post_detection_pipeline,
    # neck=dict(type="HmNumberClassifier"),
)
