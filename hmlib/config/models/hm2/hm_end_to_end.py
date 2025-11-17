"""Top-level hm2 config for the HmEndToEnd model and scoreboard overlays."""

_base_ = [
    "./hm_bytetrack.py",
]

post_tracking_pipeline = [
    dict(type="HmNumberClassifier", image_label="original_images", enabled=False),
]

number_classifier = dict(
    type="HmNumberClassifier",
    enabled=True,
    init_cfg=dict(
        type="Pretrained",
        checkpoint="pretrained/svhnc/model-65000.pth",
    ),
)

video_out_pipeline = [
    dict(type="HmConfigureScoreboard"),
    dict(type="HmCaptureScoreboard"),
    dict(
        type="HmPerspectiveRotation",
        pre_clip=True,
    ),
    dict(type="HmCropToVideoFrame"),
    dict(type="HmRenderScoreboard", image_labels=["img", "end_zone_img"]),
    dict(type="HmUnsharpMask", enabled=False, image_label="img"),
    dict(
        type="HmImageOverlays",
        watermark_config=dict(
            image="images/sports_ai_watermark.png",
        ),
    ),
]

model = dict(
    type="HmEndToEnd",
    post_tracking_pipeline=post_tracking_pipeline,
    num_classes_override=1,
    dataset_meta=dict(classes=None),  # stop a warning
)
