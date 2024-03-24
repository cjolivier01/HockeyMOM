_base_ = [
    "./hm_bytetrack.py",
]

model = dict(
    type="HmEndToEnd",
    post_detection_pipeline = [],
)

