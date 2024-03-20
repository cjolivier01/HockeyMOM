_base_ = [
    "./hm_yolox_l.py",
]

img_scale = (480, 1312)  # height, width

#
# Yolox-S
#
detector_standalone_model = dict(
    input_size=img_scale,
    init_cfg=dict(
        type="Pretrained",
        checkpoint="/home/colivier/Downloads/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
    ),
)

model = dict(
    detector=detector_standalone_model,
)
