_base_ = [
    "./hm_yolox_s.py",
]

input_size = (480, 1312)

#
# Yolox-S
#
detector_standalone_model = dict(
    input_size=input_size,  # height, width
)

model = dict(
    detector=detector_standalone_model,
)
