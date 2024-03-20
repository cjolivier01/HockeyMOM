_base_ = [
    "../datasets/coco.py",
    '../../mmdetection/configs/_base_/schedules/schedule_1x.py', 
    '..//default_hm_runtime.py',
    "./hm_yolox_l.py",
]

detector_standalone_model = dict(
    input_size=(480, 1312),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/pretrain/third_party/darknet53-a628ea1b.pth",
        #checkpoint="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
    ),
    # bbox_head=dict(num_classes=81),
)

data = dict(
    # A100 40GB, yolox-l
    samples_per_gpu=3,
    #samples_per_gpu=32,
    # A100 40GB, yolox-s
    
    # A16
    #samples_per_gpu=15,
    # P100 16GB
    #samples_per_gpu=4,
    # samples_per_gpu=24,
    workers_per_gpu=2,
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

total_epochs = 300
num_last_epochs = 10
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
evaluation = dict(interval=10, metric="bbox")
