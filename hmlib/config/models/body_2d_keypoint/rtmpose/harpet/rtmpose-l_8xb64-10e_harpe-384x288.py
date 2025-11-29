"""Pose model config: RTMPose-L trained on HARPET 384x288 for 2D keypoints."""

from hmlib.visualization import PytorchPoseLocalVisualizer

auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend="local")
base_lr = 0.004
codec = dict(
    input_size=(
        288,
        384,
    ),
    normalize=False,
    sigma=(
        6.0,
        6.93,
    ),
    simcc_split_ratio=2.0,
    type="SimCCLabel",
    use_dark=False,
)
custom_hooks = [
    dict(type="SyncBuffersHook"),
]
data_mode = "topdown"
data_root = "/mnt/ripper-data2/datasets/VIP-HARPET/"
dataset_type = "HarpeDataset"
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type="loss",
        out_dir="badcase",
        type="BadCaseAnalysisHook",
    ),
    checkpoint=dict(
        interval=10, max_keep_ckpts=1, rule="greater", save_best="PCK", type="CheckpointHook"
    ),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(enable=False, type="PoseVisualizationHook"),
)
default_scope = "mmpose"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
launcher = "none"
# load_from = '/home/colivier/src/openmm/work_dirs/rtmpose_l_harpe_384x288/epoch_1000.pth'
log_level = "INFO"
log_processor = dict(by_epoch=True, num_digits=6, type="LogProcessor", window_size=50)
max_epochs = 10
model = dict(
    backbone=dict(
        _scope_="mmdet",
        act_cfg=dict(type="SiLU"),
        arch="P5",
        channel_attention=True,
        deepen_factor=1.0,
        expand_ratio=0.5,
        norm_cfg=dict(type="SyncBN"),
        out_indices=(4,),
        type="CSPNeXt",
        widen_factor=1.0,
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="PoseDataPreprocessor",
    ),
    head=dict(
        decoder=dict(
            input_size=(
                288,
                384,
            ),
            normalize=False,
            sigma=(
                6.0,
                6.93,
            ),
            simcc_split_ratio=2.0,
            type="SimCCLabel",
            use_dark=False,
        ),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn="SiLU",
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False,
        ),
        in_channels=1024,
        in_featuremap_size=(
            9,
            12,
        ),
        input_size=(
            288,
            384,
        ),
        loss=dict(beta=10.0, label_softmax=True, type="KLDiscretLoss", use_target_weight=True),
        out_channels=18,
        simcc_split_ratio=2.0,
        type="RTMCCHead",
    ),
    test_cfg=dict(flip_test=True),
    type="TopdownPoseEstimator",
)
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type="AdamW", weight_decay=0.05),
    paramwise_cfg=dict(bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type="OptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=1e-05, type="LinearLR"),
    dict(
        T_max=5,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        eta_min=0.0002,
        type="CosineAnnealingLR",
    ),
]


dataset_info = dict(
    dataset_name="harpe18",
    paper_info=dict(
        author="VIP-HARPE Team",
        title="VIP-HARPET Dataset",
        container="Internal",
        year="2018",
        homepage="",
    ),
    keypoint_info={
        0: dict(name="j0", id=0, color=[51, 153, 255], type="upper", swap="j5"),
        1: dict(name="j1", id=1, color=[51, 153, 255], type="upper", swap="j4"),
        2: dict(name="j2", id=2, color=[51, 153, 255], type="upper", swap="j3"),
        3: dict(name="j3", id=3, color=[51, 153, 255], type="upper", swap="j2"),
        4: dict(name="j4", id=4, color=[0, 255, 0], type="upper", swap="j1"),
        5: dict(name="j5", id=5, color=[255, 128, 0], type="upper", swap="j0"),
        6: dict(name="j6", id=6, color=[51, 153, 255], type="upper", swap=""),
        7: dict(name="j7", id=7, color=[51, 153, 255], type="upper", swap=""),
        8: dict(name="j8", id=8, color=[51, 153, 255], type="upper", swap=""),
        9: dict(name="j9", id=9, color=[51, 153, 255], type="upper", swap=""),
        10: dict(name="j10", id=10, color=[255, 128, 0], type="lower", swap="j15"),
        11: dict(name="j11", id=11, color=[255, 128, 0], type="lower", swap="j14"),
        12: dict(name="j12", id=12, color=[255, 128, 0], type="lower", swap="j13"),
        13: dict(name="j13", id=13, color=[0, 255, 0], type="lower", swap="j12"),
        14: dict(name="j14", id=14, color=[0, 255, 0], type="lower", swap="j11"),
        15: dict(name="j15", id=15, color=[0, 255, 0], type="lower", swap="j10"),
        16: dict(name="j16", id=16, color=[51, 153, 255], type="lower", swap=""),
        17: dict(name="j17", id=17, color=[51, 153, 255], type="lower", swap=""),
    },
    skeleton_info={
        0: dict(link=("j0", "j1"), id=0, color=[51, 153, 255]),
        1: dict(link=("j1", "j2"), id=1, color=[51, 153, 255]),
        2: dict(link=("j2", "j3"), id=2, color=[51, 153, 255]),
        3: dict(link=("j3", "j4"), id=3, color=[0, 255, 0]),
        4: dict(link=("j4", "j5"), id=4, color=[0, 255, 0]),
        5: dict(link=("j6", "j7"), id=5, color=[51, 153, 255]),
        6: dict(link=("j7", "j8"), id=6, color=[51, 153, 255]),
        7: dict(link=("j8", "j9"), id=7, color=[51, 153, 255]),
        8: dict(link=("j10", "j11"), id=8, color=[255, 128, 0]),
        9: dict(link=("j11", "j12"), id=9, color=[255, 128, 0]),
        10: dict(link=("j12", "j13"), id=10, color=[255, 128, 0]),
        11: dict(link=("j13", "j14"), id=11, color=[0, 255, 0]),
        12: dict(link=("j14", "j15"), id=12, color=[0, 255, 0]),
        13: dict(link=("j15", "j16"), id=13, color=[0, 255, 0]),
        # 14: dict(link=("j15", "j16"), id=13, color=[0, 255, 0]),
    },
    joint_weights=[1.0] * 18,
    sigmas=[0.025] * 18,
)


randomness = dict(seed=21)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file="annot_valid.h5",
        data_mode="topdown",
        data_prefix=dict(img="images_valid"),
        data_root="/mnt/ripper-data2/datasets/VIP-HARPET/",
        metainfo=dataset_info,
        pipeline=[
            dict(backend_args=dict(backend="local"), type="LoadImage"),
            dict(type="GetBBoxCenterScale"),
            dict(
                input_size=(
                    288,
                    384,
                ),
                type="TopdownAffine",
            ),
            dict(type="PackPoseInputs"),
        ],
        test_mode=True,
        type="HarpeDataset",
    ),
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type="DefaultSampler"),
)
test_evaluator = [
    dict(norm_item="bbox", thr=0.2, type="PCKAccuracy"),
    dict(type="AUC"),
    dict(type="EPE"),
]
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file="annot_train.h5",
        data_mode="topdown",
        data_prefix=dict(img="images_train"),
        data_root="/mnt/ripper-data2/datasets/VIP-HARPET/",
        metainfo=dataset_info,
        pipeline=[
            dict(backend_args=dict(backend="local"), type="LoadImage"),
            dict(type="GetBBoxCenterScale"),
            dict(direction="horizontal", type="RandomFlip"),
            dict(type="RandomHalfBody"),
            dict(
                rotate_factor=80,
                scale_factor=[
                    0.6,
                    1.4,
                ],
                type="RandomBBoxTransform",
            ),
            dict(
                input_size=(
                    288,
                    384,
                ),
                type="TopdownAffine",
            ),
            dict(type="mmdet.YOLOXHSVRandomAug"),
            dict(
                encoder=dict(
                    input_size=(
                        288,
                        384,
                    ),
                    normalize=False,
                    sigma=(
                        6.0,
                        6.93,
                    ),
                    simcc_split_ratio=2.0,
                    type="SimCCLabel",
                    use_dark=False,
                ),
                type="GenerateTarget",
            ),
            dict(type="PackPoseInputs"),
        ],
        test_mode=False,
        type="HarpeDataset",
    ),
    num_workers=6,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline = [
    dict(backend_args=dict(backend="local"), type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(direction="horizontal", type="RandomFlip"),
    dict(type="RandomHalfBody"),
    dict(
        rotate_factor=80,
        scale_factor=[
            0.6,
            1.4,
        ],
        type="RandomBBoxTransform",
    ),
    dict(
        input_size=(
            288,
            384,
        ),
        type="TopdownAffine",
    ),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        encoder=dict(
            input_size=(
                288,
                384,
            ),
            normalize=False,
            sigma=(
                6.0,
                6.93,
            ),
            simcc_split_ratio=2.0,
            type="SimCCLabel",
            use_dark=False,
        ),
        type="GenerateTarget",
    ),
    dict(type="PackPoseInputs"),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file="annot_valid.h5",
        data_mode="topdown",
        data_prefix=dict(img="images_valid"),
        data_root="/mnt/ripper-data2/datasets/VIP-HARPET/",
        metainfo=dataset_info,
        pipeline=[
            dict(backend_args=dict(backend="local"), type="LoadImage"),
            dict(type="GetBBoxCenterScale"),
            dict(
                input_size=(
                    288,
                    384,
                ),
                type="TopdownAffine",
            ),
            dict(type="PackPoseInputs"),
        ],
        test_mode=True,
        type="HarpeDataset",
    ),
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type="DefaultSampler"),
)
val_evaluator = [
    dict(norm_item="bbox", thr=0.2, type="PCKAccuracy"),
    dict(type="AUC"),
    dict(type="EPE"),
]
val_pipeline = [
    dict(backend_args=dict(backend="local"), type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(
        input_size=(
            288,
            384,
        ),
        type="TopdownAffine",
    ),
    dict(type="PackPoseInputs"),
]
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    type=PytorchPoseLocalVisualizer,
    vis_backends=vis_backends,
    name="visualizer",
    line_width=4,
    radius=2,
)
# work_dir = '/home/colivier/src/openmm/work_dirs/rtmpose_l_harpe_384x288'
