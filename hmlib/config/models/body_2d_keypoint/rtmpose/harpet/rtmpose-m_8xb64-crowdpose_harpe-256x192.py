from hmlib.visualization import PytorchPoseLocalVisualizer

default_scope = "mmpose"

# hooks
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=10),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=False),
    badcase=dict(
        type="BadCaseAnalysisHook",
        enable=False,
        out_dir="badcase",
        metric_type="loss",
        badcase_thr=5,
    ),
)

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type="SyncBuffersHook")
]

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# visualizer
vis_backends = [
    dict(type="LocalVisBackend"),
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(
    type=PytorchPoseLocalVisualizer,
    vis_backends=vis_backends,
    name="visualizer",
    line_width=2,
)

# logger
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level = "INFO"
load_from = None
resume = False

# file I/O backend
backend_args = dict(backend="local")

# training/validation/testing progress
train_cfg = dict(by_epoch=True)
val_cfg = dict()
test_cfg = dict()


# runtime
max_epochs = 100
base_lr = 5e-4

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings (match CrowdPose 256x192 recipe)
codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# model settings (RTMPose-M)
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=768,
        out_channels=18,  # HARPE has 18 joints
        input_size=codec["input_size"],
        in_featuremap_size=tuple([s // 32 for s in codec["input_size"]]),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
    ),
)

# initialize from RTMPose-M SimCC pretrained checkpoint on CrowdPose
# load_from = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'

# base dataset settings
dataset_type = "HarpeDataset"
data_mode = "topdown"
data_root = "/mnt/ripper-data/datasets/VIP-HARPET/"

backend_args = dict(backend="local")

# pipelines
train_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform", scale_factor=[0.3, 1.4], rotate_factor=80),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    # Albumentations (optional). If not installed, skip this block.
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(from_file="configs/_base_/datasets/harpe18.py"),
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annot_train.h5",
        data_prefix=dict(img="images_train"),
        pipeline=train_pipeline,
        test_mode=False,
    ),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(from_file="configs/_base_/datasets/harpe18.py"),
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annot_valid.h5",
        data_prefix=dict(img="images_valid"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(checkpoint=dict(save_best="PCK", rule="greater", max_keep_ckpts=1))

# evaluators (PCK and AUC/EPE optional)
val_evaluator = [
    dict(type="PCKAccuracy", thr=0.2, norm_item="bbox"),
    dict(type="AUC"),
    dict(type="EPE"),
]
test_evaluator = val_evaluator
