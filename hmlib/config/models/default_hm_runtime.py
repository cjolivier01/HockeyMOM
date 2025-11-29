"""Default runtime settings (logging, dist, LR scaling) for hm2 configs."""

import os

import torch

checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=10,
    #interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')

log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
node_count = int(os.environ.get("SLURM_NNODES", "1"))
gpus_per_node = torch.cuda.device_count()
auto_scale_lr = dict(enable=True, base_batch_size=16 * node_count * gpus_per_node)
