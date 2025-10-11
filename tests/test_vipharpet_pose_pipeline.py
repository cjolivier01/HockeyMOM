import os
import os.path as osp
import subprocess
import sys

import mmengine
import numpy as np


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
MMACTION_ROOT = osp.join(REPO_ROOT, 'openmm', 'mmaction2')
DATA_ROOT = osp.join(MMACTION_ROOT, 'data', 'skeleton')
CONFIG = osp.join(
    MMACTION_ROOT,
    'configs', 'skeleton', 'stgcn', 'vipharpet_stgcn_openpose_3f.py',
)


def should_prepare_vipharpet_pose_generates_pkls():
    # Ensure conversion can run and pkls exist
    script = osp.join(REPO_ROOT, 'scripts', 'prepare_vip_harpet_pose.py')
    assert osp.isfile(script)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{MMACTION_ROOT}:{env.get('PYTHONPATH','')}"
    subprocess.check_call([sys.executable, script, '--out-dir', DATA_ROOT], env=env)

    for split in ['train', 'val', 'test']:
        out = osp.join(DATA_ROOT, f'vipharpet_{split}.pkl')
        assert osp.isfile(out), f"Missing {out}"
        ann = mmengine.load(out)
        assert isinstance(ann, list) and len(ann) > 0
        # Validate required keys in first sample
        sample = ann[0]
        for k in ['frame_dir', 'label', 'keypoint', 'keypoint_score', 'total_frames', 'img_shape']:
            assert k in sample
        kp = sample['keypoint']
        assert isinstance(kp, np.ndarray) and kp.ndim == 4  # (M,T,V,2)


def should_load_training_config():
    from mmengine.config import Config
    cfg = Config.fromfile(CONFIG)
    # sanity checks
    assert cfg.model.type == 'RecognizerGCN'
    assert cfg.train_dataloader['dataset']['type'] == 'PoseDataset'
    assert 'UniformSampleFrames' in [t['type'] for t in cfg.train_pipeline]


def should_build_and_forward_minimal_model():
    # Load one sample and run a minimal model forward to catch shape issues
    pkls = [osp.join(DATA_ROOT, 'vipharpet_train.pkl')]
    data = mmengine.load(pkls[0])
    assert len(data) > 0
    sample = data[0]
    import torch
    # ensure local mmaction2 is importable
    if MMACTION_ROOT not in sys.path:
        sys.path.insert(0, MMACTION_ROOT)
    from mmaction.models import STGCN
    # STGCN expects (N, M, T, V, C)
    x = torch.from_numpy(sample['keypoint']).unsqueeze(0)  # (1,M,T,V,2)
    # pack with score to in_channels=3
    score = torch.from_numpy(sample['keypoint_score']).unsqueeze(0).unsqueeze(-1)  # (1,M,T,V,1)
    x = torch.cat([x, score], dim=-1)  # (1,M,T,V,3)
    model = STGCN(graph_cfg=dict(layout='openpose', mode='stgcn_spatial'), in_channels=3)
    with torch.no_grad():
        y = model(x)
    assert y.shape[0] == 1


def should_train_and_test_entrypoints_smoke():
    # Run 1 epoch training on a tiny dataset via cfg override
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{MMACTION_ROOT}:{env.get('PYTHONPATH','')}"
    work = osp.join(REPO_ROOT, 'work_dirs', 'vipharpet_stgcn_openpose_3f_ci')
    # Train for 1 epoch, small batch, using AMP to match runtime
    subprocess.check_call([
        sys.executable,
        osp.join(MMACTION_ROOT, 'tools', 'train.py'),
        CONFIG,
        '--work-dir', work,
        '--amp',
        '--cfg-options',
        'train_cfg.max_epochs=1',
        'train_dataloader.batch_size=8',
        'val_dataloader.batch_size=8',
        'test_dataloader.batch_size=8',
    ], env=env)

    # Evaluate with the latest checkpoint
    ckpt = None
    for f in sorted(os.listdir(work)):
        if f.endswith('.pth'):
            ckpt = osp.join(work, f)
    assert ckpt and osp.isfile(ckpt)
    subprocess.check_call([
        sys.executable,
        osp.join(MMACTION_ROOT, 'tools', 'test.py'),
        CONFIG,
        ckpt,
        '--work-dir', osp.join(work, 'test_results_ci'),
    ], env=env)
