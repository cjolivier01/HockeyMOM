import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def should_keep_ids_across_frames_on_cpu():
    from hmlib.tracking_utils.bytetrack import HmByteTrackerCuda

    tracker = HmByteTrackerCuda(device="cpu")
    frame0 = {
        "frame_id": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.tensor([[10.0, 10.0, 30.0, 40.0], [100.0, 100.0, 140.0, 160.0]], dtype=torch.float32),
        "labels": torch.tensor([1, 1], dtype=torch.long),
        "scores": torch.tensor([0.9, 0.85], dtype=torch.float32),
    }
    res0 = tracker.track(frame0)
    assert torch.equal(res0["ids"], torch.tensor([0, 1], dtype=torch.long))

    frame1 = {
        "frame_id": torch.tensor([1], dtype=torch.long),
        "bboxes": torch.tensor([[12.0, 12.0, 32.0, 42.0], [103.0, 103.0, 143.0, 163.0]], dtype=torch.float32),
        "labels": torch.tensor([1, 1], dtype=torch.long),
        "scores": torch.tensor([0.92, 0.8], dtype=torch.float32),
    }
    res1 = tracker.track(frame1)
    assert torch.equal(res1["ids"], torch.tensor([0, 1], dtype=torch.long))


def should_start_no_tracks_below_init_threshold():
    from hmlib.tracking_utils.bytetrack import HmByteTrackerCuda

    tracker = HmByteTrackerCuda(device="cpu")
    frame0 = {
        "frame_id": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.tensor([[10.0, 10.0, 30.0, 40.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
        "scores": torch.tensor([0.6], dtype=torch.float32),  # default init_track_thr=0.7
    }
    res0 = tracker.track(frame0)
    assert res0["ids"].numel() == 0
    assert res0["bboxes"].shape == (0, 4)


def should_create_new_id_when_label_changes():
    from hmlib.tracking_utils.bytetrack import HmByteTrackerCuda

    tracker = HmByteTrackerCuda(device="cpu")
    frame0 = {
        "frame_id": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.tensor([[10.0, 10.0, 30.0, 40.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
        "scores": torch.tensor([0.9], dtype=torch.float32),
    }
    res0 = tracker.track(frame0)
    assert torch.equal(res0["ids"], torch.tensor([0], dtype=torch.long))

    frame1 = {
        "frame_id": torch.tensor([1], dtype=torch.long),
        "bboxes": torch.tensor([[10.0, 10.0, 30.0, 40.0]], dtype=torch.float32),
        "labels": torch.tensor([2], dtype=torch.long),  # label mismatch => no match
        "scores": torch.tensor([0.9], dtype=torch.float32),
    }
    res1 = tracker.track(frame1)
    assert torch.equal(res1["ids"], torch.tensor([1], dtype=torch.long))


def should_pad_static_outputs():
    from hmlib.tracking_utils.bytetrack import HmByteTrackerCudaStatic

    static = HmByteTrackerCudaStatic(max_detections=8, max_tracks=8, device="cpu")
    frame0 = {
        "frame_id": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.zeros((8, 4), dtype=torch.float32),
        "labels": torch.zeros((8,), dtype=torch.long),
        "scores": torch.zeros((8,), dtype=torch.float32),
        "num_detections": torch.tensor([2], dtype=torch.long),
    }
    frame0["bboxes"][:2] = torch.tensor(
        [[10.0, 10.0, 30.0, 40.0], [100.0, 100.0, 140.0, 160.0]], dtype=torch.float32
    )
    frame0["labels"][:2] = torch.tensor([1, 1], dtype=torch.long)
    frame0["scores"][:2] = torch.tensor([0.9, 0.85], dtype=torch.float32)

    out = static.track(frame0)
    assert out["ids"].shape == (8,)
    assert out["bboxes"].shape == (8, 4)
    assert out["labels"].shape == (8,)
    assert out["scores"].shape == (8,)
    assert torch.equal(out["num_tracks"], torch.tensor([2], dtype=torch.long))
    assert torch.equal(out["num_detections"], torch.tensor([2], dtype=torch.long))

    assert torch.equal(out["ids"][:2], torch.tensor([0, 1], dtype=torch.long))
    assert torch.all(out["ids"][2:] == -1)
