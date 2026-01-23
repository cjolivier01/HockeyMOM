import torch

from hmlib.jersey.jersey_tracker import JerseyTracker
from hmlib.jersey.number_classifier import TrackJerseyInfo


def should_stabilize_and_switch_numbers() -> None:
    tracker = JerseyTracker(
        show=True,
        evidence_decay=0.95,
        min_display_evidence=1.0,
        switch_ratio=1.4,
        anchor_ema_alpha=0.5,
    )

    # Track 1: establish #12 with strong evidence.
    tracker.observe_tracking_id_number_info(0, TrackJerseyInfo(tracking_id=1, number=12, score=0.9))
    tracker.observe_tracking_id_number_info(1, TrackJerseyInfo(tracking_id=1, number=12, score=0.9))

    # Low-confidence conflicting evidence should not flip the display.
    tracker.observe_tracking_id_number_info(2, TrackJerseyInfo(tracking_id=1, number=7, score=0.1))
    assert tracker._tracking_id_jersey[1].current_number == 12

    # Sustained stronger evidence should switch.
    tracker.observe_tracking_id_number_info(3, TrackJerseyInfo(tracking_id=1, number=7, score=1.0))
    tracker.observe_tracking_id_number_info(4, TrackJerseyInfo(tracking_id=1, number=7, score=1.0))
    tracker.observe_tracking_id_number_info(5, TrackJerseyInfo(tracking_id=1, number=7, score=1.0))
    assert tracker._tracking_id_jersey[1].current_number == 7

    # Drawing should not raise and should keep tensor shape.
    image = torch.zeros((3, 240, 320), dtype=torch.uint8)
    tracking_ids = torch.tensor([1], dtype=torch.int64)
    bboxes_tlwh = torch.tensor([[50, 60, 80, 120]], dtype=torch.float32)
    out = tracker.draw(image=image, tracking_ids=tracking_ids, bboxes=bboxes_tlwh)
    assert isinstance(out, torch.Tensor)
    assert out.shape == image.shape
