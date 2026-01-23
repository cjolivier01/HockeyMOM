import torch

from hmlib.aspen.plugins.jersey_koshkina_plugin import KoshkinaJerseyNumberPlugin


def _mk_pose(
    ls: tuple[float, float, float],
    rs: tuple[float, float, float],
    le: tuple[float, float, float],
    re: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    kpts = torch.zeros((17, 2), dtype=torch.float32)
    kps = torch.zeros((17,), dtype=torch.float32)
    for idx, (x, y, s) in {5: ls, 6: rs, 7: le, 8: re}.items():
        kpts[idx, 0] = x
        kpts[idx, 1] = y
        kps[idx] = s
    return kpts, kps


def should_detect_side_view_from_shoulders() -> None:
    plugin = KoshkinaJerseyNumberPlugin(
        side_view_enabled=True, side_view_shoulder_ratio_thresh=0.25
    )

    # Shoulders very close compared to bbox width -> side view.
    kpts, kps = _mk_pose(
        ls=(100, 100, 0.9),
        rs=(110, 102, 0.95),
        le=(90, 140, 0.9),
        re=(120, 140, 0.9),
    )
    side_view, side = plugin._is_side_view_pose(kpts, kps, bbox_w=200)
    assert side_view is True
    assert side in ("left", "right")

    # Wide shoulders -> not side view.
    kpts2, kps2 = _mk_pose(
        ls=(60, 100, 0.9),
        rs=(160, 100, 0.9),
        le=(50, 140, 0.9),
        re=(170, 140, 0.9),
    )
    side_view2, side2 = plugin._is_side_view_pose(kpts2, kps2, bbox_w=200)
    assert side_view2 is False
    assert side2 is None


def should_build_upper_arm_roi() -> None:
    kpts, kps = _mk_pose(
        ls=(100, 100, 0.9),
        rs=(160, 100, 0.2),
        le=(90, 150, 0.9),
        re=(0, 0, 0.0),
    )
    roi = KoshkinaJerseyNumberPlugin._upper_arm_roi_from_pose(
        kpts=kpts, kps=kps, img_w=640, img_h=360, side="left"
    )
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert 0 <= x1 < x2 <= 639
    assert 0 <= y1 < y2 <= 359
    assert (x2 - x1) >= 4
    assert (y2 - y1) >= 4
