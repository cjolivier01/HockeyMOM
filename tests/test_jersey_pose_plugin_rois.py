from types import SimpleNamespace


def _pose_inst_one_person(
    ls: tuple[float, float, float],
    rs: tuple[float, float, float],
    lh: tuple[float, float, float],
    rh: tuple[float, float, float],
    le: tuple[float, float, float],
    re: tuple[float, float, float],
) -> SimpleNamespace:
    import torch

    kpts = torch.zeros((1, 17, 2), dtype=torch.float32)
    kps = torch.zeros((1, 17), dtype=torch.float32)
    for idx, (x, y, s) in {5: ls, 6: rs, 11: lh, 12: rh, 7: le, 8: re}.items():
        kpts[0, idx, 0] = x
        kpts[0, idx, 1] = y
        kps[0, idx] = s
    return SimpleNamespace(keypoints=kpts, keypoint_scores=kps)


def should_build_pose_torso_roi_when_confident() -> None:
    import torch

    from hmlib.aspen.plugins.jersey_pose_plugin import JerseyNumberFromPosePlugin

    plugin = JerseyNumberFromPosePlugin(roi_mode="pose", side_view_enabled=False)
    bboxes = torch.tensor([[50.0, 40.0, 150.0, 240.0]], dtype=torch.float32)
    pose = _pose_inst_one_person(
        ls=(70, 80, 0.9),
        rs=(120, 82, 0.9),
        lh=(75, 170, 0.9),
        rh=(115, 170, 0.9),
        le=(65, 120, 0.9),
        re=(125, 120, 0.9),
    )
    rois = plugin._build_rois_for_frame(bboxes_xyxy=bboxes, pose_inst=pose, img_w=640, img_h=360)
    assert len(rois) == 1
    roi = rois[0].roi
    assert int(roi[0]) >= 0 and int(roi[2]) <= 639
    assert int(roi[1]) >= 0 and int(roi[3]) <= 359
    assert int(roi[2] - roi[0]) > 0
    assert int(roi[3] - roi[1]) > 0


def should_add_sleeve_roi_when_side_on() -> None:
    import torch

    from hmlib.aspen.plugins.jersey_pose_plugin import JerseyNumberFromPosePlugin

    plugin = JerseyNumberFromPosePlugin(
        roi_mode="pose",
        side_view_enabled=True,
        side_view_shoulder_ratio_thresh=0.25,
        side_view_vote_scale=1.5,
    )
    bboxes = torch.tensor([[50.0, 40.0, 250.0, 260.0]], dtype=torch.float32)
    # Shoulders close relative to bbox width -> side view -> add sleeve ROI.
    pose = _pose_inst_one_person(
        ls=(120, 90, 0.95),
        rs=(132, 92, 0.9),
        lh=(115, 190, 0.9),
        rh=(138, 190, 0.9),
        le=(105, 150, 0.9),
        re=(0, 0, 0.0),
    )
    rois = plugin._build_rois_for_frame(bboxes_xyxy=bboxes, pose_inst=pose, img_w=640, img_h=360)
    assert len(rois) == 2
    assert rois[0].vote_scale == 1.0
    assert rois[1].vote_scale == 1.5


def should_filter_small_and_limit_tracks_for_performance() -> None:
    import torch

    from hmlib.aspen.plugins.jersey_pose_plugin import JerseyNumberFromPosePlugin

    plugin = JerseyNumberFromPosePlugin(
        roi_mode="bbox",
        max_tracks_per_frame=2,
        min_track_height=100,
        frame_stride=1,
    )
    # 4 tracks: heights = [50, 120, 220, 140] -> keep >=100 => [120,220,140] then top2 => [220,140]
    bboxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 60.0],  # h=50 (drop)
            [10.0, 10.0, 50.0, 130.0],  # h=120
            [10.0, 10.0, 50.0, 230.0],  # h=220
            [10.0, 10.0, 50.0, 150.0],  # h=140
        ],
        dtype=torch.float32,
    )
    # Simulate what forward() passes into _build_rois_for_frame after filtering:
    heights = (bboxes[:, 3] - bboxes[:, 1]).to(torch.float32)
    keep = heights >= float(plugin._min_track_height)
    b2 = bboxes[keep]
    assert b2.shape[0] == 3
    heights2 = (b2[:, 3] - b2[:, 1]).to(torch.float32)
    _, idx = torch.topk(heights2, k=int(plugin._max_tracks_per_frame))
    b3 = b2[idx]
    rois = plugin._build_rois_for_frame(bboxes_xyxy=b3, pose_inst=None, img_w=640, img_h=360)
    assert len(rois) == 2
