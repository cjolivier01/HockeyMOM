_base_ = [
    "./hm_crowdhuman.py",
]

#
# The overall tracking model
#
model = dict(
    type="ByteTrack",
    motion=dict(type="KalmanFilter"),
    tracker=dict(
        type="ByteTracker",
        obj_score_thrs=dict(high=0.4, low=0.1),
        init_track_thr=0.5,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30,
    ),
)

evaluation = dict(metric=["bbox", "track"], interval=100)
