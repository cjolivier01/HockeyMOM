import os
import sys
import tempfile


def should_detection_dataframe_roundtrip():
    import torch
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame

    with tempfile.TemporaryDirectory(prefix="df_det_") as tmp:
        path = os.path.join(tmp, "det.csv")
        df = DetectionDataFrame(output_file=path, input_batch_size=1)
        inst = InstanceData()
        inst.scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        inst.labels = torch.tensor([1, 2], dtype=torch.long)
        inst.bboxes = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32)
        ds = DetDataSample()
        ds.pred_instances = inst
        for f in range(1, 4):
            df.add_frame_sample(f, ds)
        df.flush()

        rd = DetectionDataFrame(input_file=path, input_batch_size=1)
        # single frame
        s1 = rd.get_sample_by_frame(1)
        assert s1 is not None and hasattr(s1, "pred_instances")
        assert s1.pred_instances.bboxes.shape[-1] == 4
        # range
        many = rd.get_samples()
        assert len(many) == 3
        assert all(hasattr(x, "pred_instances") for x in many)


def should_tracking_dataframe_roundtrip():
    import torch
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample, TrackDataSample
    from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame

    with tempfile.TemporaryDirectory(prefix="df_trk_") as tmp:
        path = os.path.join(tmp, "trk.csv")
        df = TrackingDataFrame(output_file=path, input_batch_size=1)
        for f in range(1, 4):
            inst = InstanceData()
            inst.instances_id = torch.tensor([1], dtype=torch.long)
            inst.bboxes = torch.tensor([[10.0 + f, 20.0, 40.0, 60.0]], dtype=torch.float32)
            inst.scores = torch.tensor([0.95], dtype=torch.float32)
            inst.labels = torch.tensor([0], dtype=torch.long)
            img = DetDataSample()
            img.pred_track_instances = inst
            df.add_frame_sample(f, img)
        df.flush()

        rd = TrackingDataFrame(input_file=path, input_batch_size=1)
        # single frame (length-1 TrackDataSample)
        one = rd.get_sample_by_frame(2)
        assert isinstance(one, TrackDataSample) and len(one) == 1
        # range (multi-frame TrackDataSample)
        clip = rd.get_samples()
        assert isinstance(clip, TrackDataSample) and len(clip) == 3


def should_pose_dataframe_roundtrip():
    import torch
    from mmengine.structures import InstanceData
    from mmpose.structures import PoseDataSample
    from hmlib.tracking_utils.pose_dataframe import PoseDataFrame

    with tempfile.TemporaryDirectory(prefix="df_pose_") as tmp:
        path = os.path.join(tmp, "pose.csv")
        df = PoseDataFrame(output_file=path, input_batch_size=1)
        pinst = InstanceData()
        pinst.keypoints = torch.zeros((1, 17, 2), dtype=torch.float32)
        pinst.keypoint_scores = torch.ones((1, 17), dtype=torch.float32)
        pds = PoseDataSample()
        pds.pred_instances = pinst
        for f in range(1, 3):
            df.add_frame_sample(f, pds)
        df.flush()

        rd = PoseDataFrame(input_file=path, input_batch_size=1)
        s1 = rd.get_sample_by_frame(1)
        assert isinstance(s1, PoseDataSample)
        allp = rd.get_samples()
        assert len(allp) == 2 and all(hasattr(p, "pred_instances") for p in allp)


def should_action_dataframe_roundtrip():
    from hmlib.tracking_utils.action_dataframe import ActionDataFrame

    with tempfile.TemporaryDirectory(prefix="df_act_") as tmp:
        path = os.path.join(tmp, "actions.csv")
        df = ActionDataFrame(output_file=path, input_batch_size=1)
        for f in range(1, 4):
            df.add_frame_sample(f, [dict(tracking_id=1, label="idle", label_index=0, score=0.9)])
        df.flush()

        rd = ActionDataFrame(input_file=path, input_batch_size=1)
        s1 = rd.get_sample_by_frame(1)
        assert isinstance(s1, list) and len(s1) == 1
        all_act = rd.get_samples()
        assert len(all_act) == 3 and all(isinstance(a, list) for a in all_act)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
