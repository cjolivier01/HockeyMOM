from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]

if torch is not None:
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample, TrackDataSample
    from mmpose.structures import PoseDataSample

    from hmlib.aspen import AspenNet
    from hmlib.aspen.plugins.base import Plugin
    from hmlib.aspen.plugins.detector_factory_plugin import _TorchDetectorWrapper
    from hmlib.aspen.plugins.ice_rink_boundaries_plugins import IceRinkSegmBoundariesPlugin
    from hmlib.aspen.plugins.pose_plugin import PosePlugin
    from hmlib.utils.cuda_graph import CudaGraphCallable
    from hmlib.utils.gpu import unwrap_tensor
else:
    InstanceData = object  # type: ignore[assignment]
    DetDataSample = object  # type: ignore[assignment]
    TrackDataSample = object  # type: ignore[assignment]
    PoseDataSample = object  # type: ignore[assignment]
    AspenNet = None  # type: ignore[assignment]
    Plugin = object  # type: ignore[assignment]
    _TorchDetectorWrapper = None  # type: ignore[assignment]
    IceRinkSegmBoundariesPlugin = None  # type: ignore[assignment]
    PosePlugin = None  # type: ignore[assignment]
    CudaGraphCallable = None  # type: ignore[assignment]
    unwrap_tensor = None  # type: ignore[assignment]

_TORCH_MODULE_BASE = torch.nn.Module if torch is not None else object
requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")
requires_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="requires CUDA"
)


class GraphableAffinePlugin(Plugin):  # type: ignore[misc]
    def __init__(self, scale: float, bias: float, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.scale = float(scale)
        self.bias = float(bias)
        self._cuda_graph_enabled = False
        self._cg: CudaGraphCallable | None = None
        self._cg_device: torch.device | None = None

    def input_keys(self):
        return {"x"}

    def output_keys(self):
        return {"x"}

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        x = context["x"]
        if self._cuda_graph_enabled and isinstance(x, torch.Tensor) and x.is_cuda:
            if self._cg is None or self._cg_device != x.device:
                self._cg = CudaGraphCallable(
                    lambda t: (t * self.scale) + self.bias,
                    (x,),
                    warmup=0,
                    name="graphable_affine",
                )
                self._cg_device = x.device
            return {"x": self._cg(x)}
        return {"x": (x * self.scale) + self.bias}


class YOLOXHead(_TORCH_MODULE_BASE):
    def __init__(self):
        super().__init__()
        self.test_cfg = {"max_per_img": 4}
        self.register_buffer(
            "_bboxes_template",
            torch.tensor([[1.0, 2.0, 5.0, 6.0]], device="cuda", dtype=torch.float32),
        )
        self.register_buffer(
            "_scores_template",
            torch.tensor([0.9], device="cuda", dtype=torch.float32),
        )
        self.register_buffer(
            "_labels_template",
            torch.tensor([1], device="cuda", dtype=torch.long),
        )

    def forward(self, feats):  # type: ignore[override]
        return (feats,), (feats,)

    def predict_by_feat(
        self,
        cls_scores,
        bbox_preds,
        objectnesses,
        batch_img_metas,
        cfg,
        rescale,
        with_nms,
    ):
        inst = InstanceData()
        inst.bboxes = self._bboxes_template
        inst.scores = self._scores_template
        inst.labels = self._labels_template
        return [inst]


class FakeDetectorModel(_TORCH_MODULE_BASE):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
        self.bbox_head = YOLOXHead()

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._param


class FakeDetectorNMS:
    def __init__(self, *args, **kwargs):
        pass

    def run_single(self, inst, meta):
        return inst

    def _run_trt(self, instances, metas):
        inst = instances[0]
        out = InstanceData()
        out.bboxes = inst.bboxes
        out.scores = inst.scores
        out.labels = inst.labels
        out.num_valid_after_nms = torch.ones_like(inst.scores[:1], dtype=torch.int32) * int(
            inst.bboxes.shape[0]
        )
        return [out]


class FakeRinkSegm:
    def __init__(self):
        self._draw = False

    @staticmethod
    def _static_outputs(det_bboxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
        max_k = 2
        kept = min(int(det_bboxes.shape[0]), max_k)
        out_b = torch.cat(
            [det_bboxes[:kept], det_bboxes.new_zeros((max_k - kept, 4))],
            dim=0,
        )
        out_l = torch.cat(
            [labels[:kept], labels.new_zeros((max_k - kept,))],
            dim=0,
        )
        out_s = torch.cat(
            [scores[:kept], scores.new_zeros((max_k - kept,))],
            dim=0,
        )
        num_valid = torch.ones_like(scores[:1], dtype=torch.int32) * kept
        return out_b, out_l, out_s, num_valid

    def prune_detections_static(self, det_bboxes, labels, scores):
        return self._static_outputs(det_bboxes, labels, scores)

    def __call__(self, payload):
        out_b, out_l, out_s, num_valid = self._static_outputs(
            payload["det_bboxes"], payload["labels"], payload["scores"]
        )
        return {
            "det_bboxes": out_b,
            "labels": out_l,
            "scores": out_s,
            "num_detections": num_valid,
        }


class FakePoseHead(_TORCH_MODULE_BASE):
    def __init__(self):
        super().__init__()
        self.decoder = SimpleNamespace(simcc_split_ratio=1.0)

    def forward(self, feats):  # type: ignore[override]
        x = feats[0]
        pred_x = torch.zeros((x.shape[0], 1, 8), device=x.device, dtype=torch.float32)
        pred_y = torch.zeros((x.shape[0], 1, 8), device=x.device, dtype=torch.float32)
        pred_x[:, 0, 2] = 1.0
        pred_y[:, 0, 3] = 1.0
        return pred_x, pred_y


class FakePoseModel(_TORCH_MODULE_BASE):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
        self.backbone = torch.nn.Identity()
        self.neck = None
        self.head = FakePoseHead()
        self.data_preprocessor = SimpleNamespace(
            mean=None,
            std=None,
            bgr_to_rgb=False,
            rgb_to_bgr=False,
        )
        self.dataset_meta = {}


class FakePoseImpl:
    def __init__(self):
        self.cfg = SimpleNamespace(data_mode="topdown")
        self.model = FakePoseModel()
        self.pipeline = lambda inst: inst
        self.collate_fn = self._collate
        self.visualizer = None

    @staticmethod
    def _collate(items):
        inputs = torch.stack([item["img"].to(dtype=torch.float32) for item in items], dim=0)
        data_samples = []
        for item in items:
            bbox = item["bbox"].reshape(1, 4).to(device=inputs.device, dtype=torch.float32)
            bbox_score = item["bbox_score"].reshape(1).to(device=inputs.device, dtype=torch.float32)
            data_samples.append(
                SimpleNamespace(
                    metainfo={
                        "hm_frame_index": int(item["hm_frame_index"]),
                        "input_center": torch.tensor([4.0, 4.0], device=inputs.device),
                        "input_scale": torch.tensor([8.0, 8.0], device=inputs.device),
                        "input_size": torch.tensor([8.0, 8.0], device=inputs.device),
                    },
                    gt_instances=InstanceData(
                        bboxes=bbox,
                        bbox_scores=bbox_score,
                    ),
                )
            )
        return {"inputs": inputs, "data_samples": data_samples}

    @staticmethod
    def forward(proc_inputs, merge_results=False, bbox_thr=-1, pose_based_nms=False):
        _ = (merge_results, bbox_thr, pose_based_nms)
        outputs = []
        for ds in proc_inputs["data_samples"]:
            sample = PoseDataSample(metainfo=dict(ds.metainfo))
            sample.gt_instances = ds.gt_instances
            sample.pred_instances = InstanceData(
                keypoints=torch.tensor([[[2.0, 3.0]]], device=ds.gt_instances.bboxes.device),
                keypoint_scores=torch.tensor([[1.0]], device=ds.gt_instances.bboxes.device),
                bboxes=ds.gt_instances.bboxes,
                bbox_scores=ds.gt_instances.bbox_scores,
            )
            outputs.append(sample)
        return outputs


class FakePoseInferencer:
    def __init__(self):
        self.inferencer = FakePoseImpl()
        self.filter_args = {}


def should_report_torch_runtime_requirement_for_cuda_graph_pipeline() -> None:
    if torch is None:
        pytest.skip("torch is unavailable in this Bazel test runtime")


@requires_cuda
def should_enable_cuda_graph_for_supported_aspen_pipeline_plugins():
    graph_cfg = {
        "pipeline": {"cuda_graph": True},
        "plugins": {
            "a": {
                "class": f"{__name__}.GraphableAffinePlugin",
                "depends": [],
                "params": {"scale": 2.0, "bias": 1.0},
            },
            "b": {
                "class": f"{__name__}.GraphableAffinePlugin",
                "depends": ["a"],
                "params": {"scale": 3.0, "bias": 4.0},
            },
        },
    }
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    net = AspenNet("cuda_graph_pipeline", graph_cfg)
    out = net({"x": x.clone()})
    expected = ((x * 2.0) + 1.0) * 3.0 + 4.0
    assert torch.allclose(out["x"], expected)
    assert net.shared["aspen_cuda_graph_enabled"] is True
    assert getattr(net.node_map["a"].module, "_cuda_graph_enabled", False) is True
    assert getattr(net.node_map["b"].module, "_cuda_graph_enabled", False) is True


@requires_cuda
def should_match_detector_outputs_with_and_without_cuda_graph(monkeypatch):
    monkeypatch.setattr(
        "hmlib.aspen.plugins.detector_factory_plugin.DetectorNMS",
        FakeDetectorNMS,
    )
    imgs = torch.zeros((1, 3, 8, 8), device="cuda", dtype=torch.float32)
    data_samples = [SimpleNamespace(metainfo={})]

    no_graph = _TorchDetectorWrapper(
        FakeDetectorModel(),
        nms_backend="trt",
        nms_test=False,
        nms_plugin="batched",
        cuda_graph=False,
    )
    with_graph = _TorchDetectorWrapper(
        FakeDetectorModel(),
        nms_backend="trt",
        nms_test=False,
        nms_plugin="batched",
        cuda_graph=True,
    )

    ref = no_graph.predict(imgs, data_samples)[0].pred_instances
    got = with_graph.predict(imgs, data_samples)[0].pred_instances
    assert torch.allclose(ref.bboxes, got.bboxes)
    assert torch.allclose(ref.scores, got.scores)
    assert torch.equal(ref.labels, got.labels)


@requires_cuda
def should_match_pose_outputs_with_and_without_cuda_graph():
    frame = DetDataSample()
    frame.pred_track_instances = InstanceData(
        bboxes=torch.tensor([[0.0, 0.0, 8.0, 8.0]], device="cuda", dtype=torch.float32),
        scores=torch.tensor([0.9], device="cuda", dtype=torch.float32),
        labels=torch.tensor([0], device="cuda", dtype=torch.long),
        instances_id=torch.tensor([1], device="cuda", dtype=torch.long),
    )
    track_data_sample = TrackDataSample(video_data_samples=[frame])
    context = {
        "pose_inferencer": FakePoseInferencer(),
        "original_images": torch.zeros((1, 3, 8, 8), device="cuda", dtype=torch.float32),
        "data_samples": track_data_sample,
        "plot_pose": False,
    }

    ref_plugin = PosePlugin(cuda_graph=False)
    ref = ref_plugin(dict(context))["pose_results"][0]["predictions"][0].pred_instances

    cg_plugin = PosePlugin(cuda_graph=True, cuda_graph_max_instances=4)
    got = cg_plugin(dict(context))["pose_results"][0]["predictions"][0].pred_instances

    assert torch.allclose(ref.keypoints, got.keypoints)
    assert torch.allclose(ref.keypoint_scores, got.keypoint_scores)
    assert torch.allclose(ref.bboxes, got.bboxes)
    assert torch.allclose(ref.bbox_scores, got.bbox_scores)


@requires_cuda
def should_match_rink_pruning_with_and_without_cuda_graph():
    def _make_context():
        det_inst = InstanceData()
        det_inst.bboxes = torch.tensor(
            [[0.0, 0.0, 4.0, 4.0], [5.0, 5.0, 8.0, 8.0], [9.0, 9.0, 12.0, 12.0]],
            device="cuda",
            dtype=torch.float32,
        )
        det_inst.labels = torch.tensor([1, 2, 3], device="cuda", dtype=torch.long)
        det_inst.scores = torch.tensor([0.8, 0.7, 0.6], device="cuda", dtype=torch.float32)
        frame = DetDataSample()
        frame.pred_instances = det_inst
        track_data_sample = TrackDataSample(video_data_samples=[frame])
        return {
            "data_samples": track_data_sample,
            "rink_profile": {"mask": True},
        }

    ref_plugin = IceRinkSegmBoundariesPlugin(
        raise_bbox_center_by_height_ratio=0.0,
        lower_bbox_bottom_by_height_ratio=0.0,
        max_detections_in_mask=2,
        cuda_graph=False,
    )
    ref_plugin._segm = FakeRinkSegm()
    ref_context = _make_context()
    ref_plugin(ref_context)
    ref_inst = ref_context["data_samples"].video_data_samples[0].pred_instances

    cg_plugin = IceRinkSegmBoundariesPlugin(
        raise_bbox_center_by_height_ratio=0.0,
        lower_bbox_bottom_by_height_ratio=0.0,
        max_detections_in_mask=2,
        cuda_graph=True,
    )
    cg_plugin._segm = FakeRinkSegm()
    cg_plugin._iter_num = 2
    cg_context = _make_context()
    cg_plugin(cg_context)
    got_inst = cg_context["data_samples"].video_data_samples[0].pred_instances

    assert torch.allclose(unwrap_tensor(ref_inst.bboxes), unwrap_tensor(got_inst.bboxes))
    assert torch.equal(unwrap_tensor(ref_inst.labels), unwrap_tensor(got_inst.labels))
    assert torch.allclose(unwrap_tensor(ref_inst.scores), unwrap_tensor(got_inst.scores))
