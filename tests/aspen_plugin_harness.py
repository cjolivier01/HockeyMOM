from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Optional

import torch
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample, TrackDataSample

try:
    from mmpose.structures import PoseDataSample
except Exception:  # pragma: no cover - optional dependency
    PoseDataSample = None  # type: ignore[assignment]


PLUGIN_DISCOVERY_EXCLUDES = {
    "Plugin",
    "SavePluginBase",
    "RecordPlugin",
    "BarrierPlugin",
    "KeyPlugin",
    "BadOutputPlugin",
    "FlakyOutputPlugin",
}


def discover_production_plugin_classes(repo_root: Path) -> set[str]:
    plugin_paths = list((repo_root / "hmlib" / "aspen" / "plugins").glob("*.py"))
    plugin_paths.append(repo_root / "hmlib" / "camera" / "apply_camera_plugin.py")
    discovered: set[str] = set()
    for path in plugin_paths:
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).with_suffix("")
        module_name = ".".join(rel.parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.endswith("Plugin"):
                continue
            if node.name in PLUGIN_DISCOVERY_EXCLUDES:
                continue
            discovered.add(f"{module_name}.{node.name}")
    return discovered


def install_fake_module(monkeypatch, module_name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def make_instance_data(
    *,
    bboxes: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    instances_id: Optional[torch.Tensor] = None,
    keypoints: Optional[torch.Tensor] = None,
    keypoint_scores: Optional[torch.Tensor] = None,
    reid_features: Optional[torch.Tensor] = None,
    metainfo: Optional[dict[str, Any]] = None,
) -> InstanceData:
    inst = InstanceData()
    inst.bboxes = bboxes if bboxes is not None else torch.empty((0, 4), dtype=torch.float32)
    inst.scores = (
        scores
        if scores is not None
        else torch.empty((int(inst.bboxes.shape[0]),), dtype=torch.float32)
    )
    inst.labels = (
        labels
        if labels is not None
        else torch.zeros((int(inst.bboxes.shape[0]),), dtype=torch.long)
    )
    if instances_id is not None:
        inst.instances_id = instances_id
    if keypoints is not None:
        inst.keypoints = keypoints
    if keypoint_scores is not None:
        inst.keypoint_scores = keypoint_scores
    if reid_features is not None:
        inst.reid_features = reid_features
    if metainfo:
        inst.set_metainfo(metainfo)
    return inst


def make_det_sample(
    *,
    frame_id: int,
    ori_shape: tuple[int, int] = (32, 48),
    pred_instances: Optional[InstanceData] = None,
    pred_track_instances: Optional[InstanceData] = None,
) -> DetDataSample:
    sample = DetDataSample()
    sample.set_metainfo({"img_id": frame_id, "frame_id": frame_id, "ori_shape": ori_shape})
    if pred_instances is not None:
        sample.pred_instances = pred_instances
    if pred_track_instances is not None:
        sample.pred_track_instances = pred_track_instances
    return sample


def make_track_data_sample(
    *,
    num_frames: int = 1,
    ori_shape: tuple[int, int] = (32, 48),
    pred_instances: Optional[list[Optional[InstanceData]]] = None,
    pred_track_instances: Optional[list[Optional[InstanceData]]] = None,
) -> TrackDataSample:
    frames = []
    for index in range(num_frames):
        det_inst = pred_instances[index] if pred_instances is not None else None
        track_inst = pred_track_instances[index] if pred_track_instances is not None else None
        frames.append(
            make_det_sample(
                frame_id=index,
                ori_shape=ori_shape,
                pred_instances=det_inst,
                pred_track_instances=track_inst,
            )
        )
    return TrackDataSample(video_data_samples=frames)


def make_pose_data_sample(
    *,
    keypoints: torch.Tensor,
    keypoint_scores: torch.Tensor,
    bboxes: Optional[torch.Tensor] = None,
    bbox_scores: Optional[torch.Tensor] = None,
) -> Any:
    if PoseDataSample is None:  # pragma: no cover - mmpose is present in this repo env
        raise RuntimeError("PoseDataSample is unavailable")
    pose_sample = PoseDataSample()
    pose_sample.pred_instances = make_instance_data(
        bboxes=bboxes,
        scores=bbox_scores,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
    )
    return pose_sample


def make_pose_results(
    *,
    num_frames: int,
    num_people: int = 1,
    keypoint_count: int = 17,
) -> list[dict[str, Any]]:
    pose_results: list[dict[str, Any]] = []
    for _ in range(num_frames):
        keypoints = torch.arange(num_people * keypoint_count * 2, dtype=torch.float32).reshape(
            num_people, keypoint_count, 2
        )
        keypoint_scores = torch.ones((num_people, keypoint_count), dtype=torch.float32)
        bboxes = torch.tensor([[1.0, 2.0, 10.0, 12.0]]).repeat(num_people, 1)
        bbox_scores = torch.full((num_people,), 0.9, dtype=torch.float32)
        pose_results.append(
            {
                "predictions": [
                    make_pose_data_sample(
                        keypoints=keypoints,
                        keypoint_scores=keypoint_scores,
                        bboxes=bboxes,
                        bbox_scores=bbox_scores,
                    )
                ]
            }
        )
    return pose_results


class FakeVideoOutput(torch.nn.Module):
    instances: list["FakeVideoOutput"] = []

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.kwargs = kwargs
        self.prepare_calls: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.stopped = False
        self.to_device = None
        self.cuda_graph_enabled = False
        FakeVideoOutput.instances.append(self)

    def to(self, device: Any):  # type: ignore[override]
        self.to_device = device
        return self

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self.cuda_graph_enabled = bool(enabled)
        return True

    def prepare_results(self, context: dict[str, Any]) -> dict[str, Any]:
        out = dict(context)
        self.prepare_calls.append(dict(out))
        out["video_out_prepared"] = True
        return out

    def write_prepared_results(self, context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(context))
        return context

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        prepared = self.prepare_results(context)
        return self.write_prepared_results(prepared)

    def stop(self) -> None:
        self.stopped = True


class FakeVideoOutputPreparer(torch.nn.Module):
    instances: list["FakeVideoOutputPreparer"] = []

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.kwargs = kwargs
        self.prepare_calls: list[dict[str, Any]] = []
        self.to_device = None
        self.cuda_graph_enabled = False
        FakeVideoOutputPreparer.instances.append(self)

    def to(self, device: Any):  # type: ignore[override]
        self.to_device = device
        return self

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self.cuda_graph_enabled = bool(enabled)
        return True

    def prepare_results(self, context: dict[str, Any]) -> dict[str, Any]:
        out = dict(context)
        self.prepare_calls.append(dict(out))
        out["video_out_prepared"] = True
        return out


class FakeCompose:
    def __init__(self, transforms: Optional[Iterable[Any]] = None):
        self.transforms = list(transforms or [])

    def __iter__(self):
        return iter(self.transforms)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        img = out.get("img")
        if isinstance(img, torch.Tensor):
            out["img"] = img + 1
        return out


class FakeEndZones:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        out["end_zone_img"] = out["img"]
        return out


class FakePlayTrackerRuntime(torch.nn.Module):
    def __init__(self, outputs: Optional[dict[str, Any]] = None):
        super().__init__()
        self.outputs = outputs or {}
        self.calls: list[dict[str, Any]] = []
        self.play_box = torch.tensor([0.0, 0.0, 16.0, 9.0], dtype=torch.float32)

    def forward(self, results: dict[str, Any]):  # type: ignore[override]
        self.calls.append(dict(results))
        return dict(self.outputs)


class FakeDetectorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.tensor(1.0))
        self.init_weights_called = False
        self.eval_called = False
        self.to_device = None

    def init_weights(self) -> None:
        self.init_weights_called = True

    def eval(self):  # type: ignore[override]
        self.eval_called = True
        return super().eval()

    def to(self, device: Any):  # type: ignore[override]
        self.to_device = device
        return self


class FakePoseInferencer:
    def __init__(self, results: list[dict[str, Any]]):
        self._results = list(results)
        self.filter_args: dict[str, Any] = {}
        self.inferencer = SimpleNamespace(visualizer=None)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any):
        self.calls.append(dict(kwargs))
        for result in self._results:
            yield result
