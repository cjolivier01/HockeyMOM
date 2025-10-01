from typing import Any, Dict, Optional

from .base import Trunk


class PoseInferencerFactoryTrunk(Trunk):
    """
    Builds and caches an MMPoseInferencer and exposes it via context['pose_inferencer'].

    Params:
      - pose_config: str path to pose2d config (mmengine-style path or alias)
      - pose_checkpoint: Optional[str] checkpoint path/URL
      - device: Optional[str] device string (overrides context['device'] if provided)
      - show_progress: bool (default False)
      - filter_args: Optional[Dict] default preprocess/forward args (bbox_thr, nms_thr, etc.)
    """

    def __init__(
        self,
        pose_config: Optional[str] = None,
        pose_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        show_progress: bool = False,
        filter_args: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._pose_config = pose_config
        self._pose_checkpoint = pose_checkpoint
        self._device = device
        self._show_progress = bool(show_progress)
        self._filter_args = dict(filter_args or {})
        self._inferencer = None

    def _default_filter_args(self, pose_config: Optional[str]) -> Dict[str, Any]:
        # Defaults taken from hmtrack CLI logic
        args = dict(bbox_thr=0.2, nms_thr=0.3, pose_based_nms=False)
        if not pose_config:
            return args
        # Customize by model string
        spec = {
            "yoloxpose": dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
            "rtmo": dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
            "rtmp": dict(kpt_thr=0.3, pose_based_nms=False, disable_norm_pose_2d=False),
        }
        for k, v in spec.items():
            if k in pose_config:
                args.update(v)
                break
        return args

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        if self._inferencer is None:
            from mmpose.apis.inferencers import MMPoseInferencer

            cfg = self._pose_config
            ckpt = self._pose_checkpoint
            dev = self._device
            if dev is None:
                device_obj = context.get("device")
                if device_obj is not None:
                    dev = str(device_obj)
            inferencer = MMPoseInferencer(
                pose2d=cfg,
                pose2d_weights=ckpt,
                device=dev,
                show_progress=self._show_progress,
            )
            # Merge default + provided filter args
            fa = self._default_filter_args(cfg)
            fa.update(self._filter_args)
            inferencer.filter_args = fa
            self._inferencer = inferencer

        context["pose_inferencer"] = self._inferencer
        return {"pose_inferencer": self._inferencer}

    def input_keys(self):
        return {"device"}

    def output_keys(self):
        return {"pose_inferencer"}

