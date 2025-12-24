from contextlib import nullcontext
from typing import Any, Dict

import torch

from hmlib.utils.gpu import unwrap_tensor, wrap_tensor
from hmlib.utils.image import make_channels_first

from .base import Plugin
from .detector_factory_plugin import _strip_static_padding


class DetectorInferencePlugin(Plugin):
    """
    Runs a pure detection model per-frame and attaches `pred_instances`.

    Expects in context:
      - data: dict with keys 'img' (1, T, C, H, W) and 'data_samples' (list[TrackDataSample])
      - detector_model: mmdet detector with .predict
      - fp16: bool (optional)
      - detect_timer: optional timer with tic/toc

    Produces in context:
      - data: updated with per-frame `pred_instances` set on each video_data_sample
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._iter_num: int = 0

    def __call__(self, *args, **kwargs) -> Any:
        self._iter_num += 1
        return super().__call__(*args, **kwargs)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        if bool(context.get("using_precalculated_detection", False)):
            return {}
        detector = context.get("detector_model")
        if detector is None:
            return {}

        fp16: bool = bool(context.get("fp16", False))
        detect_timer = context.get("detect_timer")
        if detect_timer is not None:
            detect_timer.tic()

        detection_image = unwrap_tensor(data["img"])
        data["img"] = detection_image

        track_samples = data.get("data_samples")
        # Accept either TrackDataSample or [TrackDataSample]
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        with torch.no_grad():
            det_device = (
                detection_image.device
                if isinstance(detection_image, torch.Tensor)
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            use_autocast = bool(fp16 and det_device.type == "cuda")
            amp_ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
                if use_autocast
                else nullcontext()
            )

            if isinstance(detection_image, torch.Tensor):
                batch_inputs = detection_image
            else:
                batch_inputs = torch.as_tensor(detection_image)
                batch_inputs = batch_inputs
            if batch_inputs.ndim != 4:
                raise AssertionError(
                    "DetectorInferencePlugin expects per-frame tensors shaped (T, C, H, W)"
                )
            batch_inputs = make_channels_first(batch_inputs).contiguous()
            assert batch_inputs.size(0) == video_len, "Mismatch between frames and detection inputs"

            batch_data_samples = []
            for frame_id in range(video_len):
                img_data_sample = track_data_sample[frame_id]
                # Ensure frame_id metainfo is present for downstream postprocessing
                fid = img_data_sample.metainfo.get("img_id")
                try:
                    if isinstance(fid, torch.Tensor):
                        fid = fid.reshape([1])[0].item()
                    if fid is None:
                        fid = frame_id
                except Exception:
                    fid = frame_id
                img_data_sample.set_metainfo({"frame_id": int(fid)})
                batch_data_samples.append(img_data_sample)

            with amp_ctx:
                det_results = detector.predict(batch_inputs, batch_data_samples)

            assert (
                isinstance(det_results, (list, tuple)) and len(det_results) == video_len
            ), "Batch inference must return one result per frame"
            for img_data_sample, det_sample in zip(batch_data_samples, det_results):
                # Attach detections to the video data sample for downstream tracker
                img_data_sample.pred_instances = _strip_static_padding(
                    det_sample.pred_instances, strip=False
                )
                img_data_sample.pred_instances.bboxes = wrap_tensor(
                    img_data_sample.pred_instances.bboxes
                )
                img_data_sample.pred_instances.labels = wrap_tensor(
                    img_data_sample.pred_instances.labels
                )
                img_data_sample.pred_instances.scores = wrap_tensor(
                    img_data_sample.pred_instances.scores
                )
        if detect_timer is not None:
            detect_timer.toc()

        return {"data": data}

    def input_keys(self):
        return {"data", "detector_model", "fp16", "detect_timer", "using_precalculated_detection"}

    def output_keys(self):
        return {"data"}
