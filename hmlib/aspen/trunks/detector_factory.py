from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

import torch

try:
    from mmengine.config import ConfigDict
    from mmdet.registry import MODELS
    from mmengine.structures import InstanceData
except Exception:  # pragma: no cover - optional at runtime
    MODELS = None  # type: ignore
    ConfigDict = dict  # type: ignore
    InstanceData = object  # type: ignore

from .base import Trunk


class DetectorFactoryTrunk(Trunk):
    """
    Builds and caches a pure detection model (e.g., YOLOX) from Aspen YAML.

    Params:
      - detector: dict model config (optional)
      - detector_yaml: str path to YAML with a top-level detector definition (optional)
      - data_preprocessor: dict (optional), passed to model as TrackDataPreprocessor-compatible
      - to_device: bool (default True)

    Outputs in context:
      - detector_model: the constructed detection model (eval mode)
    """

    def __init__(
        self,
        detector: Optional[Dict[str, Any]] = None,
        detector_yaml: Optional[str] = None,
        data_preprocessor: Optional[Dict[str, Any]] = None,
        to_device: bool = True,
        enabled: bool = True,
        onnx: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(enabled=enabled)
        self._detector_dict = detector
        self._detector_yaml = detector_yaml
        self._data_preprocessor = data_preprocessor
        self._to_device = to_device
        self._model = None
        # ONNX runtime/export options
        self._onnx_cfg: Dict[str, Any] = dict(onnx or {})
        self._onnx_wrapper: Optional[_OnnxDetectorWrapper] = None

    def _load_detector_from_yaml(self, path: str) -> Dict[str, Any]:
        import yaml

        with open(path, "r") as f:
            y = yaml.safe_load(f)
        if isinstance(y, dict) and "detector" in y:
            return y["detector"]
        return y

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        if bool(context.get("using_precalculated_detection", False)):
            # Skip building a detector if detections are provided externally
            return {}
        if self._model is None:
            if MODELS is None:
                raise RuntimeError("mmdet MODELS registry is unavailable; cannot build detector.")

            detector_cfg = self._detector_dict
            if detector_cfg is None and self._detector_yaml:
                detector_cfg = self._load_detector_from_yaml(self._detector_yaml)

            # Convert to ConfigDict recursively for mmengine
            def _to_cfg(x):
                if isinstance(x, dict):
                    return ConfigDict({k: _to_cfg(v) for k, v in x.items()})
                if isinstance(x, list):
                    return [_to_cfg(v) for v in x]
                return x

            model_cfg = _to_cfg(detector_cfg)
            if self._data_preprocessor is not None:
                # Attach data_preprocessor to the model config before build
                if not isinstance(model_cfg, dict):
                    # Convert to dict to inject
                    model_cfg = ConfigDict(model_cfg)
                model_cfg["data_preprocessor"] = _to_cfg(self._data_preprocessor)
            model = MODELS.build(model_cfg)

            if hasattr(model, "init_weights"):
                model.init_weights()

            if self._to_device and "device" in context and isinstance(context["device"], torch.device):
                model = model.to(context["device"])  # type: ignore[assignment]
            model.eval()
            self._model = model

        # Decide backend: PyTorch (default) or ONNX
        use_onnx = bool(self._onnx_cfg.get("enable", False))
        if use_onnx:
            if self._onnx_wrapper is None:
                try:
                    self._onnx_wrapper = _OnnxDetectorWrapper(
                        model=self._model,
                        onnx_path=str(self._onnx_cfg.get("path", "detector.onnx")),
                        force_export=bool(self._onnx_cfg.get("force_export", False)),
                        quantize_int8=bool(self._onnx_cfg.get("quantize_int8", False)),
                        calib_frames=int(self._onnx_cfg.get("calib_frames", 0)),
                        game_id=str(context.get("game_id", "")),
                    )
                except Exception as ex:
                    # Fall back to PyTorch if ONNX init fails
                    print(f"Failed to initialize ONNX detector wrapper: {ex}")
                    self._onnx_wrapper = None
            if self._onnx_wrapper is not None:
                context["detector_model"] = self._onnx_wrapper
                return {"detector_model": self._onnx_wrapper}

        # Default: PyTorch detector
        context["detector_model"] = self._model
        return {"detector_model": self._model}

    def input_keys(self):
        return {"device", "using_precalculated_detection"}

    def output_keys(self):
        return {"detector_model"}


class _BackboneNeckWrapper(torch.nn.Module):
    """Thin wrapper that exposes backbone+neck forward for export.

    Returns three feature maps (small->large stride) as individual outputs.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        feats = self.backbone(x)
        feats = self.neck(feats)
        if isinstance(feats, (list, tuple)):
            assert len(feats) >= 3, "Expected at least 3 FPN levels"
            return feats[0], feats[1], feats[2]
        # Some necks may return a single tensor; mirror into three
        return feats, feats, feats


class _OnnxDetectorWrapper:
    """Wraps YOLOX detector to run backbone+neck in ONNX Runtime, then decodes
    with the existing PyTorch YOLOX head to produce InstanceData results.

    - Exposes a `.predict(images, [data_sample])` compatible with mmdet
      SingleStageDetector.predict() return type expectations used by
      DetectorInferenceTrunk (list with object having .pred_instances).
    - Optional on-the-fly static INT8 quantization after collecting N frames.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        onnx_path: str,
        force_export: bool = False,
        quantize_int8: bool = False,
        calib_frames: int = 0,
        game_id: str = "",
    ):
        self.model = model
        self.onnx_path = onnx_path
        self.force_export = force_export
        self.quantize_int8 = bool(quantize_int8)
        self.calib_target = int(calib_frames or 0)
        self._calib_inputs: List[Any] = []
        self._quantized = False
        self._quantized_path: Optional[str] = None

        # Gather preproc config
        self._mean = None
        self._std = None
        self._swap_rgb = False
        try:
            dp = getattr(model, "data_preprocessor", None)
            if dp is not None:
                mean = getattr(dp, "mean", None)
                std = getattr(dp, "std", None)
                # TrackDataPreprocessor may store (1,C,1,1)
                if isinstance(mean, torch.Tensor):
                    if mean.ndim == 1:
                        self._mean = mean.view(1, -1, 1, 1).detach().cpu().numpy()
                    elif mean.ndim == 4:
                        self._mean = mean.detach().cpu().numpy()
                if isinstance(std, torch.Tensor):
                    if std.ndim == 1:
                        self._std = std.view(1, -1, 1, 1).detach().cpu().numpy()
                    elif std.ndim == 4:
                        self._std = std.detach().cpu().numpy()
                bgr_to_rgb = bool(getattr(dp, "bgr_to_rgb", False))
                rgb_to_bgr = bool(getattr(dp, "rgb_to_bgr", False))
                self._swap_rgb = bool(bgr_to_rgb or rgb_to_bgr)
        except Exception:
            pass

        # Export ONNX if necessary
        self._ensure_onnx_export()
        # Create an ORT session
        self._create_session(self.onnx_path)

    def _ensure_onnx_export(self) -> None:
        import os
        export_needed = self.force_export or (not os.path.exists(self.onnx_path))
        if not export_needed:
            return
        # Prepare export module and sample input (dynamic axes set)
        wrapper = _BackboneNeckWrapper(self.model).eval()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        wrapper.to(device)
        # Choose a reasonable default input size; graph is dynamic on H/W
        sample = torch.randn(1, 3, 480, 1312, device=device, dtype=torch.float32)
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                sample,
                self.onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["images"],
                output_names=["feat0", "feat1", "feat2"],
                dynamic_axes={
                    "images": {0: "batch", 2: "height", 3: "width"},
                    "feat0": {0: "batch", 2: "h0", 3: "w0"},
                    "feat1": {0: "batch", 2: "h1", 3: "w1"},
                    "feat2": {0: "batch", 2: "h2", 3: "w2"},
                },
            )

    def _create_session(self, path: str) -> None:
        import onnxruntime as ort  # type: ignore
        providers = []
        try:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        except Exception:
            providers = ["CPUExecutionProvider"]
        self._ort_providers = providers
        self._ort_sess = ort.InferenceSession(path, providers=providers)  # type: ignore

    def _maybe_quantize_after_calib(self) -> None:
        if not self.quantize_int8 or self._quantized:
            return
        if self.calib_target <= 0 or len(self._calib_inputs) < self.calib_target:
            return
        try:
            from onnxruntime.quantization import (
                CalibrationDataReader,
                QuantType,
                QuantFormat,
                quantize_static,
            )  # type: ignore
        except Exception:
            # Quantization not available; skip
            self._calib_inputs = []
            return

        # Create a simple calibration data reader
        input_name = self._ort_sess.get_inputs()[0].name

        class _Reader(CalibrationDataReader):  # type: ignore
            def __init__(self, name: str, samples: List[Any]):
                self.name = name
                self._iter = iter(samples)

            def get_next(self):  # type: ignore
                try:
                    x = next(self._iter)
                    return {self.name: x}
                except StopIteration:
                    return None

        reader = _Reader(input_name, self._calib_inputs)
        q_path = self.onnx_path.replace(".onnx", ".int8.onnx")
        try:
            quantize_static(
                model_input=self.onnx_path,
                model_output=q_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            # Reload session with the quantized model
            self._create_session(q_path)
            self._quantized = True
            self._quantized_path = q_path
        except Exception:
            # If quant fails, continue with fp32
            pass
        finally:
            self._calib_inputs = []

    def _preprocess(self, x: torch.Tensor) -> Any:
        # x: (1, C, H, W) torch tensor on any device
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean is not None and self._std is not None:
            mean = torch.from_numpy(self._mean).to(device=x.device, dtype=x.dtype)
            std = torch.from_numpy(self._std).to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        # ORT expects numpy on CPU
        x_cpu = x.detach().to("cpu")
        return x_cpu.numpy()

    def predict(self, single_img: torch.Tensor, data_samples: List[Any]):  # type: ignore[override]
        # single_img: (1, C, H, W)
        assert isinstance(data_samples, (list, tuple)) and len(data_samples) == 1
        data_sample = data_samples[0]
        # Build batch_img_metas for decode
        try:
            img_meta = getattr(data_sample, "metainfo", {})
        except Exception:
            img_meta = {}

        x_np = self._preprocess(single_img)
        # Save calibration inputs if requested
        if self.quantize_int8 and not self._quantized and self.calib_target > 0:
            # Store a copy to avoid mutation
            self._calib_inputs.append(x_np.copy())

        # Run ORT backbone+neck
        outs = self._ort_sess.run(None, {self._ort_sess.get_inputs()[0].name: x_np})
        # Convert features to torch on the same device as model head
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        feats: List[torch.Tensor] = [torch.from_numpy(o).to(device=dev) for o in outs[:3]]
        # Forward YOLOX head
        with torch.no_grad():
            cls_scores, bbox_preds, objectnesses = self.model.bbox_head(tuple(feats))
            result_list: List[InstanceData] = self.model.bbox_head.predict_by_feat(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                objectnesses=objectnesses,
                batch_img_metas=[img_meta],
                cfg=None,
                rescale=True,
                with_nms=True,
            )

        # Optional: try to quantize after gathering enough samples
        self._maybe_quantize_after_calib()

        inst = result_list[0]
        # DetectorInferenceTrunk expects an object with .pred_instances
        class _Wrap:
            def __init__(self, inst_):
                self.pred_instances = inst_

        return [_Wrap(inst)]
