from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.structures import InstanceData

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
        trt: Optional[Dict[str, Any]] = None,
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
        # TensorRT options
        self._trt_cfg: Dict[str, Any] = dict(trt or {})
        self._trt_wrapper: Optional[_TrtDetectorWrapper] = None

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
            from mmengine.config import ConfigDict
            from mmyolo.registry import MODELS

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

            if "yolodetector" in model_cfg.get("type", "").lower():
                # Ensure we have mmyolo installed for YOLOv8 support
                try:
                    import mmyolo.models.detectors.yolo_detector  # type: ignore
                except Exception as ex:
                    raise RuntimeError("mmyolo is required for YOLOv8 models but is not installed.") from ex

            model = MODELS.build(model_cfg)

            if hasattr(model, "init_weights"):
                model.init_weights()

            if self._to_device and "device" in context and isinstance(context["device"], torch.device):
                model = model.to(context["device"])  # type: ignore[assignment]
            model.eval()
            self._model = model

        # Decide backend: TensorRT (highest), ONNX, or PyTorch
        use_trt = bool(self._trt_cfg.get("enable", False))
        if use_trt:
            if self._trt_wrapper is None:
                try:
                    self._trt_wrapper = _TrtDetectorWrapper(
                        model=self._model,
                        engine_path=str(self._trt_cfg.get("engine", "detector.engine")),
                        force_build=bool(self._trt_cfg.get("force_build", False)),
                        fp16=bool(self._trt_cfg.get("fp16", True)),
                    )
                except Exception as ex:
                    # Fall back to next option if TRT init fails
                    print(f"Failed to initialize TensorRT detector wrapper: {ex}")
                    self._trt_wrapper = None
            if self._trt_wrapper is not None:
                context["detector_model"] = self._trt_wrapper
                return {"detector_model": self._trt_wrapper}

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
        self._use_cuda = False

        # Gather preproc config
        self._mean = None
        self._std = None
        self._mean_t: Optional[torch.Tensor] = None
        self._std_t: Optional[torch.Tensor] = None
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
                        self._mean_t = mean.view(1, -1, 1, 1).detach()
                    elif mean.ndim == 4:
                        self._mean = mean.detach().cpu().numpy()
                        self._mean_t = mean.detach()
                if isinstance(std, torch.Tensor):
                    if std.ndim == 1:
                        self._std = std.view(1, -1, 1, 1).detach().cpu().numpy()
                        self._std_t = std.view(1, -1, 1, 1).detach()
                    elif std.ndim == 4:
                        self._std = std.detach().cpu().numpy()
                        self._std_t = std.detach()
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

        # Session options for maximum graph optimizations
        so = ort.SessionOptions()
        try:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            pass
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        providers = []
        try:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._use_cuda = True
            else:
                providers = ["CPUExecutionProvider"]
                self._use_cuda = False
        except Exception:
            providers = ["CPUExecutionProvider"]
            self._use_cuda = False
        self._ort_providers = providers
        self._ort_sess = ort.InferenceSession(path, sess_options=so, providers=providers)  # type: ignore

    def _maybe_quantize_after_calib(self) -> None:
        if not self.quantize_int8 or self._quantized:
            return
        if self.calib_target <= 0 or len(self._calib_inputs) < self.calib_target:
            return
        try:
            from onnxruntime.quantization import (  # type: ignore
                CalibrationDataReader,
                QuantFormat,
                QuantType,
                quantize_static,
            )
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

    def _preprocess_gpu(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean_t is not None and self._std_t is not None:
            mean = self._mean_t.to(device=x.device, dtype=x.dtype)
            std = self._std_t.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x.contiguous()

    def _run_ort(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run ORT either via CUDA I/O binding (if available) or CPU numpy path.
        Returns a list of torch tensors on the model device.
        """
        import numpy as np

        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        if self._use_cuda and x.is_cuda:
            # CUDA I/O binding with zero-copy via ptr
            io = self._ort_sess.io_binding()
            input_name = self._ort_sess.get_inputs()[0].name
            xc = self._preprocess_gpu(x)
            # Bind input GPU memory
            io.bind_input(
                name=input_name,
                device_type="cuda",
                device_id=int(xc.device.index or 0),
                element_type=np.float32,
                shape=list(xc.shape),
                buffer_ptr=int(xc.data_ptr()),
            )
            # Bind outputs to CUDA to avoid host copies
            for out in self._ort_sess.get_outputs():
                io.bind_output(out.name, device_type="cuda")
            self._ort_sess.run_with_iobinding(io)
            ort_outs = io.get_outputs()
            # Convert OrtValues (on CUDA) to torch via DLPack
            outs_t: List[torch.Tensor] = []
            try:
                from torch.utils.dlpack import from_dlpack  # type: ignore

                for ov in ort_outs:
                    outs_t.append(from_dlpack(ov.to_dlpack()).to(device=dev))
            except Exception:
                # Fallback: copy to CPU numpy and back to torch
                for ov in ort_outs:
                    np_arr = np.array(ov)  # copies from device
                    outs_t.append(torch.from_numpy(np_arr).to(device=dev))
            return outs_t
        # CPU path
        x_np = self._preprocess(x)
        outs = self._ort_sess.run(None, {self._ort_sess.get_inputs()[0].name: x_np})
        return [torch.from_numpy(o).to(device=dev) for o in outs]

    def predict(self, imgs: torch.Tensor, data_samples: List[Any]):  # type: ignore[override]
        # imgs: (N, C, H, W)
        assert isinstance(data_samples, (list, tuple)) and len(data_samples) == imgs.size(0)
        N = imgs.size(0)
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        # Calibration capture (CPU) if requested
        if self.quantize_int8 and not self._quantized and self.calib_target > 0:
            # Store only up to calib_target samples
            remaining = max(0, self.calib_target - len(self._calib_inputs))
            if remaining > 0:
                take = min(N, remaining)
                for i in range(take):
                    x_np = self._preprocess(imgs[i : i + 1])
                    self._calib_inputs.append(x_np.copy())
        # Run ORT for the whole batch
        outs = self._run_ort(imgs)
        feats: List[torch.Tensor] = [o for o in outs[:3]]
        with torch.inference_mode():
            cls_scores, bbox_preds, objectnesses = self.model.bbox_head(tuple(feats))
            # Collect per-frame img_metas
            metas = []
            for i in range(N):
                try:
                    metas.append(getattr(data_samples[i], "metainfo", {}))
                except Exception:
                    metas.append({})
            result_list: List[InstanceData] = self.model.bbox_head.predict_by_feat(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                objectnesses=objectnesses,
                batch_img_metas=metas,
                cfg=None,
                rescale=True,
                with_nms=True,
            )
        # Optional: try to quantize after gathering enough samples
        self._maybe_quantize_after_calib()
        # Wrap results to mimic mmdet return objects
        results: List[Any] = []
        for inst in result_list:

            class _Wrap:
                def __init__(self, inst_):
                    self.pred_instances = inst_

            results.append(_Wrap(inst))
        return results


class _TrtDetectorWrapper:
    """Backbone+neck with TensorRT via torch2trt; decode with PyTorch YOLOX head.

    Builds the engine on-the-fly and caches to disk. Exposes a .predict compatible
    with mmdet SingleStageDetector expectations used downstream.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        engine_path: str,
        force_build: bool = False,
        fp16: bool = True,
    ):
        self.model = model
        self.engine_path = Path(engine_path)
        self.force_build = bool(force_build)
        self.fp16 = bool(fp16)
        self._trt_module = None
        self._mean = None
        self._std = None
        self._swap_rgb = False

        # Gather data_preprocessor normalization like ONNX path
        try:
            dp = getattr(model, "data_preprocessor", None)
            if dp is not None:
                mean = getattr(dp, "mean", None)
                std = getattr(dp, "std", None)
                if mean is not None:
                    if not isinstance(mean, torch.Tensor):
                        try:
                            mean = torch.as_tensor(mean)
                        except Exception:
                            mean = None
                    if isinstance(mean, torch.Tensor):
                        if mean.ndim == 1:
                            self._mean = mean.view(1, -1, 1, 1).detach().cpu()
                        elif mean.ndim == 4:
                            self._mean = mean.detach().cpu()
                if std is not None:
                    if not isinstance(std, torch.Tensor):
                        try:
                            std = torch.as_tensor(std)
                        except Exception:
                            std = None
                    if isinstance(std, torch.Tensor):
                        if std.ndim == 1:
                            self._std = std.view(1, -1, 1, 1).detach().cpu()
                        elif std.ndim == 4:
                            self._std = std.detach().cpu()
                bgr_to_rgb = bool(getattr(dp, "bgr_to_rgb", False))
                rgb_to_bgr = bool(getattr(dp, "rgb_to_bgr", False))
                self._swap_rgb = bool(bgr_to_rgb or rgb_to_bgr)
        except Exception:
            pass

    def _ensure_trt_engine(self, shape: torch.Shape, dtype: torch.dtype) -> None:
        if self._trt_module is not None:
            return
        try:
            import torch2trt  # type: ignore
        except Exception as ex:
            raise RuntimeError("torch2trt is required for TensorRT path but is not available") from ex
        import os

        portions: list[str] = [self.engine_path.stem]
        for dim in shape:
            portions.append(str(dim))
        self.engine_path.stem = "_".join(portions)

        dev = next(self.model.parameters()).device
        wrapper = _BackboneNeckWrapper(self.model).eval().to(dev)
        if (not self.force_build) and os.path.exists(self.engine_path):
            try:
                trt_mod = torch2trt.TRTModule()
                import torch as _torch

                trt_mod.load_state_dict(_torch.load(self.engine_path))
                self._trt_module = trt_mod
                return
            except Exception:
                pass
        # Build
        sample = torch.randn(*shape, device=dev, dtype=dtype)
        print("Building TensorRT engine for detector backbone+neck...")
        with torch.inference_mode():
            trt_mod = torch2trt.torch2trt(
                wrapper,
                [sample],
                fp16_mode=self.fp16,
                max_batch_size=shape[0],
                max_workspace_size=1 << 30,
            )
        # Save engine
        try:
            import torch as _torch

            _torch.save(trt_mod.state_dict(), self.engine_path)
            print(f"Saved TensorRT engine to {self.engine_path}")
        except Exception:
            print(f"Failed to save TensorRT engine to {self.engine_path}")
        self._trt_module = trt_mod

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize similar to data_preprocessor if present
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean is not None and self._std is not None:
            mean = self._mean.to(device=x.device, dtype=x.dtype)
            std = self._std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x

    def predict(self, imgs: torch.Tensor, data_samples: List[Any]):  # type: ignore[override]
        assert isinstance(data_samples, (list, tuple)) and len(data_samples) == imgs.size(0)
        results: List[Any] = []
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(imgs.size(0)):
            data_sample = data_samples[i]
            try:
                img_meta = getattr(data_sample, "metainfo", {})
            except Exception:
                img_meta = {}
            x = imgs[i : i + 1].to(device=dev, non_blocking=True)
            x = self._preprocess(x)
            self._ensure_trt_engine(shape=x.shape, dtype=x.dtype)
            assert self._trt_module is not None, "TRT engine not initialized"
            feats = self._trt_module(x)
            if isinstance(feats, torch.Tensor):
                feats = [feats, feats, feats]
            elif isinstance(feats, (list, tuple)):
                feats = list(feats)
            else:
                feats = [torch.as_tensor(feats)]
            with torch.inference_mode():
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
            inst = result_list[0]

            class _Wrap:
                def __init__(self, inst_):
                    self.pred_instances = inst_

            results.append(_Wrap(inst))
        return results
