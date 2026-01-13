from typing import Any, Dict, List, Optional

import torch

from .base import Plugin


class PoseInferencerFactoryPlugin(Plugin):
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
        disable_internal_detector: bool = False,
        enabled: bool = True,
        onnx: Optional[Dict[str, Any]] = None,
        trt: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(enabled=enabled)
        self._pose_config = pose_config
        self._pose_checkpoint = pose_checkpoint
        self._device = device
        self._show_progress = bool(show_progress)
        self._filter_args = dict(filter_args or {})
        self._disable_internal_detector = bool(disable_internal_detector)
        self._inferencer = None
        # ONNX runtime/export options for pose (backbone+neck)
        self._onnx_cfg: Dict[str, Any] = dict(onnx or {})
        self._onnx_runner: Optional[_OnnxPoseRunner] = None
        # TensorRT options for pose (backbone+neck)
        self._trt_cfg: Dict[str, Any] = dict(trt or {})
        self._trt_runner: Optional[_TrtPoseRunner] = None

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

    @staticmethod
    def _wrap_model_test_step(model: Optional[torch.nn.Module], runner: Optional[Any]) -> None:
        if model is None or runner is None:
            return
        if getattr(model, "_hm_test_step_wrapped", False):
            return
        orig_test_step = getattr(model, "test_step", None)
        if not callable(orig_test_step):
            return
        try:
            runner.set_fallback_test_step(orig_test_step)
        except Exception:
            pass

        def _test_step(data, *args, **kwargs):
            try:
                preds = runner.forward(data)
                if isinstance(preds, (list, tuple)):
                    return list(preds)
            except Exception:
                pass
            return orig_test_step(data, *args, **kwargs)

        setattr(model, "_hm_test_step_wrapped", True)
        setattr(model, "_hm_orig_test_step", orig_test_step)
        model.test_step = _test_step  # type: ignore[assignment]

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
            # Skip initializing MMPose's own detector when requested or when
            # detections are already provided upstream.
            det_model = None
            try:
                if self._disable_internal_detector or bool(
                    context.get("using_precalculated_detection", False)
                ):
                    det_model = "whole_image"
            except Exception:
                det_model = None

            inferencer = MMPoseInferencer(
                pose2d=cfg,
                pose2d_weights=ckpt,
                device=dev,
                det_model=det_model,
                show_progress=self._show_progress,
            )
            # Merge default + provided filter args
            fa = self._default_filter_args(cfg)
            fa.update(self._filter_args)
            inferencer.filter_args = fa
            self._inferencer = inferencer

            # Initialize optional accelerators using underlying inferencer's model
            try:
                pose_impl = getattr(self._inferencer, "inferencer", None)
                # Prefer TensorRT if enabled
                if bool(self._trt_cfg.get("enable", False)):
                    if (
                        pose_impl is not None
                        and getattr(getattr(pose_impl, "cfg", object()), "data_mode", None)
                        == "topdown"
                    ):
                        self._trt_runner = _TrtPoseRunner(
                            model=pose_impl.model,
                            engine_path=str(self._trt_cfg.get("engine", "pose.engine")),
                            force_build=bool(self._trt_cfg.get("force_build", False)),
                            fp16=bool(self._trt_cfg.get("fp16", True)),
                            int8=bool(self._trt_cfg.get("int8", False)),
                            calib_frames=int(self._trt_cfg.get("calib_frames", 0)),
                        )
                        setattr(self._inferencer, "_hm_trt_runner", self._trt_runner)
                # Otherwise, enable ONNX if requested
                if self._trt_runner is None and bool(self._onnx_cfg.get("enable", False)):
                    if (
                        pose_impl is not None
                        and getattr(getattr(pose_impl, "cfg", object()), "data_mode", None)
                        == "topdown"
                    ):
                        self._onnx_runner = _OnnxPoseRunner(
                            model=pose_impl.model,
                            onnx_path=str(self._onnx_cfg.get("path", "pose.onnx")),
                            force_export=bool(self._onnx_cfg.get("force_export", False)),
                            quantize_int8=bool(self._onnx_cfg.get("quantize_int8", False)),
                            calib_frames=int(self._onnx_cfg.get("calib_frames", 0)),
                        )
                        # Attach for PosePlugin to pick up
                        setattr(self._inferencer, "_hm_onnx_runner", self._onnx_runner)
            except Exception:
                # Non-fatal; fall back to PyTorch model
                self._onnx_runner = None
                self._trt_runner = None
            try:
                pose_impl = getattr(self._inferencer, "inferencer", None)
                pose_model = getattr(pose_impl, "model", None) if pose_impl is not None else None
                if self._trt_runner is not None:
                    setattr(self._inferencer, "_hm_pose_runner", self._trt_runner)
                    self._wrap_model_test_step(pose_model, self._trt_runner)
                elif self._onnx_runner is not None:
                    setattr(self._inferencer, "_hm_pose_runner", self._onnx_runner)
                    self._wrap_model_test_step(pose_model, self._onnx_runner)
            except Exception:
                pass

        context["pose_inferencer"] = self._inferencer
        return {"pose_inferencer": self._inferencer}

    def input_keys(self):
        return {"device", "using_precalculated_detection"}

    def output_keys(self):
        return {"pose_inferencer"}


class _BackboneNeckWrapper(torch.nn.Module):
    """Expose backbone+neck forward for pose export.

    Returns a single feature map tensor or a tuple of tensors, depending on the model.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.neck = getattr(model, "neck", None)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats


class _OnnxPoseRunner:
    """Run pose model with ONNX Runtime for backbone+neck, then decode with PyTorch head.

    This integrates with PosePlugin's bypass path by attaching to the inferencer
    as `._hm_onnx_runner` so PosePlugin can optionally use it.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        onnx_path: str,
        force_export: bool = False,
        quantize_int8: bool = False,
        calib_frames: int = 0,
    ):
        self.model = model
        self.onnx_path = onnx_path
        self.force_export = force_export
        self.quantize_int8 = bool(quantize_int8)
        self.calib_target = int(calib_frames or 0)
        self._calib_inputs: List[Any] = []
        self._quantized = False
        self._ort_sess = None
        self._providers = None
        self._use_cuda = False
        self._fallback_test_step = None
        self._mean = None
        self._std = None
        self._mean_t = None
        self._std_t = None
        self._swap_rgb = False
        self._init_preprocess()
        self._export_and_create_session()

    def set_fallback_test_step(self, fn) -> None:
        self._fallback_test_step = fn

    def _init_preprocess(self) -> None:
        try:
            dp = getattr(self.model, "data_preprocessor", None)
            if dp is None:
                return
            mean = getattr(dp, "mean", None)
            std = getattr(dp, "std", None)
            if mean is not None and not isinstance(mean, torch.Tensor):
                try:
                    mean = torch.as_tensor(mean)
                except Exception:
                    mean = None
            if std is not None and not isinstance(std, torch.Tensor):
                try:
                    std = torch.as_tensor(std)
                except Exception:
                    std = None
            if isinstance(mean, torch.Tensor):
                if mean.ndim == 1:
                    mean = mean.view(1, -1, 1, 1)
                elif mean.ndim == 3:
                    mean = mean.unsqueeze(0)
                elif mean.ndim != 4:
                    mean = None
            if isinstance(std, torch.Tensor):
                if std.ndim == 1:
                    std = std.view(1, -1, 1, 1)
                elif std.ndim == 3:
                    std = std.unsqueeze(0)
                elif std.ndim != 4:
                    std = None
            if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
                self._mean = mean.detach().cpu().numpy()
                self._std = std.detach().cpu().numpy()
                self._mean_t = mean.detach()
                self._std_t = std.detach()
            bgr_to_rgb = bool(getattr(dp, "bgr_to_rgb", False))
            rgb_to_bgr = bool(getattr(dp, "rgb_to_bgr", False))
            self._swap_rgb = bool(bgr_to_rgb or rgb_to_bgr)
        except Exception:
            pass

    @staticmethod
    def _ensure_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return torch.empty((0,))
            tensors: List[torch.Tensor] = []
            for item in x:
                t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                if t.ndim == 4 and t.size(0) == 1:
                    t = t.squeeze(0)
                tensors.append(t)
            if len(tensors) == 1:
                t = tensors[0]
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                return t
            return torch.stack(tensors, dim=0)
        return torch.as_tensor(x)

    def _preprocess(self, x: torch.Tensor) -> Any:
        x = self._ensure_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean is not None and self._std is not None:
            mean = torch.from_numpy(self._mean).to(device=x.device, dtype=x.dtype)
            std = torch.from_numpy(self._std).to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        x_cpu = x.detach().to("cpu")
        return x_cpu.numpy()

    def _preprocess_gpu(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean_t is not None and self._std_t is not None:
            mean = self._mean_t.to(device=x.device, dtype=x.dtype)
            std = self._std_t.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x.contiguous()

    def _export_and_create_session(self) -> None:
        import os

        export_needed = self.force_export or (not os.path.exists(self.onnx_path))
        if export_needed:
            wrapper = _BackboneNeckWrapper(self.model).eval()
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            wrapper.to(device)
            # Use a typical person crop size; dynamic H/W enabled
            sample = torch.randn(1, 3, 256, 192, device=device, dtype=torch.float32)
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    sample,
                    self.onnx_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=["images"],
                    output_names=["feats"],
                    dynamic_axes={
                        "images": {0: "batch", 2: "height", 3: "width"},
                        "feats": {0: "batch"},
                    },
                )
        import onnxruntime as ort  # type: ignore

        # Session options with higher optimization level
        so = ort.SessionOptions()
        try:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            pass
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._use_cuda = True
        else:
            self._providers = ["CPUExecutionProvider"]
            self._use_cuda = False
        self._ort_sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=self._providers)  # type: ignore

    def _run_feats(self, x: torch.Tensor) -> Any:
        import numpy as np

        x = self._ensure_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        need_calib = self.quantize_int8 and not self._quantized and self.calib_target > 0
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        if self._use_cuda and x.is_cuda:
            xc = self._preprocess_gpu(x)
            if need_calib and len(self._calib_inputs) < self.calib_target:
                try:
                    x_cpu = xc.detach().to("cpu")
                    remaining = max(0, self.calib_target - len(self._calib_inputs))
                    take = min(int(x_cpu.shape[0]), remaining)
                    for i in range(take):
                        self._calib_inputs.append(x_cpu[i : i + 1].numpy().copy())
                except Exception:
                    pass
            io = self._ort_sess.io_binding()
            input_name = self._ort_sess.get_inputs()[0].name
            io.bind_input(
                name=input_name,
                device_type="cuda",
                device_id=int(xc.device.index or 0),
                element_type=np.float32,
                shape=list(xc.shape),
                buffer_ptr=int(xc.data_ptr()),
            )
            for out in self._ort_sess.get_outputs():
                io.bind_output(out.name, device_type="cuda")
            self._ort_sess.run_with_iobinding(io)
            ort_outs = io.get_outputs()
            outs_t: List[torch.Tensor] = []
            try:
                from torch.utils.dlpack import from_dlpack  # type: ignore

                for ov in ort_outs:
                    outs_t.append(from_dlpack(ov.to_dlpack()).to(device=dev))
            except Exception:
                for ov in ort_outs:
                    np_arr = np.array(ov)
                    outs_t.append(torch.from_numpy(np_arr).to(device=dev))
            if len(outs_t) == 1:
                return outs_t[0]
            return outs_t
        # CPU fallback
        x_np = self._preprocess(x)
        if need_calib and len(self._calib_inputs) < self.calib_target:
            try:
                remaining = max(0, self.calib_target - len(self._calib_inputs))
                take = min(int(x_np.shape[0]), remaining)
                for i in range(take):
                    self._calib_inputs.append(x_np[i : i + 1].copy())
            except Exception:
                pass
        outputs = self._ort_sess.run(None, {self._ort_sess.get_inputs()[0].name: x_np})
        if len(outputs) == 1:
            out = torch.from_numpy(outputs[0]).to(device=dev)
            return out
        return [torch.from_numpy(o).to(device=dev) for o in outputs]

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
            self._calib_inputs = []
            return
        input_name = self._ort_sess.get_inputs()[0].name

        class _Reader(CalibrationDataReader):  # type: ignore
            def __init__(self, name, samples):
                self.name, self._it = name, iter(samples)

            def get_next(self):
                try:
                    x = next(self._it)
                    return {self.name: x}
                except StopIteration:
                    return None

        q_path = self.onnx_path.replace(".onnx", ".int8.onnx")
        try:
            quantize_static(
                model_input=self.onnx_path,
                model_output=q_path,
                calibration_data_reader=_Reader(input_name, self._calib_inputs),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            import onnxruntime as ort  # type: ignore

            self._ort_sess = ort.InferenceSession(q_path, providers=self._providers)
            self._quantized = True
        except Exception:
            pass
        finally:
            self._calib_inputs = []

    def forward(self, proc_inputs: Dict[str, Any]):
        """Best-effort ONNX-backed forward. Falls back to model if decoding fails."""
        try:
            inputs = proc_inputs["inputs"]  # Tensor (N,C,H,W)
            datas = proc_inputs.get("data_samples", None)
            feats = self._run_feats(inputs)
            # Decode with model head if available
            head = getattr(self.model, "head", None)
            if head is not None and hasattr(head, "predict") and datas is not None:
                with torch.no_grad():
                    preds = head.predict(feats, datas)
                # Quantize after gathering calibration frames
                self._maybe_quantize_after_calib()
                return preds
        except Exception:
            pass
        # Fallback: run model directly
        with torch.no_grad():
            try:
                if self._fallback_test_step is not None:
                    return self._fallback_test_step(proc_inputs)
                return self.model.test_step(proc_inputs)
            except Exception:
                try:
                    return self.model(**proc_inputs)
                except Exception:
                    return []


class _TrtPoseRunner:
    """Run pose backbone+neck with TensorRT via torch2trt; decode with PyTorch head."""

    def __init__(
        self,
        model: torch.nn.Module,
        engine_path: str,
        force_build: bool = False,
        fp16: bool = True,
        int8: bool = False,
        calib_frames: int = 0,
    ):
        self.model = model
        self.engine_path = engine_path
        self.force_build = bool(force_build)
        self.fp16 = bool(fp16)
        self.int8 = bool(int8)
        self.calib_frames = int(calib_frames or 0)
        self._trt_module = None
        self._calib_dataset = None
        self._wrapper_module = None
        self._fallback_test_step = None
        self._mean = None
        self._std = None
        self._mean_t = None
        self._std_t = None
        self._swap_rgb = False
        self._init_preprocess()
        self._build_or_load()

    def set_fallback_test_step(self, fn) -> None:
        self._fallback_test_step = fn

    def _init_preprocess(self) -> None:
        try:
            dp = getattr(self.model, "data_preprocessor", None)
            if dp is None:
                return
            mean = getattr(dp, "mean", None)
            std = getattr(dp, "std", None)
            if mean is not None and not isinstance(mean, torch.Tensor):
                try:
                    mean = torch.as_tensor(mean)
                except Exception:
                    mean = None
            if std is not None and not isinstance(std, torch.Tensor):
                try:
                    std = torch.as_tensor(std)
                except Exception:
                    std = None
            if isinstance(mean, torch.Tensor):
                if mean.ndim == 1:
                    mean = mean.view(1, -1, 1, 1)
                elif mean.ndim == 3:
                    mean = mean.unsqueeze(0)
                elif mean.ndim != 4:
                    mean = None
            if isinstance(std, torch.Tensor):
                if std.ndim == 1:
                    std = std.view(1, -1, 1, 1)
                elif std.ndim == 3:
                    std = std.unsqueeze(0)
                elif std.ndim != 4:
                    std = None
            if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
                self._mean = mean.detach().cpu()
                self._std = std.detach().cpu()
                self._mean_t = mean.detach()
                self._std_t = std.detach()
            bgr_to_rgb = bool(getattr(dp, "bgr_to_rgb", False))
            rgb_to_bgr = bool(getattr(dp, "rgb_to_bgr", False))
            self._swap_rgb = bool(bgr_to_rgb or rgb_to_bgr)
        except Exception:
            pass

    @staticmethod
    def _ensure_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return torch.empty((0,))
            tensors: List[torch.Tensor] = []
            for item in x:
                t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                if t.ndim == 4 and t.size(0) == 1:
                    t = t.squeeze(0)
                tensors.append(t)
            if len(tensors) == 1:
                t = tensors[0]
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                return t
            return torch.stack(tensors, dim=0)
        return torch.as_tensor(x)

    def _prepare_inputs(self, x: Any) -> torch.Tensor:
        x = self._ensure_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if x.device != dev:
            x = x.to(device=dev, non_blocking=True)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean_t is not None and self._std_t is not None:
            mean = self._mean_t.to(device=x.device, dtype=x.dtype)
            std = self._std_t.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x.contiguous()

    def _build_or_load(self) -> None:
        try:
            import torch2trt  # type: ignore
        except Exception as ex:
            raise RuntimeError(
                "torch2trt is required for TensorRT pose path but is not available"
            ) from ex
        import os

        # Try to load an existing engine at the provided path; otherwise, defer building
        if (not self.force_build) and os.path.exists(self.engine_path):
            try:
                trt_mod = torch2trt.TRTModule()
                import torch as _torch

                trt_mod.load_state_dict(_torch.load(self.engine_path))
                self._trt_module = trt_mod
                return
            except Exception:
                pass
        # Defer building until first inputs are seen so shapes/dtypes are known
        return

    def _ensure_trt_engine(self, shape: torch.Size, dtype: torch.dtype) -> None:
        if self._trt_module is not None:
            return
        try:
            import torch2trt  # type: ignore
        except Exception:
            return
        import os
        from pathlib import Path as _Path

        # Append shape (and int8 tag) to engine filename to specialize per input
        base = _Path(self.engine_path)
        portions = [base.stem]
        if self.int8:
            portions.append("int8")
        for dim in shape:
            portions.append(str(dim))
        engine_path = base.with_stem("_".join(portions))
        if (not self.force_build) and os.path.exists(engine_path):
            try:
                trt_mod = torch2trt.TRTModule()
                import torch as _torch

                trt_mod.load_state_dict(_torch.load(engine_path))
                self._trt_module = trt_mod
                return
            except Exception:
                pass
        # Build engine now
        try:
            try:
                dev = next(self.model.parameters()).device
            except StopIteration:
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            wrapper = _BackboneNeckWrapper(self.model).eval().to(dev)
            with torch.inference_mode():
                if self.int8:
                    if self._calib_dataset is None or len(self._calib_dataset) == 0:
                        return
                    trt_mod = torch2trt.torch2trt(
                        wrapper,
                        self._calib_dataset,
                        int8_mode=True,
                        int8_calib_dataset=self._calib_dataset,
                        fp16_mode=False,
                        max_workspace_size=1 << 28,
                    )
                else:
                    sample = torch.randn(*shape, device=dev, dtype=dtype)
                    trt_mod = torch2trt.torch2trt(
                        wrapper,
                        [sample],
                        fp16_mode=self.fp16,
                        max_batch_size=shape[0],
                        max_workspace_size=1 << 28,
                    )
            try:
                import torch as _torch

                _torch.save(trt_mod.state_dict(), engine_path)
            except Exception:
                pass
            self._trt_module = trt_mod
        except Exception as ex:
            from hmlib.log import get_logger

            get_logger(__name__).warning("TRT build for pose failed: %s", ex)

    def _maybe_build_int8(self, sample_shape: torch.Size, sample_dtype: torch.dtype):
        if self._trt_module is not None:
            return
        if not self.int8:
            return
        if self.calib_frames <= 0:
            return
        try:
            import torch2trt  # type: ignore
        except Exception:
            return
        # Prepare dataset and build when enough samples
        if self._calib_dataset is None:
            try:
                from torch2trt import ListDataset  # type: ignore
            except Exception:
                return
            self._calib_dataset = ListDataset()
        if len(self._calib_dataset) < self.calib_frames:
            return
        # Build INT8 engine
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wrapper = _BackboneNeckWrapper(self.model).eval().to(dev)
        try:
            with torch.inference_mode():
                trt_mod = torch2trt.torch2trt(
                    wrapper,
                    self._calib_dataset,
                    int8_mode=True,
                    int8_calib_dataset=self._calib_dataset,
                    fp16_mode=False,
                    max_workspace_size=1 << 28,
                )
            # Save
            try:
                import torch as _torch

                _torch.save(trt_mod.state_dict(), self.engine_path)
            except Exception:
                pass
            self._trt_module = trt_mod
        except Exception as ex:
            # Fallback: try building non-int8 engine to proceed
            from hmlib.log import get_logger

            get_logger(__name__).warning("Pose INT8 build failed, falling back to non-INT8: %s", ex)
            try:
                with torch.inference_mode():
                    trt_mod = torch2trt.torch2trt(
                        wrapper,
                        [torch.randn(*sample_shape, device=dev, dtype=sample_dtype)],
                        fp16_mode=self.fp16,
                        max_batch_size=1,
                        max_workspace_size=1 << 28,
                    )
                try:
                    import torch as _torch

                    _torch.save(trt_mod.state_dict(), self.engine_path)
                except Exception:
                    pass
                self._trt_module = trt_mod
            except Exception:
                self._trt_module = None

    def _run_feats(self, x: torch.Tensor):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        # Use TRT if available, otherwise fallback to PyTorch forward
        if self._trt_module is not None:
            with torch.inference_mode():
                return self._trt_module(x)
        with torch.inference_mode():
            if self._wrapper_module is None:
                self._wrapper_module = _BackboneNeckWrapper(self.model).eval().to(x.device)
            return self._wrapper_module(x)

    def forward(self, proc_inputs: Dict[str, Any]):
        try:
            inputs = self._prepare_inputs(proc_inputs["inputs"])
            datas = proc_inputs.get("data_samples", None)
            # Calibration capture for INT8; defer fp16/fp32 build to first inputs
            if self.int8:
                if self._calib_dataset is None and self.calib_frames > 0:
                    try:
                        from torch2trt import ListDataset  # type: ignore

                        self._calib_dataset = ListDataset()
                    except Exception:
                        self._calib_dataset = None
                if self._calib_dataset is not None and len(self._calib_dataset) < self.calib_frames:
                    # store GPU tensors for calibration to match module device
                    remaining = max(0, self.calib_frames - len(self._calib_dataset))
                    take = min(int(inputs.size(0)), remaining)
                    for i in range(take):
                        self._calib_dataset.insert([inputs[i : i + 1].detach()])
                # Try to build INT8 engine once enough samples collected
                if (
                    self._calib_dataset is not None
                    and len(self._calib_dataset) >= self.calib_frames
                ):
                    self._ensure_trt_engine(inputs.shape, inputs.dtype)
            else:
                # Lazy build for fp16/fp32 path
                self._ensure_trt_engine(inputs.shape, inputs.dtype)
            feats = self._run_feats(inputs)
            head = getattr(self.model, "head", None)
            if head is not None and hasattr(head, "predict") and datas is not None:
                with torch.inference_mode():
                    preds = head.predict(feats, datas)
                return preds
        except Exception:
            pass
        # Fallback to model
        with torch.no_grad():
            try:
                if self._fallback_test_step is not None:
                    return self._fallback_test_step(proc_inputs)
                return self.model.test_step(proc_inputs)
            except Exception:
                try:
                    return self.model(**proc_inputs)
                except Exception:
                    return []
