import argparse
from types import SimpleNamespace

import pytest
import torch

from hmlib.utils import torch_tensorrt_runtime


class _PreallocationController:
    def __init__(self) -> None:
        self.enabled = False

    def set_pre_allocated_output(self, enabled: bool) -> None:
        self.enabled = enabled


class _FakeTorchTensorRT:
    def __init__(self) -> None:
        self.input_specs = []
        self.compile_calls = []
        self.preallocation = _PreallocationController()
        self.runtime = SimpleNamespace(
            enable_pre_allocated_outputs=lambda _module: self.preallocation
        )

    def Input(self, **kwargs):
        self.input_specs.append(kwargs)
        return kwargs

    def compile(self, module, **kwargs):
        self.compile_calls.append((module, kwargs))
        return torch.nn.Identity()


def should_compile_static_fp16_with_performance_settings(monkeypatch, tmp_path) -> None:
    fake = _FakeTorchTensorRT()
    monkeypatch.setattr(
        torch_tensorrt_runtime.importlib,
        "import_module",
        lambda name: fake,
    )
    model = torch.nn.Conv2d(3, 8, kernel_size=3).eval()
    sample = torch.randn(1, 3, 32, 48)

    compiled = torch_tensorrt_runtime.compile_torch_tensorrt(
        model,
        [sample],
        engine_path=tmp_path / "detector.engine",
    )

    assert isinstance(compiled, torch.nn.Identity)
    assert fake.input_specs == [
        {
            "shape": (1, 3, 32, 48),
            "dtype": torch.float32,
            "format": torch.contiguous_format,
        }
    ]
    _, options = fake.compile_calls[0]
    assert options["ir"] == "dynamo"
    assert options["enabled_precisions"] == {torch.float16}
    assert options["use_explicit_typing"] is False
    assert options["require_full_compilation"] is True
    assert options["pass_through_build_failures"] is True
    assert options["min_block_size"] == 1
    assert options["use_fast_partitioner"] is False
    assert options["optimization_level"] == 5
    assert options["num_avg_timing_iters"] == 8
    assert options["tiling_optimization_level"] == "full"
    assert options["workspace_size"] == 4 << 30
    assert options["immutable_weights"] is False
    assert options["cache_built_engines"] is True
    assert options["reuse_cached_engines"] is True
    assert options["engine_cache_dir"] == str(tmp_path / "detector.engine.torch_tensorrt_cache")
    assert fake.preallocation.enabled is True


def should_force_fp32_rebuild_without_reusing_cached_engine(monkeypatch, tmp_path) -> None:
    fake = _FakeTorchTensorRT()
    monkeypatch.setattr(
        torch_tensorrt_runtime.importlib,
        "import_module",
        lambda name: fake,
    )

    torch_tensorrt_runtime.compile_torch_tensorrt(
        torch.nn.Identity(),
        [torch.randn(2, 4)],
        engine_path=tmp_path / "pose.engine",
        force_build=True,
        fp16=False,
    )

    _, options = fake.compile_calls[0]
    assert options["enabled_precisions"] == {torch.float32}
    assert options["use_explicit_typing"] is True
    assert options["reuse_cached_engines"] is False


def should_reject_legacy_int8_calibration_before_import(monkeypatch, tmp_path) -> None:
    imported = False

    def _unexpected_import(_name):
        nonlocal imported
        imported = True

    monkeypatch.setattr(
        torch_tensorrt_runtime.importlib,
        "import_module",
        _unexpected_import,
    )

    with pytest.raises(
        torch_tensorrt_runtime.TorchTensorRTConfigurationError,
        match="Q/DQ-quantized model",
    ):
        torch_tensorrt_runtime.compile_torch_tensorrt(
            torch.nn.Identity(),
            [torch.randn(1, 4)],
            engine_path=tmp_path / "detector.engine",
            int8=True,
        )

    assert imported is False


def should_surface_actionable_optional_dependency_error(monkeypatch) -> None:
    def _missing(_name):
        raise ModuleNotFoundError("No module named 'torch_tensorrt'")

    monkeypatch.setattr(
        torch_tensorrt_runtime.importlib,
        "import_module",
        _missing,
    )

    with pytest.raises(
        torch_tensorrt_runtime.TorchTensorRTUnavailableError,
        match="release matching the installed PyTorch and CUDA versions",
    ):
        torch_tensorrt_runtime.load_torch_tensorrt()


def should_default_tensor_rt_models_to_fp16() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())

    defaults = parser.parse_args([])
    fp32 = parser.parse_args(["--no-detector-trt-fp16", "--no-pose-trt-fp16"])

    assert defaults.detector_trt_fp16 is True
    assert defaults.pose_trt_fp16 is True
    assert fp32.detector_trt_fp16 is False
    assert fp32.pose_trt_fp16 is False


def should_cache_pose_engines_per_static_shape(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import pose_factory_plugin

    compiled_modules = []

    def _compile(*_args, **_kwargs):
        compiled = torch.nn.Identity()
        compiled_modules.append(compiled)
        return compiled

    monkeypatch.setattr(pose_factory_plugin, "compile_torch_tensorrt", _compile)

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 3, kernel_size=1)

    runner = pose_factory_plugin._TrtPoseRunner(_Model(), engine_path=str(tmp_path / "pose.engine"))
    batch_one = torch.randn(1, 3, 16, 16)
    batch_two = torch.randn(2, 3, 16, 16)

    runner._ensure_trt_engine(batch_one)
    first_module = runner._trt_module
    runner._ensure_trt_engine(batch_two)
    runner._ensure_trt_engine(batch_one)

    assert len(compiled_modules) == 2
    assert runner._trt_module is first_module


def should_cache_detector_engines_per_static_shape(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import detector_factory_plugin

    compiled_modules = []

    def _compile(*_args, **_kwargs):
        compiled = torch.nn.Identity()
        compiled_modules.append(compiled)
        return compiled

    monkeypatch.setattr(detector_factory_plugin, "compile_torch_tensorrt", _compile)

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 3, kernel_size=1)
            self.neck = torch.nn.Identity()
            self.bbox_head = torch.nn.Identity()

    runner = detector_factory_plugin._TrtDetectorWrapper(
        _Model(),
        engine_path=str(tmp_path / "detector.engine"),
        nms_backend="torchvision",
    )
    size_16 = torch.randn(1, 3, 16, 16)
    size_24 = torch.randn(1, 3, 24, 24)

    runner._ensure_trt_engine(size_16)
    first_module = runner._trt_module
    runner._ensure_trt_engine(size_24)
    runner._ensure_trt_engine(size_16)

    assert len(compiled_modules) == 2
    assert runner._trt_module is first_module


def should_surface_pose_compile_failure_and_use_pytorch_split_path(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import pose_factory_plugin

    def _compile(*_args, **_kwargs):
        raise RuntimeError("unsupported TensorRT operator")

    monkeypatch.setattr(pose_factory_plugin, "compile_torch_tensorrt", _compile)

    class _Head(torch.nn.Module):
        def predict(self, feats, data_samples):
            return [(feats, data_samples)]

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 3, kernel_size=1)
            self.head = _Head()

    runner = pose_factory_plugin._TrtPoseRunner(_Model(), engine_path=str(tmp_path / "pose.engine"))
    result = runner.forward(
        {
            "inputs": torch.randn(1, 3, 16, 16),
            "data_samples": [object()],
        }
    )

    assert runner._trt_disabled_reason == "unsupported TensorRT operator"
    assert runner._trt_module is None
    assert len(result) == 1
