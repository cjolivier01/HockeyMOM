"""High-performance Torch-TensorRT compilation helpers.

The application only compiles fixed-shape inference workloads.  Keeping the
Torch-TensorRT integration here makes the performance and cache settings
consistent between detector and pose acceleration without importing the
optional dependency during normal PyTorch execution.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from hmlib.log import get_logger

logger = get_logger(__name__)


class TorchTensorRTUnavailableError(RuntimeError):
    """Raised when the optional Torch-TensorRT runtime cannot be imported."""


class TorchTensorRTConfigurationError(ValueError):
    """Raised when a legacy option cannot be represented safely."""


def load_torch_tensorrt() -> Any:
    """Load Torch-TensorRT lazily and provide an actionable import error."""
    if sys.version_info >= (3, 14):
        raise TorchTensorRTUnavailableError(
            "Torch-TensorRT 2.11 and TensorRT 10.15 provide official wheels for "
            "Python 3.10-3.13 only; use a Python 3.13 environment for TensorRT "
            "model acceleration."
        )
    try:
        return importlib.import_module("torch_tensorrt")
    except Exception as exc:
        raise TorchTensorRTUnavailableError(
            "Torch-TensorRT is required for TensorRT model acceleration. "
            "Install the Torch-TensorRT release matching the installed PyTorch "
            "and CUDA versions."
        ) from exc


def engine_cache_directory(engine_path: str | Path) -> Path:
    """Map the legacy engine filename setting to a Torch-TensorRT cache directory."""
    path = Path(engine_path).expanduser()
    return path.parent / f"{path.name}.torch_tensorrt_cache"


def _cache_fingerprint(
    module: torch.nn.Module,
    torch_tensorrt: Any,
    device: torch.device,
) -> str:
    """Partition serialized engines by model structure and runtime compatibility."""
    tensorrt = importlib.import_module("tensorrt")
    properties = torch.cuda.get_device_properties(device)
    parts = [
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"torch_tensorrt={getattr(torch_tensorrt, '__version__', 'unknown')}",
        f"tensorrt={getattr(tensorrt, '__version__', 'unknown')}",
        f"gpu={properties.name}",
        f"compute={properties.major}.{properties.minor}",
    ]
    for name, value in list(module.named_parameters()) + list(module.named_buffers()):
        parts.append(f"{name}:{tuple(value.shape)}:{value.dtype}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20]


def _validate_cuda_device(
    module: torch.nn.Module,
    example_inputs: Sequence[torch.Tensor],
) -> torch.device:
    input_device = example_inputs[0].device
    if input_device.type != "cuda":
        raise TorchTensorRTConfigurationError("Torch-TensorRT example inputs must be CUDA tensors.")
    if any(value.device != input_device for value in example_inputs):
        raise TorchTensorRTConfigurationError(
            "Torch-TensorRT example inputs must all use the same CUDA device."
        )
    module_devices = {
        value.device
        for value in list(module.parameters()) + list(module.buffers())
        if value.device.type != "meta"
    }
    if any(device != input_device for device in module_devices):
        raise TorchTensorRTConfigurationError(
            "Torch-TensorRT module parameters, buffers, and inputs must use the same "
            f"CUDA device ({input_device})."
        )
    return input_device


def compile_torch_tensorrt(
    module: torch.nn.Module,
    example_inputs: Sequence[torch.Tensor],
    *,
    engine_path: str | Path,
    force_build: bool = False,
    fp16: bool = True,
    int8: bool = False,
    workspace_size: int = 4 << 30,
) -> torch.nn.Module:
    """Compile a fully static module with runtime-performance-first settings.

    Torch-TensorRT's disk engine cache keys the graph, input specifications,
    device, and compilation settings. Cached refittable engines are populated
    with the current module weights when loaded. The user-facing
    ``engine_path`` therefore acts as a stable cache namespace rather than a
    single serialized engine file.
    """
    if int8:
        raise TorchTensorRTConfigurationError(
            "The legacy TensorRT INT8 flag is unsupported by this migration. "
            "Torch-TensorRT Dynamo INT8 requires an integrated Q/DQ-quantized "
            "model (for example, one prepared with NVIDIA Model Optimizer), "
            "which this path does not yet provide; use FP16 instead."
        )
    if not example_inputs:
        raise TorchTensorRTConfigurationError(
            "At least one example input is required for static TensorRT compilation."
        )
    if any(not isinstance(value, torch.Tensor) for value in example_inputs):
        raise TorchTensorRTConfigurationError(
            "Torch-TensorRT example inputs must all be torch.Tensor instances."
        )

    input_device = _validate_cuda_device(module, example_inputs)

    torch_tensorrt = load_torch_tensorrt()
    cache_dir = engine_cache_directory(engine_path) / _cache_fingerprint(
        module, torch_tensorrt, input_device
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_had_entries = any(cache_dir.iterdir())

    input_specs = [
        torch_tensorrt.Input(
            shape=tuple(value.shape),
            dtype=value.dtype,
            format=torch.contiguous_format,
        )
        for value in example_inputs
    ]
    precision = torch.float16 if fp16 else torch.float32
    logger.info(
        "%s Torch-TensorRT engine for %s input(s); cache=%s",
        "Building" if force_build else "Compiling or reusing cached",
        len(input_specs),
        cache_dir,
    )

    compile_options = dict(
        ir="dynamo",
        inputs=input_specs,
        device=input_device,
        enabled_precisions={precision},
        use_explicit_typing=not fp16,
        require_full_compilation=True,
        pass_through_build_failures=True,
        # Torch-TensorRT 2.11 applies its default threshold of five supported
        # ops before require_full_compilation, so small graphs need this to
        # avoid being returned as uncompiled PyTorch modules.
        min_block_size=1,
        use_fast_partitioner=False,
        optimization_level=5,
        num_avg_timing_iters=8,
        tiling_optimization_level="full",
        workspace_size=int(workspace_size),
        # Torch-TensorRT's cache reloads engines by refitting the current model
        # weights. This avoids both stale-checkpoint engines and repeat builds.
        immutable_weights=False,
        cache_built_engines=True,
        reuse_cached_engines=not force_build,
        engine_cache_dir=str(cache_dir),
    )
    with torch.cuda.device(input_device):
        try:
            compiled = torch_tensorrt.compile(module.eval(), **compile_options)
        except Exception as exc:
            if force_build or not cache_had_entries:
                raise
            logger.warning(
                "Cached Torch-TensorRT engine failed to load; rebuilding it once: %s",
                exc,
            )
            compile_options["reuse_cached_engines"] = False
            compiled = torch_tensorrt.compile(module.eval(), **compile_options)

    # Static output shapes allow the runtime to overlap output-buffer setup
    # with inference.  Keep the controller alive with the compiled module.
    preallocation = torch_tensorrt.runtime.enable_pre_allocated_outputs(compiled)
    preallocation.set_pre_allocated_output(True)
    setattr(compiled, "_hm_preallocated_outputs", preallocation)
    return compiled
