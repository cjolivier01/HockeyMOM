"""High-performance Torch-TensorRT compilation helpers.

The application only compiles fixed-shape inference workloads.  Keeping the
Torch-TensorRT integration here makes the performance and cache settings
consistent between detector and pose acceleration without importing the
optional dependency during normal PyTorch execution.
"""

from __future__ import annotations

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
    try:
        return importlib.import_module("torch_tensorrt")
    except Exception as exc:
        python_compatibility = (
            " Torch-TensorRT 2.11 and TensorRT 10.15 support Python 3.10-3.13; "
            "use a Python 3.13 environment instead of Python 3.14."
            if sys.version_info >= (3, 14)
            else ""
        )
        raise TorchTensorRTUnavailableError(
            "Torch-TensorRT is required for TensorRT model acceleration. "
            "Install the Torch-TensorRT release matching the installed PyTorch "
            f"and CUDA versions.{python_compatibility}"
        ) from exc


def engine_cache_directory(engine_path: str | Path) -> Path:
    """Map the legacy engine filename setting to a Torch-TensorRT cache directory."""
    path = Path(engine_path).expanduser()
    return path.parent / f"{path.name}.torch_tensorrt_cache"


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

    torch_tensorrt = load_torch_tensorrt()
    cache_dir = engine_cache_directory(engine_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

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

    compiled = torch_tensorrt.compile(
        module.eval(),
        ir="dynamo",
        inputs=input_specs,
        enabled_precisions={precision},
        use_explicit_typing=not fp16,
        require_full_compilation=True,
        pass_through_build_failures=True,
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

    # Static output shapes allow the runtime to overlap output-buffer setup
    # with inference.  Keep the controller alive with the compiled module.
    preallocation = torch_tensorrt.runtime.enable_pre_allocated_outputs(compiled)
    preallocation.set_pre_allocated_output(True)
    setattr(compiled, "_hm_preallocated_outputs", preallocation)
    return compiled
