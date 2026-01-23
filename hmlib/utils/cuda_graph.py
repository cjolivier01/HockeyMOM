from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from hmlib.log import logger


def _same_tensor_signature(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.shape == b.shape
        and a.dtype == b.dtype
        and a.device == b.device
    )


@dataclass
class CudaGraphStats:
    captures: int = 0
    replays: int = 0
    last_signature: Optional[Tuple[Tuple[int, ...], torch.dtype, torch.device]] = None


class CudaGraphCallable:
    """Capture and replay a CUDA graph for a tensor-only callable.

    The wrapped callable must:
      - Run entirely on CUDA
      - Return a Tensor or a tuple/list of Tensors
      - Be shape-stable for a given input signature

    On signature changes (shape/dtype/device), the graph is recaptured.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        example_inputs: Sequence[torch.Tensor],
        *,
        warmup: int = 3,
        name: str = "cuda_graph",
    ) -> None:
        self._fn = fn
        self._warmup = int(max(0, warmup))
        self._name = str(name)

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._pool: Optional[torch.cuda.graphs.graph_pool_handle] = None
        self._static_inputs: Tuple[torch.Tensor, ...] = ()
        self._static_outputs: Any = None
        self.stats = CudaGraphStats()

        self._capture(example_inputs=tuple(example_inputs))

    def _capture(self, example_inputs: Tuple[torch.Tensor, ...]) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CudaGraphCallable")
        if not example_inputs:
            raise ValueError("CudaGraphCallable requires at least one input tensor")
        if any(
            (not isinstance(t, torch.Tensor) or t.device.type != "cuda") for t in example_inputs
        ):
            raise ValueError("All example inputs must be CUDA tensors")

        signature = (
            tuple(example_inputs[0].shape),
            example_inputs[0].dtype,
            example_inputs[0].device,
        )
        self.stats.last_signature = signature

        device = example_inputs[0].device
        torch.cuda.synchronize(device)
        # CUDA graphs must be captured on a non-default stream in recent PyTorch versions.
        capture_stream = torch.cuda.Stream(device=device)

        self._static_inputs = tuple(torch.empty_like(t) for t in example_inputs)
        with torch.cuda.stream(capture_stream):
            for s, t in zip(self._static_inputs, example_inputs):
                s.copy_(t)

            # Warmup to populate caches (cuDNN/autotune, etc.).
            for _ in range(self._warmup):
                _ = self._fn(*self._static_inputs)

            capture_stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graphs.graph_pool_handle()
            try:
                with torch.cuda.graph(
                    graph, pool=pool, stream=capture_stream, capture_error_mode="relaxed"
                ):
                    self._static_outputs = self._fn(*self._static_inputs)
            except Exception as ex:
                logger.warning("Failed to capture %s CUDA graph: %s", self._name, ex)
                raise
            capture_stream.synchronize()

        self._graph = graph
        self._pool = pool
        self.stats.captures += 1

    def _ensure_signature(self, inputs: Tuple[torch.Tensor, ...]) -> None:
        if not inputs:
            raise ValueError("CudaGraphCallable requires at least one input tensor")
        if self._graph is None:
            self._capture(example_inputs=inputs)
            return
        if len(inputs) != len(self._static_inputs):
            self._capture(example_inputs=inputs)
            return
        for a, b in zip(inputs, self._static_inputs):
            if not _same_tensor_signature(a, b):
                self._capture(example_inputs=inputs)
                return

    def __call__(self, *inputs: torch.Tensor):
        inp = tuple(inputs)
        self._ensure_signature(inp)
        for s, t in zip(self._static_inputs, inp):
            s.copy_(t)
        assert self._graph is not None
        self._graph.replay()
        self.stats.replays += 1
        return self._static_outputs
