from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from hmlib.log import logger

_CAPTURE_STREAMS: Dict[Tuple[str, int], torch.cuda.Stream] = {}
_CAPTURE_LOCKS: Dict[Tuple[str, int], threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _device_key(device: torch.device) -> Tuple[str, int]:
    if device.type != "cuda":
        raise ValueError(f"CUDA graph capture requires a CUDA device, got {device!r}")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return (device.type, int(index))


def _get_shared_capture_stream(device: torch.device) -> torch.cuda.Stream:
    key = _device_key(device)
    with _REGISTRY_LOCK:
        stream = _CAPTURE_STREAMS.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=torch.device(key[0], key[1]))
            _CAPTURE_STREAMS[key] = stream
        return stream


def _get_capture_lock(device: torch.device) -> threading.Lock:
    key = _device_key(device)
    with _REGISTRY_LOCK:
        lock = _CAPTURE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CAPTURE_LOCKS[key] = lock
        return lock


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
        self._capture_stream: Optional[torch.cuda.Stream] = None
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
        capture_stream = _get_shared_capture_stream(device)
        capture_lock = _get_capture_lock(device)
        self._capture_stream = capture_stream

        with capture_lock:
            torch.cuda.synchronize(device)
            self._static_inputs = tuple(torch.empty_like(t) for t in example_inputs)
            with torch.cuda.stream(capture_stream):
                with torch.inference_mode():
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
                            graph,
                            pool=pool,
                            stream=capture_stream,
                            capture_error_mode="thread_local",
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
        with torch.inference_mode():
            for s, t in zip(self._static_inputs, inp):
                s.copy_(t)
            assert self._graph is not None
            self._graph.replay()
        self.stats.replays += 1
        return self._static_outputs
