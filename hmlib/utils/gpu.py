import ctypes
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from hmlib.log import logger


@contextmanager
def cuda_stream_scope(stream: Union[torch.cuda.Stream, None]):
    """Context manager that activates a CUDA stream if provided.

    @param stream: CUDA stream instance or ``None``.
    @return: Context manager yielding the active stream or ``None``.
    """
    if stream is None:
        # If the resource is None, yield nothing but still enter the with block
        yield None
    else:
        # If the resource is not None, use it as a normal context manager
        with torch.cuda.stream(stream) as r:
            yield r


def record_stream_event(
    tensor: torch.Tensor, stream: Optional[torch.cuda.Stream] = None
) -> Optional[torch.cuda.Event]:
    """Record a blocking CUDA event on the given tensor's stream.

    @param tensor: Tensor whose device determines the CUDA stream context.
    @param stream: Optional explicit stream; defaults to the current stream on the tensor device.
    @return: Created :class:`torch.cuda.Event` or ``None`` when the tensor is on CPU.
    """
    assert tensor is not None
    if tensor.device.type != "cuda":
        return None
    if stream is None:
        stream = torch.cuda.current_stream(tensor.device)
    event = torch.cuda.Event(blocking=False)
    stream.record_event(event)
    return event


class GpuAllocator:
    """Helper for allocating GPUs based on compute capability, cores, or memory.

    The allocator tracks assigned devices and exposes strategies such as
    :meth:`allocate_modern`, :meth:`allocate_fast` and :meth:`get_largest_mem_gpu`.

    @param gpus: Comma-separated string or list of GPU indices allowed for allocation.
    @see @ref hmlib.utils.progress_bar.ProgressBar "ProgressBar" for CLI integration.
    """

    def __init__(self, gpus: str | List[int]):
        if gpus is None:
            gpus = [i for i in range(torch.cuda.device_count())]
        elif isinstance(gpus, str):
            gpus = gpus.split(",")
        if hasattr(gpus, "__iter__"):
            gpus = [int(i) for i in gpus]
        gpu_count = min(torch.cuda.device_count(), len(gpus))
        self._gpus = gpus[: gpu_count + 1]
        self._used_gpus: Dict = dict()
        self._named_allocations: Dict = dict()
        self._last_allocated: int = None
        self._is_single_lowmem_gpu: Union[bool, None] = None

        self._all_gpu_info = get_gpu_capabilities()
        for i, gpu_info in enumerate(self._all_gpu_info):
            logger.info("GPU %d: %s", i, gpu_info["name"])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._all_gpu_info[index]

    def __len__(self) -> int:
        return len(self._all_gpu_info)

    def allocate_modern(self, name: Optional[Union[str, None]] = None):
        """
        Allocate GPU with highest compute capability
        """
        if name and name in self._named_allocations:
            # Name overrides modern/fast
            return self._named_allocations[name]
        index, caps = get_gpu_with_highest_compute_capability(
            allowed_gpus=self._gpus, disallowed_gpus=self._used_gpus
        )
        if index is not None:
            assert index not in self._used_gpus
            self._used_gpus[index] = caps
            if name:
                self._named_allocations[name] = index
            self._last_allocated = index
            return index
        else:
            return self._last_allocated

    def get_modern(self, name: Optional[Union[str, None]] = None):
        """
        Allocate GPU with highest compute capability
        """
        if name and name in self._named_allocations:
            # Name overrides modern/fast
            return self._named_allocations[name]
        index, caps = get_gpu_with_highest_compute_capability(allowed_gpus=self._gpus)
        if index is not None:
            return index
        else:
            return self._last_allocated

    def allocate_fast(self, name: Optional[Union[str, None]] = None):
        """
        Allocate GPU with the most multiprocessing cores
        """
        if name and name in self._named_allocations:
            # Name overrides modern/fast
            return self._named_allocations[name]
        index, caps = get_gpu_with_most_multiprocessors(
            allowed_gpus=self._gpus, disallowed_gpus=self._used_gpus
        )
        if index is not None:
            assert index not in self._used_gpus
            self._used_gpus[index] = caps
            if name:
                self._named_allocations[name] = index
            self._last_allocated = index
            return index
        else:
            return self._last_allocated

    def free_count(self):
        """
        How many GPU's remain to be allocated?
        """
        return len(self._gpus) - len(self._used_gpus)

    def get_largest_mem_gpu(self, name: Optional[Union[str, None]] = None):
        """
        Allocate GPU with highest compute capability
        """
        if name and name in self._named_allocations:
            # Name overrides modern/fast
            return self._named_allocations[name]
        index, caps = get_gpu_with_largest_mem(allowed_gpus=self._gpus)
        if index is not None:
            return index
        else:
            return self._last_allocated

    def is_single_lowmem_gpu(self, low_threshold_mb: int = 8192) -> bool:
        """
        Return True if we are dealing with a single, low-memory GPU
        """
        if torch.cuda.device_count() != 1:
            # Assume zero is not a relevant use-case
            return False
        if self._is_single_lowmem_gpu is None:
            index, caps = get_gpu_with_largest_mem(allowed_gpus=self._gpus)
            if index is None:
                # No allowed GPUs, so we aren't under any
                # GPU memory constraint
                self._is_single_lowmem_gpu = False
            else:
                self._is_single_lowmem_gpu = float(caps["total_memory"]) * 1024 <= low_threshold_mb
        return self._is_single_lowmem_gpu


class StreamTensorBase:
    """Common interface for tensor-like wrappers that manage CUDA streams."""

    def ref(self) -> torch.Tensor:  # pragma: no cover - interface definition
        raise NotImplementedError

    def wait(self, new_stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        raise NotImplementedError

    def get(self) -> torch.Tensor:  # pragma: no cover - interface definition
        raise NotImplementedError

    def size(self, index: int) -> int:
        return self.ref().size(index)


class StreamTensorX(StreamTensorBase):
    """Enhanced tensor wrapper with explicit CUDA stream + event tracking."""

    __slots__ = (
        "_tensor",
        "_event",
        "_stream",
        "_print_thresh",
        "_sync_duration",
        "_verbose",
    )

    _SAFE_ATTRS = {
        "dtype",
        "device",
        "layout",
        "ndim",
        "dim",
        "ndimension",
        "numel",
        "names",
        "requires_grad",
        "grad",
        "grad_fn",
        "is_leaf",
        "is_cuda",
        "is_contiguous",
        "is_complex",
        "is_floating_point",
        "is_sparse",
        "qscheme",
        "strides",
        "stride",
        "storage_offset",
        "T",
        "mT",
        "real",
        "imag",
    }

    __array_priority__ = 1000

    def __init__(
        self,
        tensor: Union[torch.Tensor, "StreamTensorBase"],
        *,
        auto_checkpoint: bool = True,
        verbose: bool = True,
        print_thresh: float = 0.001,
    ):
        if isinstance(tensor, StreamTensorBase):
            tensor = tensor.wait()
        assert isinstance(tensor, torch.Tensor)
        self._tensor = tensor
        self._event: Optional[torch.cuda.Event] = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._print_thresh = print_thresh
        self._sync_duration: Optional[float] = None
        self._verbose = verbose
        if auto_checkpoint:
            self.checkpoint()

    def __repr__(self) -> str:
        state = "ready" if self.ready() else "pending"
        return f"StreamTensorX(shape={tuple(self.shape)}, device={self.device}, {state})"

    # ------------------------------------------------------------------
    # Synchronization helpers
    # ------------------------------------------------------------------
    def _clear_tracking(self) -> None:
        self._event = None
        self._stream = None

    def _ensure_ready_for_stream(
        self, target_stream: Optional[torch.cuda.Stream], clear: bool = False
    ) -> torch.Tensor:
        if self._tensor.device.type != "cuda":
            return self._tensor
        if self._event is None:
            return self._tensor
        if target_stream is None:
            target_stream = torch.cuda.current_stream(self._tensor.device)
        target_stream.wait_event(self._event)
        if clear:
            self._clear_tracking()
        return self._tensor

    def _ensure_ready_blocking(self) -> torch.Tensor:
        if self._tensor.device.type != "cuda" or self._event is None:
            return self._tensor
        start = time.time()
        self._event.synchronize()
        self._sync_duration = time.time() - start
        self._clear_tracking()
        if (
            self._verbose
            and self._sync_duration is not None
            and self._sync_duration > self._print_thresh
        ):
            logger.info(
                f"Syncing tensor with shape {tuple(self.shape)} took {self._sync_duration * 1000:.3f} ms"
            )
        return self._tensor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def sync_duration(self) -> float:
        return -1 if self._sync_duration is None else self._sync_duration

    @property
    def stream(self) -> Optional[torch.cuda.Stream]:
        return self._stream

    @property
    def shape(self):  # type: ignore[override]
        return self._tensor.shape

    @property
    def device(self):
        return self._tensor.device

    def ref(self) -> torch.Tensor:
        return self._tensor

    def ready(self) -> bool:
        if self._tensor.device.type != "cuda":
            return True
        return self._event is None

    def checkpoint(self, stream: Optional[torch.cuda.Stream] = None) -> Optional[torch.cuda.Event]:
        if self._tensor.device.type != "cuda":
            self._clear_tracking()
            return None
        if stream is None:
            stream = torch.cuda.current_stream(self._tensor.device)
        prev_event = self._event
        if prev_event is not None:
            stream.wait_event(prev_event)
        self._event = record_stream_event(self._tensor, stream=stream)
        self._stream = stream
        return self._event

    def wait(self, new_stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        return self._ensure_ready_for_stream(new_stream)

    def get(self) -> torch.Tensor:
        return self._ensure_ready_blocking()

    def cpu(self) -> torch.Tensor:
        return self.get().cpu()

    def to(self, *args, **kwargs):
        return self.get().to(*args, **kwargs)

    def numpy(self) -> np.ndarray:
        return self.cpu().numpy()

    def __len__(self) -> int:
        return self._tensor.shape[0]

    def __getitem__(self, item):
        tensor = self._ensure_ready_for_stream(None)
        return tensor.__getitem__(item)

    def permute(self, *args, **kwargs):
        tensor = self._ensure_ready_for_stream(None)
        self._tensor = tensor.permute(*args, **kwargs)
        self._clear_tracking()
        return self

    def unsqueeze(self, *args, **kwargs):
        tensor = self._ensure_ready_for_stream(None)
        self._tensor = tensor.unsqueeze(*args, **kwargs)
        self._clear_tracking()
        return self

    def squeeze(self, *args, **kwargs):
        tensor = self._ensure_ready_for_stream(None)
        self._tensor = tensor.squeeze(*args, **kwargs)
        self._clear_tracking()
        return self

    def __getattr__(self, name: str):
        if name in self._SAFE_ATTRS:
            return getattr(self._tensor, name)
        raise AttributeError(f"StreamTensorX has no attribute '{name}'")


class StreamTensor(StreamTensorX):
    """Default tensor wrapper that immediately checkpoints on creation."""

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        verbose: bool = True,
        print_thresh: float = 0.001,
    ):
        super().__init__(
            tensor=tensor,
            auto_checkpoint=True,
            verbose=verbose,
            print_thresh=print_thresh,
        )


class StreamCheckpoint(StreamTensorX):
    """One-shot checkpoint that immediately records a CUDA event."""

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        verbose: bool = True,
        print_thresh: float = 0.001,
    ):
        super().__init__(
            tensor=tensor,
            auto_checkpoint=True,
            verbose=verbose,
            print_thresh=print_thresh,
        )


def unwrap_tensor(
    tensor: Union[torch.Tensor, StreamTensorBase],
    current_stream: Optional[torch.cuda.Stream] = None,
    verbose: Optional[bool] = None,
) -> torch.Tensor:
    if isinstance(tensor, StreamTensorBase):
        if verbose is not None:
            tensor.verbose = verbose
        return tensor.wait(current_stream)
    return tensor


def wrap_tensor(
    tensor: Union[torch.Tensor, StreamTensorBase], verbose: bool = True
) -> Union[torch.Tensor, StreamTensorBase]:
    if not tensor.is_cuda:
        return tensor
    if isinstance(tensor, StreamTensorBase):
        tensor = unwrap_tensor(tensor)
    return StreamTensorX(tensor)


def copy_gpu_to_gpu_async(
    tensor: torch.Tensor,
    dest_device: torch.device,
    src_stream: torch.cuda.Stream = None,
    dest_stream: torch.cuda.Stream = None,
) -> Tuple[torch.Tensor, torch.cuda.Event]:
    if tensor.device == dest_device:
        return tensor, None
    if isinstance(tensor, StreamTensorBase):
        tensor = tensor.wait()
    if src_stream is None:
        src_stream = torch.cuda.current_stream(tensor.device)
    if dest_stream is None:
        dest_stream = torch.cuda.current_stream(dest_device)
    with cuda_stream_scope(src_stream):
        tensor_dest = torch.empty_like(tensor, device=dest_device)
        tensor_dest.copy_(tensor, non_blocking=True)
        src_event = torch.cuda.Event(blocking=True)
        src_stream.record_event(src_event)
    with cuda_stream_scope(src_stream):
        dest_stream.wait_event(src_event)
        dest_event = torch.cuda.Event(blocking=True)
        dest_stream.record_event(dest_event)
    return tensor_dest, dest_event


def get_gpu_capabilities():
    if not torch.cuda.is_available():
        return []
    num_gpus = torch.cuda.device_count()
    gpu_info = []
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)
        gpu_info.append(
            {
                "name": properties.name,
                "index": i,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory": properties.total_memory / (1024**3),  # Convert bytes to GB
                "properties": properties,
            }
        )
    return gpu_info


def get_gpu_with_highest_compute_capability(
    allowed_gpus: Union[List[int], None] = None,
    disallowed_gpus: Union[List[int], Set[int], Dict[int, Any], None] = None,
) -> Tuple[Union[int, None], Union[Dict, None]]:
    gpus = get_gpu_capabilities()
    if gpus is None:
        return None, None
    sorted_gpus = sorted(gpus, key=lambda x: float(x["compute_capability"]))
    for _, gpu in enumerate(reversed(sorted_gpus)):
        index = gpu["index"]
        if allowed_gpus is not None and index not in allowed_gpus:
            continue
        if disallowed_gpus is not None and index in disallowed_gpus:
            continue
        return index, gpu
    return None, None


def get_gpu_with_largest_mem(
    allowed_gpus: Union[List[int], None] = None,
    disallowed_gpus: Union[List[int], Set[int], Dict[int, Any], None] = None,
) -> Tuple[Union[int, None], Union[Dict, None]]:
    gpus = get_gpu_capabilities()
    if gpus is None:
        return None, None
    sorted_gpus = sorted(gpus, key=lambda x: float(x["total_memory"]))
    for _, gpu in enumerate(reversed(sorted_gpus)):
        index = gpu["index"]
        if allowed_gpus is not None and index not in allowed_gpus:
            continue
        if disallowed_gpus is not None and index in disallowed_gpus:
            continue
        return index, gpu
    return None, None


def get_gpu_with_most_multiprocessors(
    allowed_gpus: Union[List[int], None] = None,
    disallowed_gpus: Union[List[int], Set[int], None] = None,
) -> Tuple[Union[int, None], Union[Dict, None]]:
    gpus = get_gpu_capabilities()
    if gpus is None:
        return None, None
    sorted_gpus = sorted(gpus, key=lambda x: float(x["properties"].multi_processor_count))
    candidates = []
    last_count = 0
    for _, gpu in enumerate(reversed(sorted_gpus)):
        index = gpu["index"]
        if allowed_gpus is not None and index not in allowed_gpus:
            continue
        if disallowed_gpus is not None and index in disallowed_gpus:
            continue
        if last_count:
            if gpu["properties"].multi_processor_count == last_count:
                candidates.append(index)
        else:
            last_count = gpu["properties"].multi_processor_count
            candidates.append(index)
    if not candidates:
        return None, None
    candidates = sorted(candidates)
    return candidates[0], gpus[candidates[0]]


def _check_is(dev: Union[torch.device, None], flag: bool):
    if dev is not None:
        assert isinstance(dev, torch.device)
    assert (flag and dev is not None) or (not flag and dev is None)


def select_gpus(
    allowed_gpus: List[int],
    rank: Union[int, None] = None,
    gpu_allocator: Union[GpuAllocator, None] = None,
    inference: bool = True,
    is_stitching: bool = True,
    is_detecting: bool = True,
    is_multipose: bool = False,
    is_encoding: bool = True,
    is_camera: bool = True,
    stitch_with_fastest: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.device], bool]:
    #
    # BEGIN GPU SELECTION
    #
    if gpu_allocator is None:
        gpu_allocator = GpuAllocator(gpus=allowed_gpus)

    stitching_device: Union[torch.device, None] = None
    video_encoding_device: Union[torch.device, None] = None
    detection_device: Union[torch.device, None] = None
    multipose_device: Union[torch.device, None] = None
    camera_device = torch.device("cpu")
    if is_multipose:
        # Multi-Pose gets the biggest, baddest one
        multipose_device = torch.device("cuda", gpu_allocator.allocate_fast())
    if inference:
        if is_multipose:
            if is_detecting:
                detection_device = multipose_device
        else:
            if stitch_with_fastest:
                if is_stitching:
                    if gpu_allocator.free_count() or not is_detecting:
                        stitching_device = torch.device("cuda", gpu_allocator.allocate_fast())
                    else:
                        stitching_device = detection_device

            if is_detecting:
                if gpu_allocator.free_count():
                    detection_device = torch.device("cuda", gpu_allocator.allocate_fast())
                else:
                    if is_multipose:
                        assert multipose_device is not None
                        detection_device = multipose_device
                    elif detection_device is None:
                        detection_device = stitching_device
                        assert detection_device is not None

        if is_encoding:
            # Always used most modern GPU for encoding
            video_encoding_device = torch.device("cuda", gpu_allocator.get_modern())
            # if gpu_allocator.free_count():
            #     video_encoding_device = torch.device("cuda", gpu_allocator.allocate_modern())
            # else:
            #     video_encoding_device = (
            #         detection_device if not multipose_device else multipose_device
            #     )
            # if video_encoding_device is None:
            #     video_encoding_device = torch.device("cuda", gpu_allocator.allocate_modern())

        if is_stitching and stitching_device is None:
            if gpu_allocator.free_count():
                stitching_device = torch.device("cuda", gpu_allocator.allocate_fast())
            else:
                stitching_device = detection_device
    else:
        if rank is not None:
            for i in range(rank):
                _ = gpu_allocator.allocate_fast()
            detection_device = torch.device("cuda", gpu_allocator.allocate_fast())
        if is_stitching and stitching_device is None:
            stitching_device = detection_device

    if False:
        # just always have encoder be detection device
        if video_encoding_device is not None and detection_device is not None:
            video_encoding_device = detection_device

    #
    # END GPU SELECTION
    #
    gpus = dict()
    _check_is(stitching_device, is_stitching)
    _check_is(detection_device, is_detecting)
    _check_is(multipose_device, is_multipose)
    _check_is(stitching_device, is_stitching)
    _check_is(camera_device, is_camera)
    _check_is(video_encoding_device, is_encoding)
    if detection_device is not None:
        gpus["detection"] = detection_device
    if stitching_device is not None:
        gpus["stitching"] = stitching_device
    if multipose_device is not None:
        gpus["multipose"] = multipose_device
    if camera_device is not None:
        gpus["camera"] = camera_device
    if video_encoding_device is not None:
        gpus["encoder"] = video_encoding_device

    if verbose:
        for k in sorted(gpus.keys()):
            device = gpus[k]
            if device.type != "cuda":
                continue
            name = gpu_allocator[device.index]["name"]
            logger.info("%s device: %s (%s)", k, device, name)

    return gpus, gpu_allocator.is_single_lowmem_gpu(), gpu_allocator


# cudaStreamNonBlocking (per CUDA docs this is 1)
CUDA_STREAM_NON_BLOCKING = 0x01


def get_external_torch_stream(device=None, flags=CUDA_STREAM_NON_BLOCKING):
    """
    Create a CUDA stream with cudaStreamCreateWithFlags and wrap it in
    a torch.cuda.ExternalStream (subclass of torch.cuda.Stream).

    The CUDA runtime (libcudart) and function pointers are initialized
    only on the first call.

    Parameters
    ----------
    device : int or torch.device or None
        CUDA device index. Defaults to torch.cuda.current_device().
    flags : int
        cudaStreamCreateWithFlags flags (default: cudaStreamNonBlocking).

    Returns
    -------
    torch.cuda.Stream
        A PyTorch stream object backed by the external cudaStream_t.

    Notes
    -----
    torch.cuda.ExternalStream does NOT manage the lifetime of the
    underlying CUDA stream; you're responsible for eventually
    destroying it if you create many streams. :contentReference[oaicite:1]{index=1}
    """
    # --- lazy init libcudart + function pointers (first call only) ---
    if not hasattr(get_external_torch_stream, "_initialized"):
        if os.name == "nt":
            libnames = [
                "cudart64_12.dll",
                "cudart64_11.dll",
                "cudart64_102.dll",
                "cudart64_101.dll",
                "cudart64_100.dll",
            ]
        elif sys.platform == "darwin":
            libnames = [
                "libcudart.dylib",
                "libcudart.12.dylib",
                "libcudart.11.dylib",
            ]
        else:  # Linux / Unix
            libnames = [
                "libcudart.so",
                "libcudart.so.12",
                "libcudart.so.11",
                "libcudart.so.10.2",
                "libcudart.so.10.1",
            ]

        libcudart = None
        last_err = None
        for name in libnames:
            try:
                libcudart = ctypes.CDLL(name)
                break
            except OSError as e:
                last_err = e

        if libcudart is None:
            raise OSError(f"Could not load CUDA runtime library. Tried: {libnames}") from last_err

        cudaStream_t = ctypes.c_void_p

        cudaStreamCreateWithFlags = libcudart.cudaStreamCreateWithFlags
        cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(cudaStream_t),
            ctypes.c_uint,
        ]
        cudaStreamCreateWithFlags.restype = ctypes.c_int

        cudaStreamDestroy = libcudart.cudaStreamDestroy
        cudaStreamDestroy.argtypes = [cudaStream_t]
        cudaStreamDestroy.restype = ctypes.c_int

        # Cache on the function object so later calls skip this setup
        get_external_torch_stream._libcudart = libcudart
        get_external_torch_stream._cudaStream_t = cudaStream_t
        get_external_torch_stream._cudaStreamCreateWithFlags = cudaStreamCreateWithFlags
        get_external_torch_stream._cudaStreamDestroy = cudaStreamDestroy
        get_external_torch_stream._initialized = True

    # --- choose device ---
    if device is None:
        device_index = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device_index = device.index
    else:
        device_index = int(device)

    torch.cuda.set_device(device_index)

    # --- create raw cudaStream_t ---
    cudaStream_t = get_external_torch_stream._cudaStream_t
    raw_stream = cudaStream_t()

    err = get_external_torch_stream._cudaStreamCreateWithFlags(
        ctypes.byref(raw_stream),
        ctypes.c_uint(flags),
    )
    if err != 0:
        raise RuntimeError(f"cudaStreamCreateWithFlags failed with error code {err}")

    ptr = raw_stream.value
    if ptr is None:
        raise RuntimeError("cudaStreamCreateWithFlags returned NULL stream")

    # --- wrap it in a PyTorch stream (ExternalStream) ---
    stream = torch.cuda.ExternalStream(ptr, device=device_index)
    # Stash the raw handle so you can destroy it later if you want
    stream._cuda_raw_stream = raw_stream

    return stream


def destroy_external_torch_stream(stream):
    """Destroy the cudaStream_t created by get_external_torch_stream."""
    if not hasattr(get_external_torch_stream, "_cudaStreamDestroy"):
        raise RuntimeError("CUDA runtime not initialized")

    raw = getattr(stream, "_cuda_raw_stream", None)
    if raw is None:
        raise ValueError("Stream has no associated raw CUDA handle")

    # make sure all work on this stream is done
    stream.synchronize()

    err = get_external_torch_stream._cudaStreamDestroy(raw)
    if err != 0:
        raise RuntimeError(f"cudaStreamDestroy failed with error code {err}")
