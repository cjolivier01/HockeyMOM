import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Set

from hmlib.utils.utils import create_queue


class GpuAllocator:
    def __init__(self, gpus: List[int]):
        if gpus is None:
            gpus = [i for i in range(torch.cuda.device_count())]
        gpu_count = min(torch.cuda.device_count(), len(gpus))
        self._gpus = gpus[: gpu_count + 1]
        self._used_gpus = set()

    def allocate_modern(self):
        index, caps = get_gpu_with_highest_compute_capability(
            allowed_gpus=self._gpus, disallowed_gpus=self._used_gpus
        )
        if index is not None:
            self._used_gpus.add(index)
        return index

    def allocate_fast(self):
        index, caps = get_gpu_with_most_multiprocessors(
            allowed_gpus=self._gpus, disallowed_gpus=self._used_gpus
        )
        if index is not None:
            self._used_gpus.add(index)
        return index

    def free_count(self):
        return len(self._gpus) - len(self._used_gpus)


class CachedIterator:
    def __init__(self, iterator, cache_size: int = 2, pre_callback_fn: callable = None):
        self._iterator = iterator
        self._q = create_queue(mp=False) if cache_size else None
        self._pre_callback_fn = pre_callback_fn
        for _ in range(cache_size):
            try:
                item = next(self._iterator)
                if self._pre_callback_fn is not None:
                    item = self._pre_callback_fn(item)
                self._q.put(item)
            except StopIteration:
                self._q.put(None)
                break

    def __iter__(self):
        return self

    def __next__(self):
        if self._q is None:
            item = next(self._iterator)
            if self._pre_callback_fn is not None:
                item = self._pre_callback_fn(item)
        else:
            item = self._q.get()
            if item is None:
                raise StopIteration
            try:
                item = next(self._iterator)
                if self._pre_callback_fn is not None:
                    item = self._pre_callback_fn(item)
                self._q.put(item)
            except StopIteration:
                self._q.put(None)
        return item


class StreamTensor:
    def __init__(self, tensor: torch.Tensor = None, stream: torch.cuda.Stream = None):
        self._tensor = tensor
        self._stream = stream
        if not hasattr(self, "_event"):
            self._event = None

    def get(self):
        if isinstance(self._tensor, StreamTensor):
            return self._tensor.get()
        if self._stream is not None:
            if self._event is not None:
                with torch.cuda.stream(self._stream):
                    self._event.synchronize()
            else:
                assert False
                self._stream.synchronize()
        return self._tensor

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def device(self):
        return self._tensor.device

    @property
    def ndim(self):
        return self._tensor.ndim

    @property
    def size(self, index: int):
        return self._tensor.size(index)

    @property
    def shape(self):
        return self._tensor.shape

    def ref(self):
        if isinstance(self._tensor, StreamTensor):
            return self._tensor.ref()
        return self._tensor

    def to(self, *args, **kwargs):
        assert False and "Not implemented"

    def __len__(self):
        return self._tensor.shape[0]


class StreamCheckpoint(StreamTensor):
    def __init__(self, tensor: torch.Tensor, stream: torch.cuda.Stream):
        super(StreamCheckpoint, self).__init__(tensor=tensor, stream=stream)
        if self._stream is not None:
            with torch.cuda.stream(stream):
                self._event = torch.cuda.Event()
                self._event.record()


class StreamTensorToDevice(StreamTensor):
    def __init__(
        self, tensor: torch.Tensor, device: torch.device, contiguous: bool = False
    ):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.device == device:
            stream = None
        else:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream=stream):
                tensor = tensor.to(device, non_blocking=True)
                self._event = torch.cuda.Event()
                self._event.record()
        if contiguous:
            tensor = tensor.contiguous()
        super(StreamTensorToDevice, self).__init__(tensor=tensor, stream=stream)


class StreamTensorToDtype(StreamTensor):
    def __init__(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        contiguous: bool = False,
        scale_down_factor: float = None,
    ):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        super(StreamTensorToDtype, self).__init__()
        assert tensor.dtype != dtype

        if isinstance(tensor, StreamTensor):
            self._stream = tensor._stream
            with torch.cuda.stream(stream=self._stream):
                self._tensor = tensor.ref().to(dtype=dtype, non_blocking=True)
                if scale_down_factor and scale_down_factor != 1:
                    self._tensor /= scale_down_factor
                if contiguous:
                    self._tensor = self._tensor.contiguous()
                self._event = torch.cuda.Event()
                self._event.record()
        else:
            self._stream = torch.cuda.Stream(tensor.device)
            with torch.cuda.stream(stream=self._stream):
                self._tensor = tensor.to(dtype=dtype, non_blocking=True)
                if contiguous:
                    self._tensor = self._tensor.contiguous()
                self._event = torch.cuda.Event()
                self._event.record()


def get_gpu_capabilities():
    if not torch.cuda.is_available():
        return None
    num_gpus = torch.cuda.device_count()
    gpu_info = []
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)
        gpu_info.append(
            {
                "name": properties.name,
                "index": i,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory": properties.total_memory
                / (1024**3),  # Convert bytes to GB
                "properties": properties,
            }
        )
    return gpu_info


def get_gpu_with_highest_compute_capability(
    allowed_gpus: List[int] = None,
    disallowed_gpus: Union[List[int], Set[int]] = None,
) -> Tuple[int, Dict]:
    gpus = get_gpu_capabilities()
    if gpus is None:
        return None
    sorted_gpus = sorted(gpus, key=lambda x: float(x["compute_capability"]))
    for _, gpu in enumerate(reversed(sorted_gpus)):
        index = gpu["index"]
        if allowed_gpus is not None and index not in allowed_gpus:
            continue
        if disallowed_gpus is not None and index in disallowed_gpus:
            continue
        return index, gpu
    return None, None


def get_gpu_with_most_multiprocessors(
    allowed_gpus: List[int] = None,
    disallowed_gpus: Union[List[int], Set[int]] = None,
) -> Tuple[int, Dict]:
    gpus = get_gpu_capabilities()
    if gpus is None:
        return None
    sorted_gpus = sorted(
        gpus, key=lambda x: float(x["properties"].multi_processor_count)
    )
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


def _check_is(dev: torch.device, flag: bool):
    if dev is not None:
        assert isinstance(dev, torch.device)
    assert (flag and dev is not None) or (not flag and dev is None)


def select_gpus(
    allowed_gpus: List[int],
    rank: int = None,
    gpu_allocator: GpuAllocator = None,
    inference: bool = True,
    is_stitching: bool = True,
    is_detecting: bool = True,
    is_encoding: bool = True,
    is_camera: bool = True,
):
    #
    # BEGIN GPU SELECTION
    #
    if gpu_allocator is None:
        gpu_allocator = GpuAllocator(gpus=allowed_gpus)

    stitching_device = None
    video_encoding_device = None
    camera_device = torch.device("cpu")
    if inference:
        if is_encoding:
            video_encoding_device = torch.device(
                "cuda", gpu_allocator.allocate_modern()
            )
        if gpu_allocator.free_count():
            detection_device = torch.device("cuda", gpu_allocator.allocate_fast())
        else:
            detection_device = video_encoding_device
            if is_encoding:
                video_encoding_device = torch.device("cpu")

        if is_stitching:
            if gpu_allocator.free_count():
                stitching_device = torch.device("cuda", gpu_allocator.allocate_fast())
            else:
                stitching_device = detection_device
    else:
        if rank is not None:
            for i in range(rank):
                _ = gpu_allocator.allocate_fast()
            detection_device = torch.device("cuda", gpu_allocator.allocate_fast())
        if is_stitching:
            stitching_device = detection_device
    #
    # END GPU SELECTION
    #
    gpus = dict()
    _check_is(stitching_device, is_stitching)
    _check_is(detection_device, is_detecting)
    _check_is(stitching_device, is_stitching)
    _check_is(camera_device, is_camera)
    _check_is(video_encoding_device, is_encoding)
    if stitching_device is not None:
        gpus["stitching"] = stitching_device
    if detection_device is not None:
        gpus["detection"] = detection_device
    if camera_device is not None:
        gpus["camera"] = camera_device
    if video_encoding_device is not None:
        gpus["encoder"] = video_encoding_device
    return gpus
