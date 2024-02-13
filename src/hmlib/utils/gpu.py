import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Set

from hmlib.utils.utils import create_queue


class GpuAllocator:
    def __init__(self, gpus: List[int]):
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

    def get(self):
        if isinstance(self._tensor, StreamTensor):
            return self._tensor.get()
        if self._stream is not None:
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
        return self._tensor

    def to(self, *args, **kwargs):
        assert False and "Not implemented"

    def __len__(self):
        return self._tensor.shape[0]


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
        if contiguous:
            tensor = tensor.contiguous()
        super(StreamTensorToDevice, self).__init__(tensor=tensor, stream=stream)


class StreamTensorToDtype(StreamTensor):
    def __init__(
        self, tensor: torch.Tensor, dtype: torch.dtype, contiguous: bool = False
    ):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        super(StreamTensorToDtype, self).__init__()
        assert tensor.dtype != dtype

        if isinstance(tensor, StreamTensor):
            self._stream = tensor._stream
            with torch.cuda.stream(stream=self._stream):
                self._tensor = tensor.ref().to(dtype=dtype, non_blocking=True)
        else:
            self._stream = torch.cuda.Stream(tensor.device)
            with torch.cuda.stream(stream=self._stream):
                self._tensor = tensor.to(dtype=dtype, non_blocking=True)
        if contiguous:
            self._tensor = self._tensor.contiguous()


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
