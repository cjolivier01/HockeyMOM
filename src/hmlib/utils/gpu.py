import torch
import time
import queue
import numpy as np
from threading import Thread
from typing import Any, Dict, List, Tuple, Union, Set, Optional

from hmlib.utils.utils import create_queue


class GpuAllocator:
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

        gpu_info = get_gpu_capabilities()
        for i in range(len(gpu_info)):
            print(f"GPU {i}: {gpu_info[i]['name']}")

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


class SimpleCachedIterator:
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
            result_item = next(self._iterator)
            if self._pre_callback_fn is not None:
                result_item = self._pre_callback_fn(result_item)
        else:
            result_item = self._q.get()
            if result_item is None:
                raise StopIteration
            try:
                cached_item = next(self._iterator)
                if self._pre_callback_fn is not None:
                    cached_item = self._pre_callback_fn(cached_item)
                self._q.put(cached_item)
            except StopIteration:
                self._q.put(None)
        return result_item


class ThreadedCachedIterator:
    def __init__(self, iterator, cache_size: int = 2, pre_callback_fn: callable = None):
        self._iterator = iterator
        self._q = create_queue(mp=False) if cache_size else None
        self._pre_callback_fn = pre_callback_fn
        self._save_cache_size = cache_size
        self._pull_queue_to_worker: queue.Queue = queue.Queue()
        self._pull_thread = Thread(target=self._pull_worker)
        for i in range(cache_size):
            self._pull_queue_to_worker.put("ok")

        # Finally, start the worker thread
        self._pull_thread.start()

    def _pull_worker(self):
        try:
            while True:
                msg = self._pull_queue_to_worker.get()
                if msg is None:
                    raise StopIteration()
                item = next(self._iterator)
                if self._pre_callback_fn is not None:
                    item = self._pre_callback_fn(item)
                self._q.put(item)
        except StopIteration:
            self._q.put(None)
            return
        except Exception as ex:
            print(ex)
            print(f"ThreadedCachedIterator exiting due to exception: {ex}")
            raise

    def _stop(self):
        if self._pull_thread is not None:
            self._pull_queue_to_worker.put(None)
            self._pull_thread.join()
            self._pull_thread = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._q is None:
            result_item = next(self._iterator)
            if self._pre_callback_fn is not None:
                result_item = self._pre_callback_fn(result_item)
        else:
            get_next_cached_start = time.time()
            result_item = self._q.get()
            if result_item is None:
                raise StopIteration()
            get_next_cached_duration = time.time() - get_next_cached_start
            # if get_next_cached_duration > 10 / 1000:
            #     print(
            #         f"Waited {get_next_cached_duration * 1000} ms "
            #         f"for the next cached item for cache size of {self._save_cache_size}"
            #     )
            self._pull_queue_to_worker.put("ok")
        return result_item

    def __del__(self):
        self._stop()


# CachedIterator = ThreadedCachedIterator
CachedIterator = SimpleCachedIterator


class StreamTensorBase:
    def size(self, index: int) -> int:
        assert False and "Not implemented"


class StreamTensor(StreamTensorBase):
    def __init__(
        self,
        tensor: Union[torch.Tensor, StreamTensorBase],
        stream: Union[torch.cuda.Stream, None] = None,
        event: Optional[Union[torch.cuda.Event, None]] = None,
        owns_stream: Optional[Union[bool, None]] = None,
        verbose: Optional[bool] = True,
        print_thresh: Optional[float] = 0.001,
    ):
        self._tensor = tensor
        self._stream = stream
        self._event = event
        self._print_thresh = print_thresh
        if not isinstance(tensor, StreamTensor):
            # assert owns_stream is None
            self._owns_stream = owns_stream
        else:
            self._owns_stream = owns_stream
        self._sync_duraton = None
        self._verbose = verbose

    # def clone(self):
    #     #assert self._owns_stream
    #     with torch.cuda.stream(self._stream):
    #         return StreamTensor(
    #             tensor=self.ref().clone(),
    #             stream=self._stream,
    #             event=torch.cuda.Event(),
    #             owns_stream=self._owns_stream,
    #         )

    def get(self):
        if self._stream is not None:
            if self._event is not None:
                with torch.cuda.stream(self._stream):
                    start = time.time()
                    self._event.synchronize()
                    self._sync_duraton = time.time() - start
            else:
                assert False
                self._stream.synchronize()
        elif self._event is not None:
            start = time.time()
            self._event.synchronize()
            self._sync_duraton = time.time() - start
        if (
            self._verbose
            and self._sync_duraton is not None
            and self._sync_duraton > self._print_thresh
        ):
            # print(
            #     f"Syncing tensor with shape {self.shape} took {self._sync_duraton * 1000} ms"
            # )
            pass
        return self._tensor

    @property
    def sync_duration(self):
        if self._sync_duraton is None:
            return -1
        return self._sync_duraton

    @property
    def stream(self):
        if isinstance(self._tensor, StreamTensor):
            return self._tensor.stream
        if self._owns_stream:
            return self._stream
        else:
            return None

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def device(self):
        return self._tensor.device

    @property
    def ndim(self):
        return self._tensor.ndim

    def size(self, index: int) -> int:
        return self._tensor.size(index)

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def owns_stream(self):
        if isinstance(self._tensor, StreamTensor):
            return self._tensor.owns_stream
        return self._owns_stream

    def ref(self):
        if isinstance(self._tensor, StreamTensor):
            return self._tensor.ref()
        return self._tensor

    def _set_ref(self, tensor: torch.Tensor):
        """
        Set tensor to some modified version of itself that does not need compute, such as permute()
        """
        if isinstance(self._tensor, StreamTensor):
            self._tensor._set_ref(tensor)
        else:
            self._tensor = tensor

    def __truediv__(self, other):
        assert isinstance(other, (int, float))
        assert self._owns_stream
        with torch.cuda.stream(self._stream):
            self._set_ref(self.ref() / other)
            self._event = torch.cuda.Event()
            self._event.record()

    def permute(self, *args):
        self._set_ref(self.ref().permute(*args))
        return self

    def to(self, *args, **kwargs):
        assert False and "Not implemented"

    def __len__(self):
        return self._tensor.shape[0]


class StreamCheckpoint(StreamTensor):
    def __init__(
        self,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream,
        event: Optional[Union[torch.cuda.Event, None]] = None,
        owns_stream: Optional[Union[bool, None]] = None,
        verbose: Optional[bool] = True,
        print_thresh: Optional[float] = 0.001,
    ):
        super(StreamCheckpoint, self).__init__(
            tensor=tensor,
            stream=stream,
            event=event,
            owns_stream=owns_stream,
            verbose=verbose,
            print_thresh=print_thresh,
        )
        if self._stream is not None:
            with torch.cuda.stream(stream):
                self._event = torch.cuda.Event()
                self._event.record()
        else:
            if tensor.device.type == "cuda":
                self._event = torch.cuda.Event()
                self._event.record()


class StreamTensorToDevice(StreamTensor):
    def __init__(
        self,
        tensor: torch.Tensor,
        device: torch.device,
        contiguous: bool = False,
        stream: Union[None, torch.cuda.Stream] = None,
        verbose: bool = True,
    ):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.device != device:
            if stream is None:
                stream = torch.cuda.Stream(
                    device=device if device.type == "cuda" else tensor.device
                )
            with torch.cuda.stream(stream=stream):
                tensor = tensor.to(device, non_blocking=True)
                self._event = torch.cuda.Event()
                self._event.record()
        if contiguous:
            tensor = tensor.contiguous()
        super(StreamTensorToDevice, self).__init__(
            tensor=tensor, stream=stream, event=self._event, verbose=verbose
        )


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
        super(StreamTensorToDtype, self).__init__(tensor=tensor)
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


def async_to(
    tensor: Union[torch.tensor, StreamTensor],
    device: Union[torch.device, str, None] = None,
    dtype: Union[torch.dtype, None] = None,
) -> StreamTensor:
    assert device is not None or dtype is not None
    if isinstance(tensor, StreamTensor):
        assert tensor.owns_stream
        stream = tensor.stream
        with torch.cuda.stream(stream):
            if device is not None:
                assert (
                    device == tensor.device
                )  # Need to handle this case, do we need to sync first?
                tensor._tensor = tensor.ref().to(device=device, non_blocking=True)
                tensor._event = torch.cuda.Event()
                tensor._event.record()
            if dtype is not None:
                tensor._tensor = tensor.ref().to(dtype=dtype, non_blocking=True)
                tensor._event = torch.cuda.Event()
                tensor._event.record()
        return tensor
    else:
        # How do we do this across two devices?
        stream = torch.cuda.Stream(device if device.type != "cpu" else tensor.device)
        with torch.cuda.stream(stream):
            if device is not None:
                tensor = tensor.to(device=device, non_blocking=True)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            return StreamCheckpoint(
                tensor=tensor, stream=stream, event=event, owns_stream=True
            )


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


def get_gpu_with_most_multiprocessors(
    allowed_gpus: Union[List[int], None] = None,
    disallowed_gpus: Union[List[int], Set[int], None] = None,
) -> Tuple[Union[int, None], Union[Dict, None]]:
    gpus = get_gpu_capabilities()
    if gpus is None:
        return None, None
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
):
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
            detection_device = multipose_device
        else:
            if True:
                if is_stitching:
                    if gpu_allocator.free_count():
                        stitching_device = torch.device(
                            "cuda", gpu_allocator.allocate_fast()
                        )
                    else:
                        stitching_device = detection_device

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
            if gpu_allocator.free_count():
                video_encoding_device = torch.device(
                    "cuda", gpu_allocator.allocate_modern()
                )
            else:
                video_encoding_device = (
                    detection_device if not multipose_device else multipose_device
                )

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
    if stitching_device is not None:
        gpus["stitching"] = stitching_device
    if detection_device is not None:
        gpus["detection"] = detection_device
    if multipose_device is not None:
        gpus["multipose"] = multipose_device
    if camera_device is not None:
        gpus["camera"] = camera_device
    if video_encoding_device is not None:
        gpus["encoder"] = video_encoding_device
    return gpus
