import queue
import sys
import threading
from pathlib import Path
from typing import Tuple


def _ensure_repo_on_path():
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _torch_cuda():
    try:
        import torch  # type: ignore
    except Exception:
        print("SKIP: torch not available", file=sys.stderr)
        sys.exit(0)
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        print("SKIP: CUDA not available", file=sys.stderr)
        sys.exit(0)
    return torch


def _stream_tensor_cls():
    _ensure_repo_on_path()
    from hmlib.utils.gpu import StreamTensorX  # local import to ensure module is present

    return StreamTensorX


def _pattern(torch, shape: Tuple[int, int, int]):
    channels, height, width = shape
    base = torch.arange(float(channels * height * width), dtype=torch.float32)
    return base.reshape(channels, height, width)


def should_stream_tensorx_checkpoint_chain():
    torch = _torch_cuda()
    StreamTensorX = _stream_tensor_cls()

    device = torch.device("cuda")
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_c = torch.cuda.Stream()

    shape = (3, 256, 320)
    pattern = _pattern(torch, shape)
    gpu_pattern = pattern.to(device)

    with torch.cuda.stream(stream_a):
        tensor = torch.empty_like(gpu_pattern)
        tensor.copy_(gpu_pattern, non_blocking=True)
        torch.cuda._sleep(100_000)
        tensor.add_(5.0)
        wrapped = StreamTensorX(tensor)

    with torch.cuda.stream(stream_b):
        wrapped.checkpoint()
        wrapped.ref().mul_(2.0)
        torch.cuda._sleep(100_000)
        wrapped.checkpoint()

    with torch.cuda.stream(stream_c):
        patch = wrapped[:, :64, :96]
        patch_mean = patch.float().mean().item()
        wrapped.ref().add_(3.0)
        wrapped.checkpoint()

    expected_before_bias = (pattern + 5.0) * 2.0
    expected = expected_before_bias + 3.0
    expected_patch_mean = float(expected_before_bias[:, :64, :96].mean())
    result = wrapped.get().cpu()

    assert torch.allclose(result, expected, atol=0.0, rtol=0.0)
    assert abs(patch_mean - expected_patch_mean) <= 0.1
    assert wrapped.ready()


def should_stream_tensorx_wait_and_recheckpoint():
    torch = _torch_cuda()
    StreamTensorX = _stream_tensor_cls()

    device = torch.device("cuda")
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    host = torch.linspace(0.0, 1.0, steps=1024, dtype=torch.float32).reshape(32, 32)

    with torch.cuda.stream(stream_a):
        tensor = host.to(device)
        tensor.sin_()
        wrapped = StreamTensorX(tensor)

    assert not wrapped.ready()

    with torch.cuda.stream(stream_b):
        waited = wrapped.wait()
        waited.cos_()
        wrapped.checkpoint()

    result = wrapped.get().cpu()
    expected = torch.cos(torch.sin(host))

    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)
    assert wrapped.ready()


def should_stream_tensorx_threaded_pipeline():
    torch = _torch_cuda()
    StreamTensorX = _stream_tensor_cls()

    device = torch.device("cuda")
    shape = (3, 64, 64)
    base_cpu = _pattern(torch, shape)
    base_gpu_template = base_cpu.to(device)

    num_items = 100
    start_value = lambda idx: 1.0 + idx * 0.25  # noqa: E731

    stage_queues = [queue.Queue() for _ in range(5)]
    results_queue: queue.Queue = queue.Queue()

    def stage0_worker():
        stream = torch.cuda.Stream()
        while True:
            payload = stage_queues[0].get()
            if payload is None:
                stage_queues[1].put(None)
                stage_queues[0].task_done()
                break
            idx, start_val = payload
            with torch.cuda.stream(stream):
                tensor = base_gpu_template.clone()
                tensor.add_(start_val)
                torch.cuda._sleep(40_000)
                wrapped = StreamTensorX(tensor, auto_checkpoint=False)
                wrapped.checkpoint(stream)
            stage_queues[1].put((idx, wrapped))
            stage_queues[0].task_done()

    def make_transform_worker(in_idx: int, out_idx: int, op, sleep_cycles: int):
        stream = torch.cuda.Stream()

        def worker():
            while True:
                payload = stage_queues[in_idx].get()
                if payload is None:
                    if out_idx >= 0:
                        stage_queues[out_idx].put(None)
                    else:
                        results_queue.put(None)
                    stage_queues[in_idx].task_done()
                    break
                idx, tensor = payload
                with torch.cuda.stream(stream):
                    data = tensor.wait(stream)
                    op(data, idx)
                    torch.cuda._sleep(sleep_cycles)
                    tensor.checkpoint(stream)
                if out_idx >= 0:
                    stage_queues[out_idx].put((idx, tensor))
                else:
                    results_queue.put((idx, tensor.get().cpu()))
                stage_queues[in_idx].task_done()

        return worker

    def op_stage1(data, _idx):
        data.mul_(2.0)

    def op_stage2(data, idx):
        data.add_(idx * 0.5)

    def op_stage3(data, idx):
        scale = float((idx % 7) + 1)
        data.mul_(scale)

    def op_stage4(data, _idx):
        data.add_(42.0)

    workers = [
        threading.Thread(target=stage0_worker, daemon=False),
        threading.Thread(target=make_transform_worker(1, 2, op_stage1, 35_000), daemon=False),
        threading.Thread(target=make_transform_worker(2, 3, op_stage2, 30_000), daemon=False),
        threading.Thread(target=make_transform_worker(3, 4, op_stage3, 25_000), daemon=False),
        threading.Thread(target=make_transform_worker(4, -1, op_stage4, 20_000), daemon=False),
    ]

    for worker in workers:
        worker.start()

    for idx in range(num_items):
        stage_queues[0].put((idx, start_value(idx)))
    stage_queues[0].put(None)

    results = {}
    finished = False
    while not finished or len(results) < num_items:
        item = results_queue.get()
        if item is None:
            finished = True
            continue
        idx, tensor_cpu = item
        results[idx] = tensor_cpu

    for worker in workers:
        worker.join()

    assert len(results) == num_items

    for idx, tensor_cpu in results.items():
        expected = base_cpu.clone()
        expected.add_(start_value(idx))
        expected.mul_(2.0)
        expected.add_(idx * 0.5)
        expected.mul_(float((idx % 7) + 1))
        expected.add_(42.0)
        assert torch.allclose(tensor_cpu, expected, atol=1e-5, rtol=1e-5)


def should_stream_tensorx_checkpoint_cross_stream_wait():
    torch = _torch_cuda()
    StreamTensorX = _stream_tensor_cls()

    device = torch.device("cuda")
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    host = torch.linspace(0.0, 1.0, steps=256, dtype=torch.float32).reshape(16, 16)

    with torch.cuda.stream(stream_a):
        tensor = host.to(device)
        tensor.mul_(2.0)
        torch.cuda._sleep(200_000)
        wrapped = StreamTensorX(tensor)

    with torch.cuda.stream(stream_b):
        wrapped.checkpoint(stream_b)
        wrapped.ref().add_(3.0)
        torch.cuda._sleep(50_000)
        wrapped.checkpoint(stream_b)

    result = wrapped.get().cpu()
    expected = host * 2.0 + 3.0

    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)


def should_stream_checkpoint_accept_wrapped_stream_tensor():
    torch = _torch_cuda()
    _ensure_repo_on_path()
    from hmlib.utils.gpu import StreamCheckpoint, StreamTensor

    device = torch.device("cuda")
    stream = torch.cuda.Stream()

    host = torch.arange(0, 64, dtype=torch.float32).reshape(8, 8)

    with torch.cuda.stream(stream):
        tensor = host.to(device)
        tensor.add_(5.0)
        wrapped = StreamTensor(tensor)
        wrapped.checkpoint(stream)

    checkpointed = StreamCheckpoint(wrapped)
    fetched = checkpointed.get().cpu()

    assert torch.allclose(fetched, host + 5.0, atol=1e-6, rtol=1e-6)
