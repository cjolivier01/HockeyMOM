import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_replay_cuda_graph_outputs():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor) -> torch.Tensor:
        return (x * 2) + 1

    x0 = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    cg = CudaGraphCallable(fn, (x0,), warmup=1, name="test_fn")

    y0 = cg(x0).detach().cpu()
    assert torch.allclose(y0, fn(x0).detach().cpu())

    x1 = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4) + 10
    y1 = cg(x1).detach().cpu()
    assert torch.allclose(y1, fn(x1).detach().cpu())
    assert cg.stats.captures == 1
    assert cg.stats.replays >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_return_replay_outputs_with_independent_storage():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    x0 = torch.arange(8, device="cuda", dtype=torch.float32)
    cg = CudaGraphCallable(fn, (x0,), warmup=0, name="independent_storage")

    y0 = cg(x0)
    y0_before = y0.detach().clone()
    y1 = cg(x0 + 100)

    assert y0.data_ptr() != y1.data_ptr()
    assert torch.allclose(y0, y0_before)
    assert torch.allclose(y1, x0 + 101)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_recapture_on_signature_change():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + 3

    x0 = torch.zeros((2, 3), device="cuda", dtype=torch.float16)
    cg = CudaGraphCallable(fn, (x0,), warmup=0, name="sig_change")
    assert cg.stats.captures == 1

    _ = cg(x0)
    x1 = torch.zeros((2, 4), device="cuda", dtype=torch.float16)
    _ = cg(x1)
    assert cg.stats.captures == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_support_multiple_tensor_outputs():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor):
        return x + 1, x - 1

    x = torch.randn(8, device="cuda", dtype=torch.float32)
    cg = CudaGraphCallable(fn, (x,), warmup=0, name="multi_out")
    a, b = cg(x)
    assert torch.allclose(a, x + 1)
    assert torch.allclose(b, x - 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_reuse_shared_capture_stream_per_device():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    x0 = torch.randn(8, device="cuda", dtype=torch.float32)
    cg0 = CudaGraphCallable(fn, (x0,), warmup=0, name="shared_stream_a")
    cg1 = CudaGraphCallable(fn, (x0,), warmup=0, name="shared_stream_b")

    assert cg0._capture_stream is not None
    assert cg1._capture_stream is not None
    assert cg0._capture_stream is cg1._capture_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_replay_with_fresh_inputs_from_non_default_stream():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x * 3 + 1

    x0 = torch.zeros(4096, device="cuda", dtype=torch.float32)
    cg = CudaGraphCallable(fn, (x0,), warmup=0, name="non_default_stream")

    producer_stream = torch.cuda.Stream()
    caller_stream = torch.cuda.Stream()
    x = torch.empty_like(x0)
    with torch.cuda.stream(producer_stream):
        x.fill_(7)
    with torch.cuda.stream(caller_stream):
        caller_stream.wait_stream(producer_stream)
        out = cg(x)
        observed = out + 0

    torch.cuda.current_stream().wait_stream(caller_stream)
    assert torch.allclose(observed.cpu(), torch.full_like(observed.cpu(), 22))


def should_zero_trt_nms_rows_beyond_num_valid():
    from hmlib.utils.nms import TrtBatchedNMS

    boxes = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [100.0, 100.0, 10.0, 10.0],
            [20.0, 20.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.7, 99.0, float("nan")], dtype=torch.float32)
    labels = torch.tensor([2, 4232601635951843710, -5], dtype=torch.long)

    out_b, out_s, out_l = TrtBatchedNMS._zero_padded_outputs(
        boxes,
        scores,
        labels,
        torch.tensor([1], dtype=torch.int32),
    )

    assert torch.equal(out_b[0], boxes[0])
    assert torch.equal(out_s[:1], scores[:1])
    assert torch.equal(out_l[:1], labels[:1])
    assert torch.equal(out_b[1:], torch.zeros_like(out_b[1:]))
    assert torch.equal(out_s[1:], torch.zeros_like(out_s[1:]))
    assert torch.equal(out_l[1:], torch.zeros_like(out_l[1:]))


def should_prune_detections_static_ignore_rows_beyond_num_detections():
    from hmlib.tracking_utils.segm_boundaries import SegmBoundaries

    height, width = 64, 64
    mask = torch.ones((height, width), dtype=torch.bool)
    centroid = torch.tensor([width / 2, height / 2], dtype=torch.float32)
    segm = SegmBoundaries(segment_mask=mask, centroid=centroid, max_detections_in_mask=3)

    det_bboxes = torch.tensor(
        [
            [10.0, 10.0, 20.0, 20.0],
            [11.0, 11.0, 21.0, 21.0],
            [12.0, 12.0, 22.0, 22.0],
            [13.0, 13.0, 23.0, 23.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 4232601635951843710, 4565255099436772587])
    scores = torch.tensor([0.8, 0.7, 100.0, 99.0], dtype=torch.float32)

    out_b, out_l, out_s, num_valid = segm.prune_detections_static(
        det_bboxes,
        labels,
        scores,
        num_detections=torch.tensor([2], dtype=torch.int32),
    )

    assert isinstance(num_valid, torch.Tensor)
    assert int(num_valid.item()) == 2
    assert out_b.shape == (3, 4)
    assert out_l.shape == (3,)
    assert out_s.shape == (3,)
    assert set(out_l[:2].tolist()) == {0, 1}
    assert 4232601635951843710 not in out_l.tolist()
    assert torch.equal(out_b[2:], torch.zeros_like(out_b[2:]))
    assert torch.equal(out_s[2:], torch.zeros_like(out_s[2:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_prune_detections_static_with_fixed_outputs():
    from hmlib.tracking_utils.segm_boundaries import SegmBoundaries

    H, W = 64, 64
    mask = torch.ones((H, W), device="cuda", dtype=torch.bool)
    centroid = torch.tensor([W / 2, H / 2], device="cuda", dtype=torch.float32)
    segm = SegmBoundaries(segment_mask=mask, centroid=centroid, max_detections_in_mask=5)

    det_bboxes = torch.tensor(
        [
            [10, 10, 20, 20],
            [11, 11, 21, 21],
            [12, 12, 22, 22],
            [13, 13, 23, 23],
            [14, 14, 24, 24],
            [15, 15, 25, 25],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    labels = torch.arange(det_bboxes.shape[0], device="cuda", dtype=torch.int64)
    scores = torch.linspace(0.1, 0.6, det_bboxes.shape[0], device="cuda", dtype=torch.float32)

    out_b, out_l, out_s, num_valid = segm.prune_detections_static(det_bboxes, labels, scores)
    assert out_b.shape == (5, 4)
    assert out_l.shape == (5,)
    assert out_s.shape == (5,)
    assert isinstance(num_valid, torch.Tensor)
    assert int(num_valid.detach().cpu()) == 5
